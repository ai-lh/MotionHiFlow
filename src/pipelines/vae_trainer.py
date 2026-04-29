import torch
from os.path import join as pjoin
from src.utils.interp import interpolate

import logging
from collections import OrderedDict, defaultdict
from src.utils import seed_everything

import os
from tqdm import tqdm
from einops import rearrange

def infinite_loader(loader):
    while True:
        for data in loader:
            yield data

def pool_motion(motion, ratio, mode='linear', num_joints=22, back_to_original=False):
    # motion: [B, T, D]
    root, root_h, ric, rot, vel, contact = torch.split(motion, [3, 1, 3 * (num_joints - 1), 6 * (num_joints - 1), 3 * num_joints, 4], dim=-1) 
    def pool_latent(latent, type='none'):
        latent = rearrange(latent, 'b t d -> b d t')
        if type == 'cum-diff':
            latent = latent.cumsum(dim=-1)
        round_num = int(ratio * latent.shape[-1])
        original_num = latent.shape[-1]
        latent = torch.nn.functional.interpolate(latent, size=round_num, mode=mode)
        if back_to_original:
            latent = torch.nn.functional.interpolate(latent, size=original_num, mode=mode)
        if type == 'cum-diff':
            latent = latent.diff(dim=-1, prepend=torch.zeros_like(latent[..., :1]))
        latent = rearrange(latent, 'b d t -> b t d')
        return latent
    motion = torch.cat([
        pool_latent(root, 'cum-diff'),
        pool_latent(root_h),
        pool_latent(ric),
        pool_latent(rot),
        pool_latent(vel, 'cum-diff'),
        pool_latent(contact),
    ], dim=-1)
    return motion

class VAEtrainer:
    def __init__(self, model, logger, optimizer, lr_scheduler, cfg):
        self.device = f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu"
        self.model = model.to(self.device)
        self.logger = logger
        self.opt_vae_model = optimizer
        self.scheduler = lr_scheduler
        self.cfg = cfg

        match cfg.recons_loss:
            case 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            case 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()
            case 'l2':
                self.l1_criterion = torch.nn.MSELoss()
            # case 'huber':
            #     self.l1_criterion = torch.nn.HuberLoss()
            case _:
                raise ValueError(f"Invalid reconstruction loss: {cfg.recons_loss}")
        
        self.best_fid = 1e3
        self.best_mpjpe = 1e3


    def forward(self, batch_data):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_dict, z = self.model(motions, return_latent=True)
        loss_pool = 0
        loss = 0

        if hasattr(self.cfg, "pool_latent") and self.cfg.pool_latent != 'none':
            minibatch_size = motions.shape[0] // 4
            sub_z = z[:minibatch_size]
            ratio = torch.randint(int(z.shape[1] * 0.3), z.shape[1], (1,)).item() / z.shape[1]
            z_d = interpolate(sub_z, ratio, type=self.cfg.interpolation_type, mode=self.cfg.pool_latent)
            if self.cfg.pool_up:
                z_u = interpolate(z_d, 1/ratio, type=self.cfg.interpolation_type, mode=self.cfg.pool_latent)
                z = torch.cat([z_u, z[minibatch_size:]], dim=0)
                sub_pred_m = self.model.decode(z_u)
                pred_motion = torch.cat([sub_pred_m, pred_motion[minibatch_size:]], dim=0)
                sub_motion = pool_motion(motions[:minibatch_size], ratio, mode=self.cfg.pool_latent, num_joints=self.cfg.data.meta.joints)
                sub_motion = pool_motion(sub_motion, 1/ratio, mode=self.cfg.pool_latent, num_joints=self.cfg.data.meta.joints)
                motions = torch.cat([sub_motion, motions[minibatch_size:]], dim=0)
            else:
                sub_pred_m = self.model.decode(z_d)

                sub_motion = pool_motion(motions[:minibatch_size], ratio, mode=self.cfg.pool_latent, num_joints=self.cfg.data.meta.joints)

                loss_dict["loss_pool"] = self.l1_criterion(sub_pred_m, sub_motion)
                loss += loss_dict["loss_pool"] * self.cfg.lambda_pool
                pred_motion = pred_motion[minibatch_size:]
                motions = motions[minibatch_size:]

        self.motions = motions
        self.pred_motion = pred_motion

        num_joint = self.cfg.data.meta.joints
        if motions.shape[-1] == 12 * num_joint - 1:
            root, ric, rot, vel, contact = motions.split([4, 3*(num_joint-1), 6*(num_joint-1), 3*num_joint, 4], dim=-1)
            pred_root, pred_ric, pred_rot, pred_vel, pred_contact = pred_motion.split([4, 3*(num_joint-1), 6*(num_joint-1), 3*num_joint, 4], dim=-1)
        else:
            raise ValueError(f"Invalid motion dimension: {motions.shape[-1]}")

        loss_rec = self.l1_criterion(pred_motion, motions)
        loss_ric = self.l1_criterion(ric, pred_ric)
        loss_vel = self.l1_criterion(vel, pred_vel)
        loss += loss_rec + self.cfg.loss_ric * loss_ric + self.cfg.loss_vel * loss_vel + self.cfg.commit * loss_dict['loss_kl']

        loss_dict['loss'] = loss
        loss_dict['loss_rec'] = loss_rec
        loss_dict['loss_ric'] = loss_ric
        loss_dict['loss_vel'] = loss_vel
        loss_dict['loss_pool'] = loss_pool

        if torch.isnan(loss):
            print('NaN loss detected')
            import pdb; pdb.set_trace()
        return loss_dict


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vae_model.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def save(self, file_name, total_it):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.opt_vae_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'total_it': total_it,
        }
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device, weights_only=False)
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint['model'].items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        # pretrained_dict = checkpoint['model']
        missing_keys, unexpected_keys = self.model.load_state_dict(pretrained_dict, strict=False)
        assert len(unexpected_keys) == 0 and len(missing_keys) == 0, f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
        try:
            self.opt_vae_model.load_state_dict(checkpoint['optimizer']) # Optimizer
            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            logging.warning('Resume without optimizer')
        it = 0
        try:
            it = checkpoint['total_it']
        except:
            logging.warning('Resume without iteration count')
        return it

    def print_results(self, fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, mpjpe, it, split='val'):
        msg = f"--> \t Eva. Iter {it} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, MPJPE. {mpjpe:.4f}"
        self.logger.log_dict({
            f'{split}/FID': fid,
            f'{split}/Diversity Real': diversity_real,
            f'{split}/Diversity': diversity,
            f'{split}/R_precision_1_real': R_precision_real[0],
            f'{split}/R_precision_2_real': R_precision_real[1],
            f'{split}/R_precision_3_real': R_precision_real[2],
            f'{split}/R_precision_1': R_precision[0],
            f'{split}/R_precision_2': R_precision[1],
            f'{split}/R_precision_3': R_precision[2],
            f'{split}/matching_score_real': matching_score_real,
            f'{split}/matching_score_pred': matching_score_pred,
            f'{split}/MPJPE': mpjpe,
        }, it)

        if fid < self.best_fid:
            self.logger.info(f"Improved FID from {self.best_fid:.05f} to {fid}!!!")
            self.save(pjoin(self.cfg.folder.checkpoints, 'net_best_fid.tar'), it)
            self.best_fid = fid

        if mpjpe < self.best_mpjpe:
            self.logger.info(f"Improved MPJPE from {self.best_mpjpe:.05f} to {mpjpe}!!!")
            self.save(pjoin(self.cfg.folder.checkpoints, 'net_best_mpjpe.tar'), it)
            self.best_mpjpe = mpjpe


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, eval_func, plot_eval=None):
        self.model.to(self.device)

        it = 0
        if self.cfg.is_continue:
            model_dir = pjoin(self.cfg.folder.checkpoints, 'latest.tar')
            it = self.resume(model_dir)
            print("Load model iterations:%d"%(it))

        total_iters = self.cfg.max_iter if not self.cfg.DEBUG else it + 200
        logs = defaultdict(lambda: 0.0, OrderedDict())
        self.logger.info(f'Total Iters: {total_iters}')

        fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, mpjpe = eval_func(
            val_loader=eval_val_loader, net=self.model, eval_wrapper=eval_wrapper, num_joint=self.cfg.data.meta.joints)

        self.print_results(fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, mpjpe, it, split='val')

        # If training already reached total_iters, skip creating the progress bar and finish.
        if it >= total_iters:
            self.logger.info('Finish training.')
            return
        # Use tqdm's initial to set the current step (it) and total to total_iters.
        pbar = tqdm(infinite_loader(train_loader), total=total_iters, initial=it, desc="VAE Training")
        self.logger.set_progress_bar(pbar)
        tqdm_keys = ['train/loss', 'val/loss', 'lr']
        
        for batch_data in pbar:
            it += 1
            if it < self.cfg.warm_up_iter:
                self.update_lr_warm_up(it, self.cfg.warm_up_iter, self.cfg.lr)
            loss_dict = self.forward(batch_data)
            self.opt_vae_model.zero_grad()
            loss_dict['loss'].backward()
            self.opt_vae_model.step()

            if it >= self.cfg.warm_up_iter:
                self.scheduler.step()

            for k, v in loss_dict.items():
                logs[f'train/{k}'] += v
            logs['lr'] += self.opt_vae_model.param_groups[0]['lr']

            if it % self.cfg.log_every == 0:
                self.logger.log_dict({k: v / self.cfg.log_every for k, v in logs.items()}, it, tqdm_keys=tqdm_keys)
                logs = defaultdict(lambda: 0.0, OrderedDict())

            if it % self.cfg.save_latest == 0:
                self.save(pjoin(self.cfg.folder.checkpoints, 'latest.tar'), it)

            if it % self.cfg.eval_every == 0 or it == total_iters:
                print('Validation time:')
                self.model.eval()
                val_loss_dict = defaultdict(lambda: [], OrderedDict())
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        loss_dict = self.forward(batch_data)
                        for k, v in loss_dict.items():
                            val_loss_dict[k].append(v.item() if isinstance(v, torch.Tensor) else v)

                self.logger.log_dict({
                    f'val/{k}': sum(v)/len(v) for k, v in val_loss_dict.items()
                }, it, tqdm_keys=tqdm_keys)

                self.logger.info('Validation Loss: %.5f Reconstruction: %.5f, RIC: %.5f, VEL: %.5f, Commit: %.5f, Pool: %.5f' %
                    (sum(val_loss_dict['loss'])/len(val_loss_dict['loss']), sum(val_loss_dict['loss_rec'])/len(val_loss_dict['loss_rec']), 
                    sum(val_loss_dict['loss_ric'])/len(val_loss_dict['loss_ric']), sum(val_loss_dict['loss_vel'])/len(val_loss_dict['loss_vel']), sum(val_loss_dict['loss_kl'])/len(val_loss_dict['loss_kl']), sum(val_loss_dict['loss_pool'])/len(val_loss_dict['loss_pool'])))

                fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, mpjpe = eval_func(
                    val_loader=eval_val_loader, net=self.model, eval_wrapper=eval_wrapper, num_joint=self.cfg.data.meta.joints)

                self.print_results(fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, mpjpe, it, split='val')

                self.model.train()
                
                if plot_eval is not None:
                    data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()

                    save_dir = pjoin(self.cfg.folder.run, 'It%06d' % (it))
                    os.makedirs(save_dir, exist_ok=True)
                    if plot_eval is not None:
                        plot_eval(data, save_dir)
            if it >= total_iters:
                self.logger.info('Finish training.')
                break

    @staticmethod
    @torch.no_grad()
    def test(vae_model, eval_loader, eval_wrapper, cfg, eval_func, replication_times=1):
        # make things reproducible
        seed_everything(cfg.seed)

        all_metrics = defaultdict(list)
        device = f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu"

        vae_model.to(device)
        vae_model.eval()

        for _ in range(replication_times):
            fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, mpjpe = eval_func(
                val_loader=eval_loader, net=vae_model, eval_wrapper=eval_wrapper, num_joint=cfg.data.meta.joints)
            all_metrics['FID'].append(fid)
            all_metrics['Diversity Real'].append(diversity_real)
            all_metrics['Diversity'].append(diversity)
            all_metrics['R_precision_1_real'].append(R_precision_real[0])
            all_metrics['R_precision_2_real'].append(R_precision_real[1])
            all_metrics['R_precision_3_real'].append(R_precision_real[2])
            all_metrics['R_precision_1'].append(R_precision[0])
            all_metrics['R_precision_2'].append(R_precision[1])
            all_metrics['R_precision_3'].append(R_precision[2])
            all_metrics['matching_score_real'].append(matching_score_real)
            all_metrics['matching_score_pred'].append(matching_score_pred)
            all_metrics['MPJPE'].append(mpjpe)

            logging.info("Replication %d/%d: FID %.4f, Diversity Real %.4f, Diversity %.4f, R_precision_real %s, R_precision %s, matching_score_real %.4f, matching_score_pred %.4f, MPJPE %.4f" % (
                _ + 1, replication_times, fid, diversity_real, diversity, str(R_precision_real), str(R_precision), matching_score_real, matching_score_pred, mpjpe
            ))
        return all_metrics


# class LengthEstTrainer(object):
#     def __init__(self, args, estimator, text_encoder, encode_fnc):
#         self.src/utils/__pycache__cfg = args
#         self.estimator = estimator
#         self.text_encoder = text_encoder
#         self.encode_fnc = encode_fnc
#         self.device = args.device

#         if args.is_train:
#             # self.motion_dis
#             self.logger = SummaryWriter(args.log_dir)
#             self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

#     def resume(self, model_dir):
#         checkpoints = torch.load(model_dir, map_location=self.device, weights_only=False)
#         self.estimator.load_state_dict(checkpoints['estimator'])
#         # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
#         return checkpoints['epoch'], checkpoints['iter']

#     def save(self, model_dir, epoch, niter):
#         state = {
#             'estimator': self.estimator.state_dict(),
#             # 'opt_estimator': self.opt_estimator.state_dict(),
#             'epoch': epoch,
#             'niter': niter,
#         }
#         torch.save(state, model_dir)

#     @staticmethod
#     def zero_grad(opt_list):
#         for cfg in opt_list:
#             cfg.zero_grad()

#     @staticmethod
#     def clip_norm(network_list):
#         for network in network_list:
#             clip_grad_norm_(network.parameters(), 0.5)

#     @staticmethod
#     def step(opt_list):
#         for cfg in opt_list:
#             cfg.step()

#     def train(self, train_dataloader, val_dataloader):
#         self.estimator.to(self.device)
#         self.text_encoder.to(self.device)

#         self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.cfg.lr)

#         epoch = 0
#         it = 0

#         if self.cfg.is_continue:
#             model_dir = pjoin(self.cfg.folder.checkpoints, 'latest.tar')
#             epoch, it = self.resume(model_dir)

#         start_time = time.time()
#         total_iters = self.cfg.max_iter // len(train_dataloader) * len(train_dataloader) if not self.cfg.stop_immediate else self.cfg.max_iter
#         print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
#         val_loss = 0
#         min_val_loss = np.inf
#         logs = defaultdict(float)
#         while it < total_iters:
#             # time0 = time.time()
#             for i, batch_data in enumerate(train_dataloader):
#                 self.estimator.train()

#                 conds, _, m_lens = batch_data
#                 # word_emb = word_emb.detach().to(self.device).float()
#                 # pos_ohot = pos_ohot.detach().to(self.device).float()
#                 # m_lens = m_lens.to(self.device).long()
#                 text_embs = self.encode_fnc(self.text_encoder, conds, self.cfg.device).detach()
#                 # print(text_embs.shape, text_embs.device)

#                 pred_dis = self.estimator(text_embs)

#                 self.zero_grad([self.opt_estimator])

#                 gt_labels = m_lens // self.cfg.unit_length
#                 gt_labels = gt_labels.long().to(self.device)
#                 # print(gt_labels.shape, pred_dis.shape)
#                 # print(gt_labels.max(), gt_labels.min())
#                 # print(pred_dis)
#                 acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
#                 loss = self.mul_cls_criterion(pred_dis, gt_labels)

#                 loss.backward()

#                 self.clip_norm([self.estimator])
#                 self.step([self.opt_estimator])

#                 logs['loss'] += loss.item()
#                 logs['acc'] += acc.item()

#                 it += 1
#                 if it % self.cfg.log_every == 0:
#                     mean_loss = OrderedDict({'val_loss': val_loss})
#                     # self.logger.add_scalar('Val/loss', val_loss, it)

#                     for tag, value in logs.items():
#                         self.logger.add_scalar("Train/%s"%tag, value / self.cfg.log_every, it)
#                         mean_loss[tag] = value / self.cfg.log_every
#                     logs = defaultdict(float)
#                     print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

#                     if it % self.cfg.save_latest == 0:
#                         self.save(pjoin(self.cfg.folder.checkpoints, 'latest.tar'), epoch, it)

#             self.save(pjoin(self.cfg.folder.checkpoints, 'latest.tar'), epoch, it)

#             epoch += 1

#             print('Validation time:')

#             val_loss = 0
#             val_acc = 0
#             # self.estimator.eval()
#             with torch.no_grad():
#                 for i, batch_data in enumerate(val_dataloader):
#                     self.estimator.eval()

#                     conds, _, m_lens = batch_data
#                     # word_emb = word_emb.detach().to(self.device).float()
#                     # pos_ohot = pos_ohot.detach().to(self.device).float()
#                     # m_lens = m_lens.to(self.device).long()
#                     text_embs = self.encode_fnc(self.text_encoder, conds, self.cfg.device)
#                     pred_dis = self.estimator(text_embs)

#                     gt_labels = m_lens // self.cfg.unit_length
#                     gt_labels = gt_labels.long().to(self.device)
#                     loss = self.mul_cls_criterion(pred_dis, gt_labels)
#                     acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

#                     val_loss += loss.item()
#                     val_acc += acc.item()


#             val_loss = val_loss / len(val_dataloader)
#             val_acc = val_acc / len(val_dataloader)
#             print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

#             if val_loss < min_val_loss:
#                 self.save(pjoin(self.cfg.folder.checkpoints, 'finest.tar'), epoch, it)
#                 min_val_loss = val_loss
