import torch
import numpy as np
import copy
import os
from collections import defaultdict
from collections import OrderedDict
from os.path import join as pjoin
from tqdm import tqdm
from src.utils import load_model, seed_everything
import logging
    
def infinite_loader(loader):
    while True:
        for data in loader:
            yield data

class FlowTrainer:
    def __init__(self, model, logger, optimizer, lr_scheduler, cfg):
        self.model = model
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.device = f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu"
        self.vae_model = load_model(pjoin(cfg.folder.base, cfg.vae_model.name), ckpt_name=cfg.vae_model.ckpt_name)
        self.cfg = cfg

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

        self.model = self.model.to(self.device)
        self.vae_model = self.vae_model.to(self.device)
        self.ema_model = self.ema_model.to(self.device)
        
        self.best_fid = 1e3
        self.best_MM = 1e3
        self.best_R1 = 0.0

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def ema_update(self, decay=0.999):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def forward(self, batch_data):
        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, t, v, d)
        with torch.no_grad():
            motion, _ = self.vae_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # with torch.amp.autocast(device_type=self.opt.device, dtype=torch.float32):
        # with torch.autocast("cuda", dtype=torch.float32):
        loss_dict = self.model.train_forward(motion, conds, m_lens)

        return loss_dict

    def update(self, batch_data):
        loss_dict = self.forward(batch_data)

        self.optimizer.zero_grad()
        loss_dict['loss'].backward()
        self.optimizer.step()

        return loss_dict

    def save(self, file_name, total_it):
        # flow_state_dict = self.model.state_dict()
        flow_state_dict = self.ema_model.state_dict()
        clip_weights = [e for e in flow_state_dict.keys() if e.startswith('text_encoder.')]
        for e in clip_weights:
            del flow_state_dict[e]
        state = {
            'model': flow_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
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
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
        assert len(unexpected_keys) == 0
        try:
            assert all([k.startswith('text_encoder.') for k in missing_keys])
        except:
            import pdb; pdb.set_trace()
            

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer']) # Optimizer
            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            logging.warning('Resume without optimizer')
        it = 0
        try:
            it = checkpoint['total_it']
        except:
            logging.warning('Resume without iteration count')
        return it

    def print_results(self, fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, multimodality, it, split='val'):
        msg = f"--> \t Eva. Iter {it} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, Multimodality. {multimodality}"
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
            f'{split}/Multimodality': multimodality,
        }, it)
        self.logger.info(msg)

        if fid < self.best_fid:
            self.logger.info(f"Improved FID from {self.best_fid:.05f} to {fid}!!!")
            self.save(pjoin(self.cfg.folder.checkpoints, 'net_best_fid.tar'), it)
            self.best_fid = fid

        if matching_score_pred < self.best_MM:
            self.logger.info(f"Improved MM from {self.best_MM:.05f} to {matching_score_pred}!!!")
            self.save(pjoin(self.cfg.folder.checkpoints, 'net_best_mm.tar'), it)
            self.best_MM = matching_score_pred

        if R_precision[0] > self.best_R1:
            self.logger.info(f"Improved R1 from {self.best_R1:.05f} to {R_precision[0]}!!!")
            self.save(pjoin(self.cfg.folder.checkpoints, 'net_best_r1.tar'), it)
            self.best_R1 = R_precision[0]

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, eval_func, plot_eval=None):
        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.vae_model.to(self.device)

        it = 0
        best_loss = 1e5
        logs = defaultdict(lambda: 0.0, OrderedDict())
        if self.cfg.is_continue:
            model_dir = pjoin(self.cfg.folder.checkpoints, 'latest.tar')  # TODO
            it = self.resume(model_dir)
            print("Load model iterations:%d"%(it))

        total_iters = self.cfg.max_iter if not self.cfg.DEBUG else 200
        self.logger.info(f'Total Iters: {total_iters}')
        # print(f'Total Epochs: {self.cfg.max_iter // len(train_loader)}, Total Iters: {total_iters}')
        # print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

        fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, multimodality = eval_func(eval_val_loader, self.ema_model, self.vae_model, eval_wrapper=eval_wrapper, cond_scale=self.cfg.cond_scale, time_steps=self.cfg.time_steps)

        self.print_results(fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, multimodality, it, split='val')


        torch.cuda.empty_cache()
        self.model.train()
        self.vae_model.eval()

        pbar = tqdm(infinite_loader(train_loader), total=total_iters - it)
        self.logger.set_progress_bar(pbar)
        tqdm_keys = ['train/loss', 'val/loss', 'lr']

        for batch in pbar:
            it += 1
            if it < self.cfg.warm_up_iter:
                self.update_lr_warm_up(it, self.cfg.warm_up_iter, self.cfg.lr)
            else:
                self.scheduler.step()

            loss_dict = self.update(batch_data=batch)
            self.ema_update(min(0.999, it / 10_000)) # 10k iters for warm up
            
            logs['lr'] += self.optimizer.param_groups[0]['lr']
            for k, v in loss_dict.items():
                logs[f'train/{k}'] += v

            if it % self.cfg.log_every == 0:
                self.logger.log_dict({k: v / self.cfg.log_every for k, v in logs.items()}, it, tqdm_keys=tqdm_keys)
                logs = defaultdict(lambda: 0.0, OrderedDict())

            if it % self.cfg.save_latest == 0:
                self.save(pjoin(self.cfg.folder.checkpoints, 'latest.tar'), it)

            if it % self.cfg.eval_every == 0 or it == total_iters:
                self.logger.info('Validation time:')
                val_loss = []
                val_logs = defaultdict(lambda: 0.0, OrderedDict())
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        loss_dict = self.forward(batch_data)
                        val_loss.append(loss_dict['loss'].item())
                        for k, v in loss_dict.items():
                            val_logs[f'val/{k}'] += v

                self.logger.log_dict({k: v / len(val_loss) for k, v in val_logs.items()}, it, tqdm_keys=tqdm_keys)

                if np.mean(val_loss) < best_loss:
                    print(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}!!!")
                    self.save(pjoin(self.cfg.folder.checkpoints, 'net_best_loss.tar'), it)
                    best_loss = np.mean(val_loss)

                fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, multimodality = eval_func(eval_val_loader, self.ema_model, self.vae_model, eval_wrapper=eval_wrapper, cond_scale=self.cfg.cond_scale, time_steps=self.cfg.time_steps, animation_path=pjoin(self.cfg.folder.run, 'anim', f'It-{it}.mp4'))

                self.print_results(fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, multimodality, it, split='val')

            if it >= total_iters:
                self.logger.info('Finish training.')
                break


    @staticmethod
    @torch.no_grad()
    def test(model, eval_loader, eval_wrapper, cfg, eval_func, replication_times=1):
        # make things reproducible
        # seed_everything(cfg.seed)

        all_metrics = defaultdict(list)
        device = f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu"

        vae_model = load_model(pjoin(cfg.folder.base, cfg.vae_model.name), ckpt_name=cfg.vae_model.ckpt_name)
        vae_model = vae_model.to(device)
        model = model.to(device)

        for _ in range(replication_times):
            fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, multimodality = eval_func(eval_loader, model, vae_model, eval_wrapper=eval_wrapper, cond_scale=cfg.cond_scale, time_steps=cfg.time_steps, animation_path=pjoin(cfg.folder.run, 'anim_test', f'Rep-{_}.mp4'))
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
            all_metrics['Multimodality'].append(multimodality)

            logging.info("Replication %d/%d: FID %.4f, Diversity Real %.4f, Diversity %.4f, R_precision_real %s, R_precision %s, matching_score_real %.4f, matching_score_pred %.4f, Multimodality %.4f" % (
                _ + 1, replication_times, fid, diversity_real, diversity, str(R_precision_real), str(R_precision), matching_score_real, matching_score_pred, multimodality
            ))
        return all_metrics