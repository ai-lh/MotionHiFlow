import os
import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm
from typing import Union
from src.utils.metrics import *
import torch.nn.functional as F
from src.utils.motion_process import recover_from_ric
from src.utils.render import render_animations
from einops import rearrange


def _render_animations_worker(pred_skel: np.ndarray, gt_skel: np.ndarray, animation_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    render_animations([pred_skel, gt_skel], output=animation_path, titles=['gen', 'gt'], title=title)


def _launch_render_process(pred_skel: np.ndarray, gt_skel: np.ndarray, animation_path: str, title: str):
    process = mp.get_context("spawn").Process(
        target=_render_animations_worker,
        args=(pred_skel, gt_skel, animation_path, title),
    )
    process.start()


def length_to_mask(length, max_len: Union[int] = None, device: torch.device = None) -> torch.Tensor: # type: ignore
    if device is None:
        device = torch.device('cpu')

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    if max_len is None:
        max_len = int(length.max().item())
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


@torch.no_grad()
def evaluation_vae(val_loader, net, eval_wrapper, num_joint=22, cal_mm=None):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    # pampjpe = 0
    # accel = 0
    num_poses = 0
    for batch in tqdm(val_loader, desc='Evaluating VAE+MPJPE'):
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(net.device)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, z_encoded  = net(motion, return_latent=True)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # pampjpe += torch.sum(calc_pampjpe(gt, pred)) # type: ignore
            # accel += torch.sum(calc_accel(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    
    scale_factor = 1000 if num_joint == 22 else 1
    mpjpe = mpjpe / num_poses * scale_factor
    # pampjpe = pampjpe / num_poses * scale_factor
    # accel = accel / (num_poses - 2 * nb_sample) * scale_factor

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    return fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, mpjpe


@torch.no_grad()
def evaluation_dit(val_loader, dit, vae, eval_wrapper, time_steps=12, cond_scale=4.5, cal_mm=False, animation_path=None):
    dit.eval()
    vae.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader, desc='Evaluating DIT+MM. (cond_scale: %.2f | time_steps: %d)' % (cond_scale, time_steps))):
        # print(i)
        word_embeddings, pos_one_hots, text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(vae.device)

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids, pred_length = dit.generate(text, m_length // 4, time_steps, cond_scale)
                pred_length = pred_length * 4

                pred_motions = vae.decode(mids)

                mask = torch.arange(pred_motions.shape[1], device=pred_motions.device).expand(bs, pred_motions.shape[1]) < pred_length.unsqueeze(1)
                pred_motions = pred_motions.masked_fill(~mask.unsqueeze(-1), 0)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), pred_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids, pred_length = dit.generate(text, m_length // 4, time_steps, cond_scale)
            pred_length = pred_length * 4

            pred_motions = vae.decode(mids)

            mask = torch.arange(pred_motions.shape[1], device=pred_motions.device).expand(bs, pred_motions.shape[1]) < (pred_length).unsqueeze(1)
            pred_motions = pred_motions.masked_fill(~mask.unsqueeze(-1), 0)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), pred_length)
        
        if i == 0 and animation_path is not None:
            os.makedirs(os.path.dirname(animation_path), exist_ok=True)
            pred_skel = recover_from_ric(val_loader.dataset.inv_transform(pred_motions[0, :pred_length[0]]), (pred_motions.shape[-1] + 1) // 12).detach().cpu().numpy() # type: ignore
            gt_skel = recover_from_ric(val_loader.dataset.inv_transform(pose[0, :m_length[0]].to(pred_motions)), (pose.shape[-1] + 1) // 12).detach().cpu().numpy() # type: ignore
            _launch_render_process(pred_skel, gt_skel, animation_path, str(text[0]))

        pose = pose.to(vae.device).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred) # type: ignore

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True) # type: ignore
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace() # type: ignore
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    return fid, diversity_real, diversity, R_precision_real, R_precision, matching_score_real, matching_score_pred, multimodality




def test_vae_downsample(val_loader, eval_wrapper, num_joint, ratio=0.5, mode='linear'):
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    pampjpe = 0
    accel = 0
    multimodality = 0
    num_poses = 0

    def pool_motion(motion):
        # motion: [B, T, D]
        root, root_h, ric, rot, vel, contact = torch.split(motion, [3, 1, 3 * (num_joint - 1), 6 * (num_joint - 1), 3 * num_joint, 4], dim=-1) 
        def pool_latent(latent, type='none'):
            round_num = int(ratio * latent.shape[-2])
            if round_num == 0:
                return torch.zeros_like(latent)
            latent = rearrange(latent, 'b t d -> b d t')
            if type == 'cum-diff':
                latent = latent.cumsum(dim=-1)
            original_num = latent.shape[-1]
            latent = torch.nn.functional.interpolate(latent, size=round_num, mode=mode)
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

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluation", leave=False)):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        # GT motion
        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # predicted motion
        pred_pose_eval = torch.zeros_like(motion)
        for i in range(bs):
            pred_pose_eval[i:i+1, :m_length[i]] = pool_motion(motion[i:i+1, :m_length[i]])
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            pampjpe += torch.sum(calc_pampjpe(gt, pred))
            accel += torch.sum(calc_accel(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses * (1000 if num_joint == 22 else 1)
    pampjpe = pampjpe / num_poses * (1000 if num_joint == 22 else 1)
    accel = accel / (num_poses - 2 * nb_sample) * (1000 if num_joint == 22 else 1)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ratio {ratio} and mode {mode} : FID. {fid:.4f}, "\
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "\
          f"R_precision_real. ({R_precision_real[0]:.4f}, {R_precision_real[1]:.4f}, {R_precision_real[2]:.4f}), "\
          f"R_precision. ({R_precision[0]:.4f}, {R_precision[1]:.4f}, {R_precision[2]:.4f}), "\
          f"matching_real. {matching_score_real:.4f}, matching_pred. {matching_score_pred:.4f}, "\
          f"MPJPE. {mpjpe:.4f}, "\
          f"PAMPJPE. {pampjpe:.4f}, Accel. {accel:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe, pampjpe, accel
