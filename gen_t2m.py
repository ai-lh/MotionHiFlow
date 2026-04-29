import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from src.utils import load_model, seed_everything, render_animations
from src.utils.motion_process import recover_from_ric
from src.visualization.joints2bvh import Joint2BVHConvertor


def parse_prompts(cfg: DictConfig):
    """Parse text prompts and motion lengths from config."""
    prompt_list, length_list = [], []
    est_length = False

    if cfg.text_prompt:
        prompt_list.append(cfg.text_prompt)
        if cfg.motion_length == 0:
            est_length = True
        else:
            length_list.append(cfg.motion_length)

    elif cfg.text_path:
        with open(cfg.text_path, "r") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("#")
                prompt_list.append(parts[0].strip())
                if len(parts) >= 2 and parts[1].strip().isdigit():
                    length_list.append(int(parts[1].strip()))
                else:
                    est_length = True
    else:
        raise ValueError("Either 'text_prompt' or 'text_path' must be provided.")

    return prompt_list, length_list, est_length


@hydra.main(version_base=None, config_path="./configs", config_name="gen")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu"

    # ---- Load models ----
    logging.info("Loading Flow model...")
    flow_model = load_model(cfg.folder.run, cfg.ckpt_name).to(device)

    logging.info("Loading VAE model...")
    vae_model = load_model(
        str(Path(cfg.folder.base) / cfg.vae_model.name),
        ckpt_name=cfg.vae_model.ckpt_name,
    ).to(device)

    flow_model.eval()
    vae_model.eval()

    # ---- Inverse transform (un-normalize) ----
    mean = np.load(cfg.data.eval_mean)
    std = np.load(cfg.data.eval_std)

    def inv_transform(data: np.ndarray) -> np.ndarray:
        return data * std + mean

    # ---- Parse prompts ----
    prompt_list, length_list, est_length = parse_prompts(cfg)
    if not prompt_list:
        raise ValueError("No valid prompts found.")

    if est_length:
        # TODO: implement length estimation from text embeddings
        raise NotImplementedError(
            "Automatic motion length estimation is not yet implemented. "
            "Please provide 'motion_length' for each prompt (e.g. 'text#length' in the file)."
        )

    token_lens = torch.LongTensor(length_list).to(device) // 4
    m_lengths = token_lens * 4
    captions = prompt_list

    # ---- Expand batch for repeat_times ----
    # Each prompt is repeated cfg.repeat_times times so that generate()
    # produces all variations in a single forward pass (different random noise).
    n_prompts = len(captions)
    repeat = cfg.repeat_times
    captions = [p for p in captions for _ in range(repeat)]
    m_lengths = m_lengths.repeat_interleave(repeat, dim=0)
    logging.info(f"Generating {n_prompts} prompts × {repeat} repeats = {len(captions)} samples in one batch")

    # ---- Setup output directories ----
    result_dir = Path(cfg.output_dir) / cfg.name
    joints_dir = result_dir / "joints"
    animation_dir = result_dir / "animations"
    joints_dir.mkdir(parents=True, exist_ok=True)
    animation_dir.mkdir(parents=True, exist_ok=True)

    # ---- BVH converter ----
    converter = Joint2BVHConvertor()
    joints_num = cfg.data.meta.joints

    # ---- Generate in batches (with dynamic OOM halving) ----
    total = len(captions)
    init_bs = cfg.batch_size if cfg.batch_size > 0 else total
    all_latents, all_pred_lengths = [], []

    def _generate_batch(batch_captions, batch_m_lengths, bs):
        """Try generating one batch; returns (latents, pred_lengths) or raises."""
        with torch.no_grad():
            lat, pl = flow_model.generate(
                text=batch_captions,
                m_lengths=batch_m_lengths // 4,
                time_steps=cfg.time_steps,
                cond_scale=cfg.cond_scale,
                cfg_interval=list(cfg.cfg_interval),
            )
        return lat, pl * 4

    start = 0
    cur_bs = min(init_bs, total)
    while start < total:
        end = min(start + cur_bs, total)
        batch_captions = captions[start:end]
        batch_m_lengths = m_lengths[start:end]
        try:
            logging.info(f"  Generating batch [{start}:{end}] / {total}  (batch_size={cur_bs})")
            lat, pl = _generate_batch(batch_captions, batch_m_lengths, cur_bs)
            all_latents.append(lat)
            all_pred_lengths.append(pl)
            start = end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_bs = cur_bs // 2
            if new_bs < 1:
                raise RuntimeError(
                    f"OOM even with batch_size=1 for sample {start}. "
                    "Try reducing motion_length or time_steps."
                )
            logging.warning(
                f"  OOM at batch_size={cur_bs}, halving to {new_bs} and retrying..."
            )
            cur_bs = new_bs

    latents = torch.cat(all_latents, dim=0)
    pred_lengths = torch.cat(all_pred_lengths, dim=0)

    # Decode latents to motion (also in batches to avoid OOM during decode)
    all_pred_motions = []
    dec_bs = cur_bs  # reuse the safe batch size from generation
    for d_start in range(0, total, dec_bs):
        d_end = min(d_start + dec_bs, total)
        try:
            with torch.no_grad():
                decoded = vae_model.decode(latents[d_start:d_end])
            all_pred_motions.append(decoded)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            dec_bs = max(dec_bs // 2, 1)
            logging.warning(f"  OOM during decode, halving decode batch_size to {dec_bs}, retrying...")
            # Retry with smaller batch
            for d2_start in range(d_start, d_end, dec_bs):
                d2_end = min(d2_start + dec_bs, d_end)
                with torch.no_grad():
                    decoded = vae_model.decode(latents[d2_start:d2_end])
                all_pred_motions.append(decoded)

    pred_motions = torch.cat(all_pred_motions, dim=0)

    # Mask padding
    bs = pred_motions.shape[0]
    mask = (
        torch.arange(pred_motions.shape[1], device=device).expand(bs, -1)
        < pred_lengths.unsqueeze(1)
    )
    pred_motions = pred_motions.masked_fill(~mask.unsqueeze(-1), 0)
    logging.info("Motion generation complete. Processing and saving results...")

    # ---- Post-process each sample ----
    pred_np = pred_motions.detach().cpu().numpy()
    data = inv_transform(pred_np)

    for idx, (caption, joint_data, m_len) in enumerate(
        zip(captions, data, pred_lengths.cpu().numpy())
    ):
        k = idx // repeat   # prompt index
        r = idx % repeat    # repeat index
        m_len = int(m_len)
        logging.info(f"  Sample {idx} (prompt={k}, repeat={r}): '{caption[:80]}' | length={m_len}")

        # Sanitize caption for filesystem
        safe_name = caption[:100].replace("/", "_").replace(" ", "_")
        anim_path = animation_dir / safe_name
        jnt_path = joints_dir / safe_name
        anim_path.mkdir(parents=True, exist_ok=True)
        jnt_path.mkdir(parents=True, exist_ok=True)

        # Recover 3D joint positions
        joint_data_trimmed = joint_data[:m_len]
        joint = recover_from_ric(
            torch.from_numpy(joint_data_trimmed).float(), joints_num
        ).numpy()

        # Export BVH (with and without IK)
        bvh_ik = anim_path / f"sample{k}_r{r}_len{m_len}_ik.bvh"
        _, ik_joint = converter.convert(joint, filename=str(bvh_ik), iterations=100)

        bvh_raw = anim_path / f"sample{k}_r{r}_len{m_len}.bvh"
        _, joint_no_ik = converter.convert(
            joint, filename=str(bvh_raw), iterations=100, foot_ik=False
        )

        # Render animation
        mp4_path = anim_path / f"sample{k}_r{r}_len{m_len}.mp4"
        render_animations(
            jointss=[ik_joint, joint_no_ik, joint],
            output=str(mp4_path),
            title=caption,
            titles=["ik", "no ik", "original"],
            kinematic_tree=cfg.data.dataset_name,
            fps=20,
        )

        # Save raw data
        np.save(jnt_path / f"sample{k}_r{r}_len{m_len}.npy", joint)
        np.save(jnt_path / f"sample{k}_r{r}_len{m_len}_ik.npy", ik_joint)
        np.save(anim_path / f"sample{k}_r{r}_len{m_len}_data.npy", joint_data)

    logging.info(f"Generation complete. Results saved to {result_dir}")


if __name__ == "__main__":
    main()