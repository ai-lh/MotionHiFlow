import torch
import math
import random
from torch import Tensor
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from src.arch.text_encoder import FrozenTextEncoder
from src.utils import interpolate, joint_pos_id
from src.utils import capture_init_kwargs
from .scheduler_flow_matching import PyramidFlowMatchEulerDiscreteScheduler
from .embedding import TimestepEmbedding, LearnableEmbedding
from .transformer import MotionFlux, rope


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        assert sum(axes_dim) == dim, f"sum of axes_dim must be equal to dim, but got sum({axes_dim}) = {sum(axes_dim)} != {dim}"

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb

def lengths_to_mask(lengths: torch.Tensor, max_length: int=None) -> torch.Tensor:
    max_frames = int(torch.max(lengths).long().item()) if max_length is None else max_length
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


class FlowModel(torch.nn.Module):
    @capture_init_kwargs
    def __init__(
        self, latent_dim, vae_dim, joints_num, pos_type='rope',
        n_heads=6, n_double_layers=3, n_single_layers=6, dropout=0.3,
        pooled_encoder='ViT-B/32', text_encoder='ViT-B/32', scales=[0.3, 0.6, 1.0],
        time_patch=2, v_split=False, scale_emb_type='sincos', short_cut=False,
        interpolation_type='half-diff', interpolation_mode='linear', cond_drop_prob=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pooled_text_dim = 512 if pooled_encoder == "ViT-B/32" else 768 # ViT-L/14
        self.text_dim = 512 if text_encoder == "ViT-B/32" else 768 # ViT-L/14
        self.tp = time_patch
        assert self.tp >= 1, f'time patch size should not be smaller than 1, but get {self.tp}'

        self.v_split = v_split
        if v_split:
            self.pos_linear = nn.Linear(vae_dim // 4 * 3, vae_dim)
            self.v_linear = nn.Linear(vae_dim // 4, vae_dim)
            self.pos_out = nn.Linear(vae_dim, vae_dim // 4 * 3)
            self.v_out = nn.Linear(vae_dim, vae_dim // 4)
            joints_num = joints_num * 2
        # input & output process
        self.joints_num = joints_num
        self.vae_dim = vae_dim
        self.input_process = nn.Sequential(
            Rearrange('b t j d -> (b j) d t'),
            nn.ZeroPad1d((0, self.tp - 1)),
            nn.Conv1d(vae_dim, latent_dim, kernel_size=self.tp, stride=self.tp),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1, stride=1),
            Rearrange('(b j) d t -> b t j d', j=joints_num),
        )
        self.unpatching = Rearrange('b (t j) (tp d) -> b (t tp) j d', tp=self.tp, j=joints_num)
        
        # timestep embedding
        self.scale_emb_type = scale_emb_type
        self.scales = scales
        if scale_emb_type == "sincos":
            self.scale_emb = TimestepEmbedding(self.latent_dim)
        else:
            self.scale_emb = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(len(scales), self.latent_dim)))
        self.timestep_emb = TimestepEmbedding(self.latent_dim)
        if short_cut:
            self.short_cut_emb = TimestepEmbedding(self.latent_dim)
        else:
            self.short_cut_emb = None

        # CLIP text encoder
        # self.text_encoder = create_text_encoder('default')
        self.text_encoder = FrozenTextEncoder(text_encoder=text_encoder, pooled_encoder=pooled_encoder)
        self.word_emb = nn.Linear(self.text_encoder.text_dim, self.latent_dim)
        self.pooled_proj = nn.Sequential(
            nn.Linear(self.pooled_text_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.cond_drop_prob = cond_drop_prob
        self.weighting_scheme = "none"
        
        # positional embedding
        self.pos_type = pos_type
        if pos_type == "joint":
            self.pos_emb = LearnableEmbedding(self.latent_dim)
            ratios = [1,] # only temporal
        else:
            ratios = [1/2, 1/8, 1/8, 1/4]
        self.pe_embedder = EmbedND(dim=self.latent_dim // n_heads, theta=10000, axes_dim=[int(self.latent_dim // n_heads * ratio) for ratio in ratios])

        # interpolation
        self.interpolation_type = interpolation_type
        self.interpolation_mode = interpolation_mode

        # transformer
        self.transformer = MotionFlux(
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_double_layers=n_double_layers,
            n_single_layers=n_single_layers,
            output_size=vae_dim * self.tp,
            dropout=dropout,
        )

        # scheduler
        self.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
            stages=len(scales),
            stage_range=[0] + scales,
            gamma=1,
            shift=1.0, # as stable diffusion 3
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def parameters_without_clip(self):
        return [param for name, param in self.named_parameters() if "text_encoder" not in name]
    
    def state_dict_without_clip(self):
        state_dict = self.state_dict()
        remove_weights = [e for e in state_dict.keys() if "text_encoder." in e or "_cache_" in e]
        for e in remove_weights:
            del state_dict[e]
        return state_dict

    def forward(self, x, timestep, text, d_cond=None, len_mask=None, drop_text=0., scale_id=0):
        """
        sample: [B, T, J, D]
        timestep: [B,] or [1,]
        lengths: [B,]
        """
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(-2)
            squeeze = True
        if self.v_split:
            v_dim = x.shape[-1] // 4
            x = self.pos_linear(x[:, :, :, :-v_dim])
            x_v = self.v_linear(x[:, :, :, -v_dim:])
            x = torch.cat([x, x_v], dim=-1)
            x = rearrange(x, 'b t j (x d) -> b t (j x) d', x=2)

        # input process
        in_len = x.shape[1]
        x = self.input_process(x) # [B, T, J, D] -> [B, T', J, D]
        B, T, J, D = x.size()

        if len_mask is not None:
            if len_mask.shape[1] % self.tp != 0:
                len_mask = torch.cat([len_mask, torch.zeros_like(len_mask[:, :self.tp - len_mask.shape[1] % self.tp, ...])], dim=1)
            len_mask = repeat(len_mask, 'b (t tp) -> b (t j) tp', tp = self.tp, j = J).sum(dim=-1).bool()

        # text embedding
        word_emb, ca_mask, pooled_output = self.text_encoder.encode_text(text, drop_text=drop_text)
        word_emb = self.word_emb(word_emb)
        pooled_output = self.pooled_proj(pooled_output)
        # pooled text

        B, N, D = word_emb.size()

        # diffusion timestep embedding
        timestep_emb = self.timestep_emb(timestep).expand(B, D) + pooled_output
        if self.scale_emb_type == "sincos":
            timestep_emb = timestep_emb + self.scale_emb(scale_id).expand(B, D) # type: ignore
            motion_pe = repeat(torch.arange(T, device=x.device) / scale_id, 't -> b t j 1', b=B, j=J)
        else:
            timestep_emb = timestep_emb + self.scale_emb[scale_id-1].expand(B, D) # type: ignore
            motion_pe = repeat(torch.arange(T, device=x.device) * (2 ** (len(self.scales) - scale_id - 1)), 't -> b t j 1', b=B, j=J)
        if self.short_cut_emb is not None:
            if d_cond is None:
                d_cond = torch.zeros_like(timestep_emb[:,0])
            timestep_emb = self.short_cut_emb(d_cond).expand(B, D) + timestep_emb

        if self.pos_type == "joint":
            x = self.pos_emb(x)
        else:
            motion_pe = torch.cat([motion_pe, repeat(joint_pos_id[J].to(x.device), 'j x -> b t j x', b=B, t=T)], dim=-1)
        motion_pe = self.pe_embedder(motion_pe)
        motion_pe = rearrange(motion_pe, 'b t j ... -> b (t j) ...')
        
        text_pe = torch.zeros(B, N, len(self.pe_embedder.axes_dim), device=x.device)
        text_pe = self.pe_embedder(text_pe)


        # transformer
        pe = torch.cat([
            motion_pe,
            text_pe,
        ], dim=1)
        x = rearrange(x, 'b t j d -> b (t j) d')
        x = self.transformer(x, timestep_emb, word_emb, sa_mask=None if len_mask is None else ~len_mask, ca_mask=~ca_mask, pe=pe)

        # unpatching
        x = self.unpatching(x) # [B, T'*J, D*tp] -> [B, T'*tp, J, D]
        x = x[:, :in_len, ...]
        if self.v_split:
            x = rearrange(x, 'b t (j x) d -> x b t j d', x=2)
            x_pos = self.pos_out(x[0])
            x_v = self.v_out(x[1])
            x = torch.cat([x_pos, x_v], dim=-1)
        if squeeze:
            x = x.squeeze(-2)
        return x


    def train_forward(self, latent, text, m_lens):
        # latent
        with torch.no_grad():
            len_mask = lengths_to_mask(m_lens) # [B, L]
            latent = torch.nn.functional.pad(latent, (0,) * (latent.ndim * 2 - 3) + (len_mask.shape[1] - latent.shape[1],), mode="constant", value=0)
            latent = latent * append_dims(len_mask, latent.ndim)
        len_masks = [lengths_to_mask((m_lens * scale).long()) for scale in self.scales]
        lens = [mask.shape[1] for mask in len_masks]
        orig_len = (m_lens).max().item()

        # prepare noise
        noise = torch.randn_like(latent)
        latent_list = [interpolate(latent, scale_factor=l/orig_len + 1e-6, type=self.interpolation_type, mode=self.interpolation_mode) for l in lens]
        noise_list = [interpolate(noise, scale_factor=l/orig_len + 1e-6, type=self.interpolation_type, mode=self.interpolation_mode) for l in lens]

        start_sigmas = self.scheduler.start_sigmas
        end_sigmas = self.scheduler.end_sigmas

        loss_dict = {}
        loss = 0
        stage_id = random.choices(
            range(self.scheduler.config.stages), 
            weights=[self.scheduler.start_sigmas[i] - self.scheduler.end_sigmas[i] for i in range(self.scheduler.config.stages)]
        )[0]
        with torch.autocast("cuda", dtype=torch.float32):
            texts = list(text)
            len_mask = len_masks[stage_id]

            up_latent = interpolate(
                latent_list[stage_id - 1], 
                scale_factor=lens[stage_id] / lens[stage_id - 1] + 1e-6,
                type=self.interpolation_type, 
                mode=self.interpolation_mode
            ) if stage_id > 0 else torch.zeros_like(latent_list[0])
            if stage_id > 0:
                up_latent = torch.nn.functional.pad(up_latent, (0,) * (up_latent.ndim * 2 - 3) + (len_masks[stage_id].shape[1] - up_latent.shape[1],), mode="constant", value=0)
            x0 = start_sigmas[stage_id] * noise_list[stage_id] + (1 - start_sigmas[stage_id]) * up_latent
            x1 = end_sigmas[stage_id] * noise_list[stage_id] + (1 - end_sigmas[stage_id]) * latent_list[stage_id]

            indices = torch.randint(0, self.scheduler.config.num_train_timesteps, (latent.shape[0],))
            time_steps = self.scheduler.timesteps_per_stage[stage_id][indices].to(latent)
            ratios = self.scheduler.sigmas_per_stage[stage_id][indices].to(latent)
            ratios = append_dims(ratios, x0.ndim)

            xt = ratios * x0 + (1 - ratios) * x1
            ut = (x1 - x0)

            pred = self.forward(xt * append_dims(len_mask, xt.ndim), time_steps, texts, len_mask=len_mask, scale_id=self.scales[stage_id], drop_text=self.cond_drop_prob)

            # pred = pred * len_mask[..., None, None].float()
            pred = pred * append_dims(len_mask, pred.ndim)
            
            # loss_sample = self.recon_criterion(pred, ut) * weight_pi(ratios, 1, 0) # following SD3
            loss_sample = (((pred - ut) ** 2).flatten(start_dim=1).mean(dim=1) * compute_loss_weighting_for_sd3(self.weighting_scheme, ratios.flatten())).mean() # following SD3

            loss += loss_sample

            loss_dict[f"loss_{stage_id}"] = loss_sample

        loss_dict["loss"] = loss
        return loss_dict



    @torch.no_grad()
    def generate(self, text, m_lengths, time_steps=12, cond_scale=4.5, cfg_interval=[0., 1.], use_sde=False):
        m_lengths = torch.clamp(m_lengths, min=5)
        # torch.randint(low=0, high=1, size=m_lengths.shape, device=m_lengths.device)
        orig_len = int(m_lengths.max().item())
        m_lens = [(m_lengths * scale).long() for scale in self.scales]
        lens = [ml.max().item() for ml in m_lens]

        input_text = text
        if cond_scale != 1.0:
            input_text = [""] * len(text) + list(text)

        # initial noise
        stage_id = 0
        noise = torch.randn(len(text), orig_len, self.joints_num, self.vae_dim, device=self.device)
        noise_list = [interpolate(noise, scale_factor = lens[i] / orig_len + 1e-6, type=self.interpolation_type, mode=self.interpolation_mode) for i in range(len(lens))]
        latents = noise_list[0]

        # flow
        start_sigmas = self.scheduler.start_sigmas
        end_sigmas = self.scheduler.end_sigmas
        for stage_id in range(self.scheduler.config.stages):
            add_noise = 0
            if stage_id > 0: # renoise
                latents = interpolate(
                    latents - end_sigmas[stage_id-1] * noise_list[stage_id-1], 
                    scale_factor=lens[stage_id] / lens[stage_id - 1] + 1e-6,
                    type=self.interpolation_type, 
                    mode=self.interpolation_mode
                )
                latents = latents * ((1 - start_sigmas[stage_id]) / (1 - end_sigmas[stage_id-1]))
                add_noise = start_sigmas[stage_id] * noise_list[stage_id]
            latents = torch.nn.functional.pad(latents, (0,) * (latents.ndim * 2 - 3) + (noise_list[stage_id].shape[1] - latents.shape[1],), mode="constant", value=0)
            len_mask = lengths_to_mask(m_lens[stage_id], max_length=latents.shape[1]) # [B, L]
            latents = (latents + add_noise) * append_dims(len_mask, latents.ndim)

            # set diffusion timesteps
            num_inference_timesteps = int(time_steps * (self.scheduler.start_sigmas[stage_id] - self.scheduler.end_sigmas[stage_id]) + 0.5) # round to nearest integer
            self.scheduler.set_timesteps(max(num_inference_timesteps, 1), stage_id)
            sigmas = self.scheduler.sigmas.to(latents.device)
            timesteps = self.scheduler.timesteps.to(latents.device)
            
            for i, timestep in enumerate(timesteps):
                if cond_scale != 1.0:
                    input_latents = torch.cat([latents] * 2, dim=0)
                    input_len_mask = torch.cat([len_mask] * 2, dim=0)
                else:
                    input_latents = latents
                    input_len_mask = len_mask

                if self.short_cut_emb is not None:
                    pred = self.forward(input_latents, timestep, input_text, d_cond = sigmas[i] - sigmas[i+1],
                                                len_mask=input_len_mask, scale_id=self.scales[stage_id])
                else:
                    pred = self.forward(input_latents, timestep, input_text,
                                                len_mask=input_len_mask, scale_id=self.scales[stage_id])

                # classifier-free guidance
                if cond_scale != 1.0:
                    pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                    pred = (pred_uncond + cond_scale * (pred_cond - pred_uncond))
                    if sigmas[i] < cfg_interval[0] or sigmas[i] > cfg_interval[1]:
                        pred = pred_cond

                # step
                if use_sde:
                    latents = latents + sigmas[i] * pred + sigmas[i+1] * torch.randn_like(latents)
                else:
                    latents = latents + (sigmas[i] - sigmas[i+1]) * pred 

                latents = latents * append_dims(len_mask, latents.ndim)


        if self.scales[-1] != 1: # pool first or decode first, approximately the same
            latents = interpolate(latents, scale_factor = orig_len / lens[-1] + 1e-6, type=self.interpolation_type, mode=self.interpolation_mode)
        try:
            latents = latents.squeeze(-2)
        except:
            pass

        return latents, m_lengths