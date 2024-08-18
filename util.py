import os
import imageio
import numpy as np
from typing import Union
import logging
import torch
import torchvision
from tqdm import tqdm
from einops import rearrange
import datetime
import pprint
from omegaconf import ListConfig

def merge_weight(state_dict, pipeline, alpha):
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:

        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue

        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER + '_')[-1].split('_')
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET + '_')[-1].split('_')
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_' + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

def available_devices(threshold=5000,n_devices=None):
    """
    search for available GPU devices
    Args:
        threshold: the devices with larger memory than threshold is available
        n_devices: the number of devices
    Returns:
        device: the id for available GPU devices
    """
    memory = list(os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader'))
    mem = [int(x.strip()) for x in memory]
    devices = []
    for i in range(len(mem)):
        if mem[i] > threshold:
            devices.append(i)
    device = devices if n_devices is None else devices[:n_devices]
    return device

def format_devices(devices):
    if isinstance(devices, list):
        return ','.join(map(str,devices))

def backup_profile(profile: dict, path):
    """
    backup args profile
    Args:
        profile: the args profile need to backup code
        path: the path for saving args profile
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "profile_{}.txt".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    s = pprint.pformat(profile)
    with open(path, 'w') as f:
        f.write(s)

def set_logger(path, file_path=None):
    os.makedirs(path,exist_ok=True)
    #logger to print information
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    if file_path is not None:
        handler2 = logging.FileHandler(os.path.join(path,file_path), mode='w')
    else:
        handler2 = logging.FileHandler(os.path.join(path, "logs.txt"), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

def save_tensor_images_folder(videos: torch.Tensor, path: str, rescale=False, n_rows=4):
    os.makedirs(path, exist_ok=True)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        save_path = os.path.join(path, f"{i}.png")
        imageio.imsave(save_path, x)
        outputs.append(x)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    return context

def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def get_multicontrol_pre_single(latents, t, context, unet, controlnets, controls, control_scale):
    for i, (image, scale, controlnet) in enumerate(zip(controls, control_scale, controlnets)):
        down_samples, mid_sample = controlnet(
            latents,
            t,
            encoder_hidden_states=context,
            controlnet_cond=image,
            return_dict=False,
        )
        down_samples = [
            down_samples * scale
            for down_samples in down_samples
        ]
        mid_sample *= scale

        # merge samples
        if i == 0:
            down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
        else:
            down_block_res_samples = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
            ]
            mid_block_res_sample += mid_sample

    noise_pred = unet(
        latents,
        t,
        encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    return noise_pred

def get_noise_pred_single(latents, t, context, unet, controlnet, controls, controlnet_conditioning_scale=1.0):
    if isinstance(controlnet_conditioning_scale, ListConfig):
        down_block_res_samples, mid_block_res_sample = controlnet(
            latents,
            t,
            encoder_hidden_states=context,
            controlnet_cond=controls,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )

    else:
        down_block_res_samples, mid_block_res_sample = controlnet(
            latents,
            t,
            encoder_hidden_states=context,
            controlnet_cond=controls,
            return_dict=False,
        )
        down_block_res_samples = [
            down_block_res_sample * controlnet_conditioning_scale
            for down_block_res_sample in down_block_res_samples
        ]
        mid_block_res_sample *= controlnet_conditioning_scale

    noise_pred = unet(
        latents,
        t,
        encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    return noise_pred

def get_multicontrol_pre_single(latents, t, context, unet, controlnets, controls, control_scale):
    for i, (image, scale, controlnet) in enumerate(zip(controls, control_scale, controlnets)):
        down_samples, mid_sample = controlnet(
            latents,
            t,
            encoder_hidden_states=context,
            controlnet_cond=image,
            return_dict=False,
        )
        down_samples = [
            down_samples * scale
            for down_samples in down_samples
        ]
        mid_sample *= scale

        # merge samples
        if i == 0:
            down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
        else:
            down_block_res_samples = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
            ]
            mid_block_res_sample += mid_sample

    noise_pred = unet(
        latents,
        t,
        encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    return noise_pred

def get_index(video_length, sub_frames, overlap):
    sub = 0
    starts = []
    ends = []
    while True:
        start = sub * (sub_frames - overlap)
        end = min(sub * (sub_frames - overlap) + sub_frames, video_length)
        sub += 1
        starts.append(start)
        ends.append(end)
        if end>=video_length:
            break
    return starts, ends

@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt,controls, controlnet_conditioning_scale):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet, pipeline.controlnet, controls, controlnet_conditioning_scale)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", controls=None, controlnet_conditioning_scale=1.0):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, controls, controlnet_conditioning_scale)
    return ddim_latents

def get_weights(length, var=0.1, type='Gaussian'):
    midpoint = (length - 1) / 2

    if type == 'Gaussian':
        from numpy import exp, pi, sqrt

        midpoint = (length - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (length * length) / (2 * var)) / sqrt(2 * pi * var)
            for x in range(length)
        ]

    elif type == 'linear':
        max_val = length / 2
        x_probs = [1 - abs(x - midpoint) / max_val + 0.2 for x in range(length)]

    elif type == 'constant':
        x_probs = [1.2 for _ in range(length)]

    elif type == 'cosine':
        x_probs = [np.cos(np.pi * (x - midpoint) / length) + 0.2 for x in range(length)]

    elif type == 'exp':
        x_probs = [np.exp(-abs(x - midpoint) / length) + 0.2 for x in range(length)]

    elif type == 'convex_inverse_square_root':
        x_probs = [1 / np.sqrt(abs(x - midpoint) + 1) for x in range(length)]
        max_val = max(x_probs)
        x_probs = [x / max_val + 0.2 for x in x_probs]

    elif type == 'convex_gaussian':
        var = (length / 8) ** 2
        x_probs = [np.exp(-(x - midpoint) ** 2 / (2 * var)) for x in range(length)]
        max_gauss = max(x_probs)
        x_probs = [x / max_gauss + 0.2 for x in x_probs]

    return x_probs

@torch.no_grad()
def ddim_inversion_keyframe_long(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", controls=None, controlnet_conditioning_scale=1.0,video_length=100, sub_frames=24, overlap=8, var=0.01,key_weight=0.8, weights_type='Gaussian'):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [video_latent]
    latents = video_latent.clone().detach()
    starts, ends = get_index(video_length, sub_frames, overlap)
    keyframes = torch.tensor([starts])[0]
    weight = torch.tensor(get_weights(sub_frames, var, weights_type), device=latents.device)
    weights = weight.reshape(1, 1, sub_frames, 1, 1)
    weights = weights.repeat(1, 4, 1, int(latents.shape[3]), int(latents.shape[4]))

    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_preds = []
        for start, end in zip(starts, ends):
            sub_latents = latents[:, :, start:end, :, :]
            sub_images = controls[:, :, start:end, :, :]
            noise_pred = get_noise_pred_single(sub_latents, t, cond_embeddings, pipeline.unet, pipeline.controlnet, sub_images,
                                               controlnet_conditioning_scale)
            noise_preds.append(noise_pred)

        whole_noise_pred = torch.zeros_like(latents).float()
        contributors = torch.zeros_like(latents).float()
        for start, end, noise_pred in zip(starts, ends, noise_preds):
            if end - start < sub_frames:
                weight = weights[:, :, :end - start, :, :]
            else:
                weight = weights
            whole_noise_pred[:, :, start:end, :, :] += (
                    noise_pred * weight
            )
            contributors[:, :, start:end, :, :] += weight
        whole_noise_pred /= contributors
        key_latents = latents[:, :, keyframes, :, :]
        key_images = controls[:, :, keyframes, :, :]
        noise_pred = get_noise_pred_single(key_latents, t, cond_embeddings, pipeline.unet, pipeline.controlnet,
                                           key_images, controlnet_conditioning_scale)
        whole_noise_pred[:, :, keyframes, :, :] = key_weight * noise_pred + (1 - key_weight) * whole_noise_pred[:, :, keyframes, :, :]
        latents = next_step(whole_noise_pred, t, latents, ddim_scheduler)
        all_latent.append(latents)
    return all_latent

@torch.no_grad()
def save_trainable_weights(
    model,
    path,
    trainable_modules
):
    weights = {}
    state_dict = model.state_dict()
    for name in state_dict:
        if any(trainable_module in name for trainable_module in trainable_modules):
            weights[name] = state_dict[name]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(weights, path)

from annotator.util import HWC3
def get_detected_map(batch, type, control_config, apply_control, device, weight_dtype):
    # only support 1 batchsize
    assert batch["control"].shape[0] == 1
    control_ = batch["control"].squeeze() #[f h w c] in {0,1,……,255}
    control = []

    # compute control for each frame
    for i in control_:
        if type == 'canny':
            detected_map = apply_control(i.cpu().numpy(), control_config.low_threshold, control_config.high_threshold)
        elif type == 'openpose':
            detected_map, _ = apply_control(i.cpu().numpy())
        else:
            detected_map = apply_control(i.cpu().numpy())
        control.append(HWC3(detected_map))

    # stack control with all frames with shape [b c f h w]
    control = np.stack(control)
    control = np.array(control).astype(np.float32) / 255.0
    control = torch.from_numpy(control).to(device)
    control = control.unsqueeze(0) #[f h w c] -> [b f h w c ]
    control = rearrange(control, "b f h w c -> b c f h w")
    control = control.to(weight_dtype)
    return control

def get_vae_latents(pixel_values, vae):
    # prepare latents with shape [b c f h w]
    video_length = pixel_values.shape[1]
    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    return latents
