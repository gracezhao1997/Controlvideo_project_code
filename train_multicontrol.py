import os
from typing import Dict, Optional, Tuple, List, Optional, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from util import save_videos_grid, set_logger, save_tensor_images_folder, save_trainable_weights, get_detected_map, get_noise_pred_single, get_vae_latents, get_multicontrol_pre_single
from dataset import VideoDataset
from libs.piplines import ControlVideoPipeline
from einops import rearrange
from annotator.util import get_control
import logging
from sampling import VideoGen
from libs.unet import UNet3DConditionModel
from libs.controlnet3d import ControlNetModel
from util import get_index
from safetensors.torch import load_file
from util import merge_weight
from omegaconf import ListConfig
from libs.multicontrolnet import MultiControlNetModel
logger = get_logger(__name__)

def main(
    function: str,
    pretrained_model_path: str,
    output_dir: str,
    pretrained_controlnet_path: Union[str, List[str]],
    lora: Dict,
    train_data: Dict,
    validation_data: Dict,
    control_config: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None
):
    if seed is not None:
        set_seed(seed)

    # set logging file
    output_dir_log = output_dir
    os.makedirs(output_dir_log, exist_ok=True)
    set_logger(output_dir_log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logging.info(output_dir_log)

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # prepare models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    controlnets = []
    for model_path in pretrained_controlnet_path:
        controlnet = ControlNetModel.from_pretrained_2d(model_path)
        controlnet.enable_xformers_memory_efficient_attention()
        controlnet.enable_gradient_checkpointing()
        controlnets.append(controlnet)
        controlnet = accelerator.prepare(controlnet)
    multicontrolnet = MultiControlNetModel(controlnets)

    apply_controls = []
    for control_type in control_config.type:
        apply_controls.append(get_control(control_type))

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()


    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    # set the params need to optimize
    # do not set unet.requires_grad_(False) because of the bug in gradient checkpointing in torch, where if the all the inputs don't need grad, the module in gradient checkpointing will not compute grad.
    optimize_params = []
    params_len = 0
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            optimize_params += list(module.parameters())
            for params in module.parameters():
                params_len += len(params.reshape(-1, ))

    for controlnet in controlnets:
        for name, module in controlnet.named_modules():
            if name.endswith(tuple(trainable_modules)):
                optimize_params += list(module.parameters())
                for params in module.parameters():
                    params_len += len(params.reshape(-1, ))

    logger.info(f"trainable params: {params_len / (1024 * 1024):.2f} M")

    optimizer = optimizer_cls(
        optimize_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # prepare dataloader
    train_dataset = VideoDataset(**train_data)
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # prepare ControlVideoPipeline
    validation_pipeline = ControlVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnets,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()

    if function == 'image':
        state_dict = load_file(lora.pretrained_lora_path)
        merge_weight(state_dict, validation_pipeline, alpha=lora.alpha)

    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_steps)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("controlvideo")

    # show the progress bar
    global_step = 0
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if function == 'long':
        starts, ends = get_index(validation_data.video_length, validation_data.sub_frames, validation_data.overlap)
    else:
        starts, ends = [0], [validation_data.video_length]


    for step, batch in enumerate(train_dataloader):
        while global_step < max_train_steps:

            for start, end in zip(starts, ends):

                file_list = sorted(os.listdir('./masks/dance5'))
                zero_indices = []
                overall_mask = torch.ones(batch["control"].squeeze().shape[1:3])
                import torchvision
                for file in file_list:
                    path = os.path.join('./masks/dance5', file)
                    mask = torchvision.io.read_image(path).squeeze()
                    mask_indices = (mask != 0).nonzero()
                    [zero_top, zero_left], _ = torch.min(mask_indices, 0)
                    [zero_bottom, zero_right], _ = torch.max(mask_indices, 0)
                    zero_indices.append([zero_top, zero_bottom, zero_left, zero_right])
                    overall_mask = overall_mask.mul(mask) / 255

                # prepare control, latents and text embedding
                multi_controls = []
                sub_controls = []
                for control_type, apply_control,zero_indice in zip(control_config.type, apply_controls, zero_indices):
                    control = get_detected_map(batch, control_type, control_config, apply_control, accelerator.device, weight_dtype, zero_indice)
                    multi_controls.append(control)
                    sub_controls.append(control[:, :, start:end, :, :])



                pixel_values = batch["pixel_values"].to(weight_dtype)
                latents = get_vae_latents(pixel_values, vae)
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]
                x0 = (rearrange(pixel_values, "b f c h w -> b c f h w") + 1.0) / 2.0  # for save original video

                sub_latents = latents[:, :, start:end, :, :]

                unet.train()
                for controlnet in controlnets:
                    controlnet.train()
                train_loss = 0.0

                # noise prediction
                noise = torch.randn_like(sub_latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (sub_latents.shape[0],), device=sub_latents.device).long()
                noisy_latents = noise_scheduler.add_noise(sub_latents, noise, timesteps)
                model_pred = get_noise_pred_single(noisy_latents, timesteps, encoder_hidden_states, unet, multicontrolnet, sub_controls, control_config.control_scale)

                # compute loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters()) + list(controlnet.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % validation_steps == 0:
                        if accelerator.is_main_process:
                            # save models
                            save_path = f"{output_dir_log}/checkpoints/{global_step}-unet.pt"
                            save_trainable_weights(unet, save_path, trainable_modules)
                            logger.info(f"Saved unet to {save_path}")
                            save_path = f"{output_dir_log}/checkpoints/{global_step}-controlnet.pt"
                            save_trainable_weights(controlnet, save_path, trainable_modules)
                            logger.info(f"Saved controlnet to {save_path}")

                            unet.eval()
                            for controlnet in controlnets:
                                controlnet.eval()

                            samples = [x0.cpu().float(), control.cpu().float()]
                            generator = torch.Generator(device=latents.device)
                            generator.manual_seed(seed)
                            samples = VideoGen(function, validation_data, generator, latents, validation_pipeline, ddim_inv_scheduler, train_data, multi_controls, weight_dtype, control_config.control_scale, samples)
                            sample_save = samples[-1]
                            samples = torch.concat(samples)

                            # save results
                            save_path = f"{output_dir_log}/{global_step}.mp4"
                            save_videos_grid(samples, save_path)
                            save_path = f"{output_dir}/{global_step}/controlvideo"
                            save_tensor_images_folder(sample_save, save_path)

                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break



    accelerator.end_training()


