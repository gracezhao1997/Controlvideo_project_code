from util import ddim_inversion, ddim_inversion_keyframe_long
from einops import rearrange, repeat
import torch
from util import get_weights, get_index
def VideoGen(function, validation_data, generator, latents, validation_pipeline, ddim_inv_scheduler, train_data, control, weight_dtype, control_scale, samples):
    # start value
    if validation_data.start == 'noise':
        B, C, f, H, W = latents.shape
        noise = torch.randn([C, H, W], device=latents.device)
        noise = rearrange(noise, 'c h w -> 1 c h w')
        noise = repeat(noise, '1 ... -> f ...', f=f)
        noise = rearrange(noise, 'f c h w -> 1 f c h w')
        noise = repeat(noise, '1 ... -> b ...', b=B)
        noise = rearrange(noise, "b f c h w -> b c f h w")
        start_value = noise.to(weight_dtype)
    elif validation_data.start == 'inversion':
        if function=='long':
            start_value = ddim_inversion_keyframe_long(
                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                num_inv_steps=validation_data.num_steps, prompt=train_data.prompt, controls=control,
                controlnet_conditioning_scale=control_scale, video_length=validation_data.video_length,
                sub_frames=validation_data.sub_frames, overlap=validation_data.overlap, var=validation_data.var,
                key_weight=validation_data.key_weight, weights_type=validation_data.weights_type)[-1].to(weight_dtype)
        else:
            start_value = ddim_inversion(
                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                num_inv_steps=validation_data.num_steps, prompt=train_data.prompt, controls=control,
                controlnet_conditioning_scale=control_scale)[-1].to(weight_dtype)
    else:
        raise TypeError(validation_data.start)

    # sampling
    if function == 'long': # long video editing
        weights = torch.tensor(get_weights(validation_data.sub_frames, validation_data.var, validation_data.weights_type), device=latents.device)
        starts, ends = get_index(validation_data.video_length, validation_data.sub_frames, validation_data.overlap)
        for idx, prompt in enumerate(validation_data.prompts):
            sample = validation_pipeline(prompt, generator=generator, latents=start_value, controls=control,
                                         controlnet_conditioning_scale=control_scale, weights=weights, starts=starts, ends=ends,
                                         **validation_data).videos
            samples.append(sample)
    else:
        starts, ends = [0], [validation_data.video_length]
        for idx, prompt in enumerate(validation_data.prompts):
            sample = validation_pipeline(prompt, generator=generator, latents=start_value, controls=control,
                                         controlnet_conditioning_scale=control_scale, starts=starts, ends=ends,
                                         **validation_data).videos
            samples.append(sample)

    return samples

