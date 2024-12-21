from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from utils import retouch_mask,canny_mask, anyimage2npL
from sdxl_pipeline import get_sdxl_needed_attr, ddim_step

def get_dps_mask(pipe, image, args, l):
    typ = args.dps_mask_type
    if typ == '':
        ret = 1
    else:
        image_np = anyimage2npL(image)
        if typ == 'retouch':
            np_mask = retouch_mask(pipe, image_np, args.dps_retouch_k)
            renoise_mask = torch.from_numpy(np_mask).to(pipe.device)
        elif typ == 'canny':
            np_mask = canny_mask(pipe, image_np, args.dps_canny_l, args.dps_canny_h)
            renoise_mask = (torch.from_numpy(np_mask)).to(pipe.device).to(torch.float32)
            renoise_mask = renoise_mask / renoise_mask.max()
        renoise_mask = F.interpolate(renoise_mask.unsqueeze(0).unsqueeze(0), size=[128, 128],  # todo
                                     mode='bilinear').squeeze(0).squeeze(0)
        renoise_mask = torch.clamp(renoise_mask, 0, 1)
        in_v = args.dps_retouch_v
        mask = (renoise_mask > in_v).to(torch.float16)
    ret = [mask] * l
    return ret

def dps(pipe, args, x_prev, x_t, x_0_hat, y, scale):
    difference = y - x_0_hat
    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
    x_t -= norm_grad * scale

    return x_t



def over_noising(pipe, image, prompt, negative_prompt, args, in_xt, target_t):
    xt_steps = args.num_steps
    denoising_end = 1 - args.strength
    do_dps = args.do_dps
    strength = args.over_noising_st

    ## 3. denoising
    (
        init_latents_wnoise,
        init_latents,
        prompt_embeds,
        added_cond_kwargs,
        do_classifier_free_guidance,
        num_inference_steps,
        timesteps
    ) = get_sdxl_needed_attr(pipe, image, prompt, negative_prompt,
                             xt_steps, strength=strength, guidance_scale=args.guidance_scale,
                             denoising_end=denoising_end)

    if denoising_end is not None:
        discrete_timestep_cutoff = int(
            round(
                pipe.scheduler.config.num_train_timesteps
                - (denoising_end * pipe.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    with (torch.enable_grad() if do_dps else torch.no_grad()):

        latents = init_latents_wnoise

        if do_dps:
            y = init_latents
            latents = latents.requires_grad_(True)
            y.requires_grad_(False)
            scale = get_dps_mask(pipe, image, args, len(timesteps))

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                x_prev = latents
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if (do_dps):
                    x_0_hat = ddim_step(pipe.scheduler, noise_pred, t, latents, prev_timestep=0)
                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    x_t = latents
                    scale_ = scale[i] if i < len(scale) else 0
                    latents = dps(pipe, args, x_prev, x_t, x_0_hat, y, scale_)
                    del x_0_hat, x_t
                else:
                    print(latents.shape)
                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                del x_prev
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

    ## fix
    with torch.no_grad():
        t = t - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

    return latents, None
