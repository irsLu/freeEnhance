import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm.auto import tqdm
from diffusers.utils import load_image
from diffusers import DDIMScheduler, DDIMInverseScheduler
import torch
import PIL

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
            FusedAttnProcessor2_0,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)



def prepare_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    # Offload text encoder if `enable_model_cpu_offload` was enabled
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image
    else:
        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            init_latents = [
                retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        if self.vae.config.force_upcast:
            self.vae.to(dtype)

        init_latents = init_latents.to(dtype)
        init_latents = self.vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    if add_noise:
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # get latents
        init_latents_wnoise = self.scheduler.add_noise(init_latents, noise, timestep)
        return init_latents_wnoise, init_latents

    return init_latents, init_latents


@torch.no_grad()
def get_sdxl_needed_attr(pipe, image, prompt, negative_prompt, num_inference_steps, strength, guidance_scale,
                         add_noise=True, denoising_end=None):
    if type(image) != PIL.Image.Image:
        image_pil = load_image(image)
    else:
        image_pil = image
    image_pil = image_pil.convert("RGB").resize((1024, 1024), 3)

    # 0.1  set param
    # region
    prompt_2 = None
    timesteps = None
    denoising_start = None
    denoising_end = denoising_end
    negative_prompt_2 = None
    num_images_per_prompt = 1
    eta = 0.0
    generator = None
    prompt_embeds = None
    negative_prompt_embeds = None
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    cross_attention_kwargs = None

    original_size = None
    crops_coords_top_left = (0, 0)
    target_size = None
    negative_original_size = None
    negative_crops_coords_top_left = (0, 0)
    negative_target_size = None
    aesthetic_score = 6
    negative_aesthetic_score = 2.5
    clip_skip = None
    do_classifier_free_guidance = guidance_scale > 1
    height, width = 1024, 1024
    original_size = (1024, 1024)
    target_size = (1024, 1024)
    callback_on_step_end = None
    callback = None
    batch_size = 1
    device = pipe.device
    # endregion

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    # with torch.no_grad():
    # with torch.no_grad():
    # prompt_embeds [2, 77, 2048]
    # negative_prompt_embeds [1, 77, 2048]
    # pooled_prompt_embeds [1, 1280]
    # negative_pooled_prompt_embeds [1, 1280]
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=pipe.device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
        clip_skip=clip_skip,
    )
    # 4. Preprocess image
    image = pipe.image_processor.preprocess(image_pil)

    # 5. Prepare timesteps
    def denoising_value_valid(dnv):
        return isinstance(denoising_end, float) and 0 < dnv < 1

    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, timesteps)
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps,
        strength,
        device,
        denoising_start=denoising_start if denoising_value_valid else None,
    )

    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    add_noise = add_noise and (True if denoising_start is None else False)
    init_latents_wnoise, init_latents = prepare_latents(pipe,
                                                        image,
                                                        latent_timestep,
                                                        batch_size,
                                                        num_images_per_prompt,
                                                        prompt_embeds.dtype,
                                                        device,
                                                        generator,
                                                        add_noise,
                                                        )

    # 7. Prepare extra step kwargs. ？？
    # extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    height, width = (128, 128)  # TODO
    height = height * pipe.vae_scale_factor
    width = width * pipe.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 8. Prepare added time ids & embeddings ？？
    if negative_original_size is None:
        negative_original_size = original_size
    if negative_target_size is None:
        negative_target_size = target_size

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids, add_neg_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)  # [2, 77, 2048]
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)  # [2, 1280]
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    return (init_latents_wnoise,
            init_latents,
            prompt_embeds,
            added_cond_kwargs,
            do_classifier_free_guidance,
            num_inference_steps,
            timesteps,
            )


@torch.no_grad()
def ddim_inverse(pipe, image, args, prompt, negative_prompt):
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.num_steps, pipe.device, None)
    timesteps, num_inference_steps = pipe.get_timesteps(
        args.num_steps,
        args.strength,
        'cuda',
        denoising_start=None,
    )

    pipe.scheduler = DDIMInverseScheduler.from_pretrained(args.ckpt, subfolder='scheduler')

    (
        init_latents_wnoise, init_latents,
        prompt_embeds,
        added_cond_kwargs,
        do_classifier_free_guidance,
        _,
        _
    ) = get_sdxl_needed_attr(pipe, image, prompt, negative_prompt, args.num_steps, args.strength, args.guidance_scale,
                             add_noise=False)

    ret_eps = []
    ret_latents = []

    latents = init_latents_wnoise
    latents = latents * pipe.scheduler.init_noise_sigma

    def progress_bar(iterable=None, total=None):
        _progress_bar_config = {}

        if iterable is not None:
            return tqdm(iterable, **_progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **_progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    timesteps = torch.flip(timesteps, dims=[0])
    # print("ddim inverse ", timesteps)
    # 12.0 denoise
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

    with progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and args.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=args.guidance_rescale)

            ret_eps.append(noise_pred)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            ret_latents.append(latents)
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    pipe.scheduler = DDIMScheduler.from_pretrained(args.ckpt, subfolder='scheduler')

    timesteps = torch.flip(timesteps, dims=[0])
    ret_eps = ret_eps[::-1]
    ret_latents = ret_latents[::-1]
    return (latents, ret_eps, ret_latents, timesteps)



def ddim_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = False,
        prev_timestep=None
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if prev_timestep is None:
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        variance = std_dev_t * variance_noise

        prev_sample = prev_sample + variance

    return prev_sample


@torch.no_grad()
def latent2img(latents, pipe, output_type='pil'):
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        upcast_vae(pipe.vae)
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    image_ = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]  # [1, 3, 1024, 1024]
    # cast back to fp16 if needed
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)
    image_ = pipe.image_processor.postprocess(image_, output_type=output_type)
    return image_[0]

