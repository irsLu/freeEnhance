from diffusers import StableDiffusionXLImg2ImgPipeline
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm
from diffusers import DDIMScheduler
import torch
import argparse
import os
import torch.nn.functional as F
from utils import anyimage2npL,canny_mask,retouch_mask
from sdxl_pipeline import  ddim_inverse, ddim_step, get_sdxl_needed_attr, latent2img
from dps import dps_enhance
from over_noising import over_noising
from torch.utils.data import Dataset

def AdditionBlend(inverse_nosie, eps, weight, bias=0.0):
    weight += bias
    return weight * inverse_nosie + (1 - weight) * eps  #

def AdditionBlendEx(in_latent, latent, weight, bias=0.0):
    tau = weight + bias
    in_latent = in_latent.to(torch.float32)
    latent = latent.to(torch.float32)
    fix_fac = ((1 - 2 * tau + 2 * (tau ** 2))) ** 0.5
    ret = (tau / fix_fac) * in_latent + ((1 - tau) / fix_fac) * latent  #

    return ret.to(torch.float16)

def get_xt_blend(pipe, image, args):
    ret = []
    for blend_fac, blend_func in list(zip(args.blend_fac, args.blend_func)):
        factors = None
        func = None

        if blend_fac == 'cst':
            factors = [args.xt_tau] * args.num_steps
        elif blend_fac == 'retouch' or blend_fac == 'canny':
            image_np = anyimage2npL(image)

            if blend_fac == 'retouch':
                np_mask = retouch_mask(pipe, image_np, args.xt_retouch_k)
                renoise_mask = torch.from_numpy(np_mask).to(pipe.device)
            elif blend_fac == 'canny':
                np_mask = canny_mask(pipe, image_np, args.xt_canny_l, args.xt_canny_h)
                renoise_mask = (torch.from_numpy(np_mask)).to(pipe.device).to(torch.float32)
                renoise_mask = renoise_mask / renoise_mask.max()

            renoise_mask = F.interpolate(renoise_mask.unsqueeze(0).unsqueeze(0), size=[128, 128],  # todo
                                         mode='bilinear').squeeze(0).squeeze(0)
            renoise_mask = torch.clamp(renoise_mask, 0, 1)

            if blend_fac == 'retouch':
                in_v = args.xt_retouch_v
            elif blend_fac == 'canny':
                in_v = args.xt_canny_v

            mask = (renoise_mask > in_v).to(torch.float16)
            factors = [mask] * args.num_steps

        # blend_func = args.blend_func
        if blend_func == 'addition':
            func = AdditionBlend
        elif blend_func == 'additionEx':
            func = AdditionBlendEx
        ret.append((factors, func))

    return ret

class HPSv2Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        images = os.listdir(root)
        self.names = [i.split('.jpg')[0] for i in images]
        self.images = [os.path.join(root, i) for i in images]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        name = self.names[idx]

        return img, name

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--data_path', default='/remote-home/share/luoyang/test/HPDv2/SDXL-base-0.9-small')
    parser.add_argument('--output_dir', default='/remote-home/share/luoyang/output/HPDv2/SDXL-base-0.9-small')
    parser.add_argument('--euler_s_churn', default=0, type=float)
    parser.add_argument('--ddim_eta', default=0, type=float)
    parser.add_argument('--strength', default=0.5, type=float)
    parser.add_argument('--num_steps', default=100, type=int)
    parser.add_argument('--ckpt',
                        default='/remote-home/luoyang/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0',
                        type=str)
    parser.add_argument('--ckpt_unet', default='', type=str, help='for self-design sd model (e.g., dreamshapeXL)')
    parser.add_argument('--guidance_scale', default=1, type=float)
    parser.add_argument('--guidance_rescale', default=0, type=float)

    # stable stream
    parser.add_argument('--in_noise_type', default='ddim', type=str,
                        help='getting inversion noise according different type, containing ddim (DDIM inversion)')
    parser.add_argument('--in_cos_alpha', default=1, type=float, help='the cosine factor from DemonFusion')

    # creative stream
    parser.add_argument('--do_in_sbs', action='store_true', help='whether add creative stream step by step')
    parser.add_argument('--xt_type', default='over_noising', type=str, help='creative stream type')
    parser.add_argument('--over_noising_st', default=1, type=float, help='creative stream strength')
    parser.add_argument('--do_dps', action='store_true',
                        help='whether using diffusion-posterior-sampling (dps) in creative stream')
    parser.add_argument('--dps_mask_type', default='retouch', type=str)
    parser.add_argument('--dps_retouch_v', default=0, type=float)
    parser.add_argument('--dps_retouch_k', default=2, type=float)
    parser.add_argument('--dps_canny_l', default=10, type=float)
    parser.add_argument('--dps_canny_h', default=10, type=float)

    # fusing stable stream and creative stream
    parser.add_argument('--blend_func', default=['additionEx', 'addition'], nargs='+', type=str,
                        help='the methods of blending creative noise and inversion noise (Equ 5 in paper), including: addition(alpha-composition) additionEx(alpha-composition with calibration)')
    parser.add_argument('--blend_fac', default=['cst', 'retouch'], nargs='+', type=str,
                        help='retouch (the high/low frequency fusion)')
    parser.add_argument('--blend_stop', default=0.02, type=float)
    parser.add_argument('--xt_retouch_v', default=0, type=float,
                        help='the threshold of the operator in frequency decoulping')
    parser.add_argument('--xt_retouch_k', default=3, type=float,
                        help='the kernel size of the operator in frequency decoulping')
    parser.add_argument('--xt_canny_l', default=40, type=float, help='canny opeator')
    parser.add_argument('--xt_canny_h', default=200, type=float, help='canny opeator')
    parser.add_argument('--xt_canny_v', default=0, type=float)
    parser.add_argument('--xt_sobel_s', default=1, type=float)
    parser.add_argument('--xt_tau', default=0.2, type=float, )

    parser.add_argument('--do_xt_multi', action='store_true')
    parser.add_argument('--xt_multi_upscale', default=2, type=float)
    parser.add_argument('--xt_multi_w', default=64, type=int)
    parser.add_argument('--xt_multi_s', default=32, type=int)

    # regularization in denoising
    parser.add_argument('--do_enhance', action='store_true', help='whether using three regularization')
    parser.add_argument('--enhance_sharpen_scale', default=1, type=float, help='Acutance Regularization in paper')
    parser.add_argument('--enhance_deblur_scale', default=0.2, type=float, help='Adversarial Regularization in paper')
    parser.add_argument('--enhance_var', default=20, type=float, help='Distribution Regularization in paper')
    parser.add_argument('--e_d_ex', action='store_true')
    parser.add_argument('--e_v_ex', action='store_true')
    parser.add_argument('--e_s_ex', action='store_true')

    # efficient
    parser.add_argument('--jump_step', default=1, type=int, help='accelerate, reducing denoising steps')
    parser.add_argument('--enhance_jump', action='store_true', help='accelerate, iteratively apply regularization')

    parser.add_argument('--png', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--output_prefix', default='', type=str)
    parser.add_argument('--limit', default='', type=str)
    parser.add_argument('--limit_num', default=[], nargs='+', type=int)

    args, unknown = parser.parse_known_args()
    return args

if __name__ == '__main__':
    # 1. load args
    args = get_args()
    
    # 2. load pipeline
    scheduler = DDIMScheduler.from_pretrained(args.ckpt, subfolder="scheduler", set_alpha_to_one=True)
  
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, variant="fp16",scheduler=scheduler,add_watermarker=False,local_files_only=True, use_safetensors=True, use_auth_token=True
    ).to("cuda")

    sub_path = args.output_dir
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    # 3. dataset
    dataset = HPSv2Dataset(args.data_path)

    # 4.img2img pipeline
    prompt = ''
    negative_prompt = ''
    output_type = 'pil'
    num_inference_steps = args.num_steps
    strength = args.strength

    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description('Processing:')
        for i, (image,name) in enumerate(dataset):
            seed_everything(args.seed)
            sub_type = args.data_path.split('/')[-1]

            # 4.1. stable stream noise:
            in_xt, in_eps, in_latents, in_timesteps = ddim_inverse(pipe, image, args, prompt, negative_prompt)

            # 4.2. get blend function and factors
            do_in_sbs = args.do_in_sbs
            blend_list = get_xt_blend(pipe, image, args)
            blend_idx = 0

            ## 4.3. get sdxl img2img needed attribute
            (
                init_latents_wnoise, init_latents,
                prompt_embeds,
                added_cond_kwargs,
                do_classifier_free_guidance,
                num_inference_steps,
                timesteps
            ) = get_sdxl_needed_attr(pipe, image, prompt, negative_prompt, num_inference_steps, strength,
                                     args.guidance_scale)

            num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
            timestep_cond = None

            #4.4 creative stream noise
            if args.xt_type == 'in_xt':
                latents = in_xt # from ddim inversion
            elif args.xt_type == 'q_sample':
                latents = init_latents_wnoise # from q sample
            elif args.xt_type == 'over_noising':
                latents, over_xt_x0 = over_noising(pipe, image, prompt, negative_prompt, args, in_xt, timesteps[0]) # from pure Gaussian noise

            if do_in_sbs:
                blend_stop = int(args.blend_stop * len(timesteps))
                in_timesteps = in_timesteps[:blend_stop]
                in_latents = in_latents[:blend_stop]

            if args.do_enhance:
                latents = latents.requires_grad_(True)

            # 4.5 jumping for efficient
            timesteps_ = timesteps[::args.jump_step]
            pipe.scheduler.num_inference_steps = 100 // args.jump_step
            num_inference_steps_ = num_inference_steps // args.jump_step

            # 4.6 denoising
            with tqdm(total=num_inference_steps_) as progress_bar:
                for i, t in enumerate(timesteps_):
                    if args.enhance_jump:
                        if i % 2 == 0:
                            args.do_enhance = True
                            latents = latents.requires_grad_(True)
                        else:
                            args.do_enhance = False
                            latents_ = latents.detach()
                            del latents
                            latents = latents_
                    # 4.6.1 blending
                    with torch.enable_grad() if args.do_enhance else torch.no_grad():
                        if do_in_sbs and blend_idx < len(in_timesteps) and t == in_timesteps[blend_idx]:
                            for (blend_fac, blend_func) in blend_list:
                                latents = blend_func(in_latents[blend_idx],
                                                     latents,
                                                     blend_fac[blend_idx])
                            blend_idx = blend_idx + 1

                        x_prev = latents

                        # expand the latents if we are doing classifier free guidance
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

                        # 4.6.1 dps enhance
                        x_0_hat = ddim_step(pipe.scheduler, noise_pred, t, latents, prev_timestep=0)
                        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                        x_t = latents
                        if args.do_enhance:
                            latents = dps_enhance(pipe, args, x_prev, x_t, x_0_hat, noise_pred)

                        del x_0_hat, x_t, x_prev

                        if i == len(timesteps) - 1 or (
                                (i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                            progress_bar.update()

                out_image = latent2img(latents, pipe)

                out_image.save(os.path.join(sub_path, name + '.jpg'))
            pbar.update(1)
