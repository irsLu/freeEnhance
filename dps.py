from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

import torch.nn as nn
import numpy as np
import torch
import PIL
from utils import gaussian_blur_2d


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


def dps_enhance_sharpen(pipe, args, x_prev, x_t, x_0_hat):
    x_0_hat = x_0_hat[:]
    r = x_0_hat[:, :1, :, :]
    r = torch_sobel(r.to(torch.float32))
    r = r.abs()
    r_ = r.detach()
    r_ = ((r_.max() - r_) ** 1)
    r_ = r_ / r_.max()
    scale = args.enhance_sharpen_scale * r_

    return (r * (-scale)).mean()


def dps_enhance_sharpenEx(pipe, args, x_prev, x_t, x_0_hat):
    x_0_hat = x_0_hat[:]
    r = x_0_hat[:, :1, :, :]
    r = torch_sobel(r.to(torch.float32))
    r = r.abs()

    r_ = r.detach()
    r_ = ((r_.max() - r_) ** 1)
    r_ = r_ / r_.max()
    scale = args.enhance_sharpen_scale * r_

    probs = torch.histc(torch.flatten(r), int(r.max()))
    samples = torch.multinomial(probs, 2, replacement=False)

    mmin, mmax = samples.min(), samples.max()

    if mmin == mmax:
        mask = ((r > (mmin - 0.5)) & (r < (mmax + 0.5))).to(torch.float16)
    else:
        mask = ((r > mmin) & (r < mmax)).to(torch.float16)

    return (r * (-scale) * mask).mean()




def dps_enhance_deblur(pipe, args, x_prev, x_t, x_0_hat):
    y = gaussian_blur_2d(x_0_hat, 3, 1).detach()

    difference = x_0_hat - y
    norm = torch.linalg.norm(difference)
    return norm * - args.enhance_deblur_scale


def dps_enhance_deblurEx(pipe, args, x_prev, x_t, x_0_hat):
    r = x_0_hat[:, :1, :, :].detach()
    r = torch_sobel(r.to(torch.float32))
    r = r.abs()
    mask = (r > r.mean()).to(torch.float16)
    y = gaussian_blur_2d(x_0_hat, 3, 1).detach()
    difference = (x_0_hat - y) * mask
    norm = torch.linalg.norm(difference)
    return norm * - args.enhance_deblur_scale


def dps_enhance_var(args, eps, x_0_hat):
    y = eps.var()
    difference = 1 - y
    norm = torch.linalg.norm(difference)

    return norm * -args.enhance_var


def dps_enhance_varEx(args, eps, x_0_hat):
    r = x_0_hat[:, :1, :, :]
    r = torch_sobel(r.to(torch.float32))
    r = r.abs()

    mask = (r > r.mean()).to(torch.float16).detach()
    y = (eps * mask).var()
    difference = 1 - y
    norm = torch.linalg.norm(difference)
    return (norm) * -args.enhance_var


torch_sobel = None

def dps_enhance(pipe, args, x_prev, x_t, x_0_hat, eps):
    global torch_sobel
    if torch_sobel is None:
        _torch_sobel = Sobel()
        torch_sobel = _torch_sobel.eval().to(pipe.device)

    if args.e_d_ex:
        grad_blur = dps_enhance_deblurEx(pipe, args, x_prev, x_t, x_0_hat)
    else:
        grad_blur = dps_enhance_deblur(pipe, args, x_prev, x_t, x_0_hat)

    if args.e_s_ex:
        grad_sharpen = dps_enhance_sharpenEx(pipe, args, x_prev, x_t, x_0_hat)
    else:
        grad_sharpen = dps_enhance_sharpen(pipe, args, x_prev, x_t, x_0_hat)

    if args.e_v_ex:
        grad_var = dps_enhance_varEx(args, eps, x_0_hat)
    else:
        grad_var = dps_enhance_var(args, eps, x_0_hat)

    loss = grad_blur + grad_sharpen + grad_var
    norm_grad = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]

    x_t = x_t.to(torch.float32)
    norm_grad = norm_grad.to(torch.float32)
    if torch.isnan(norm_grad).any():
        print("norm_grad has nan")
        norm_grad = torch.where(torch.isnan(norm_grad), torch.full_like(norm_grad, 0), norm_grad)
    x_t_ = x_t - norm_grad
    if torch.isnan(x_t_).any():
        print("xt has nan")
        x_t_ = torch.where(torch.isnan(x_t_), x_t, x_t_)
    return x_t_.to(torch.float16)



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
