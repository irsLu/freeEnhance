from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from diffusers.utils import load_image
import PIL

def anyimage2npL(image):
    if type(image) == str:
        image_np = np.asarray(load_image(image).convert('L').resize((1024, 1024), 3))
    elif type(image) == PIL.Image.Image:
        image_np = np.asarray(image.convert('L'))
    else:
        image_np = image
    return image_np


def retouch_mask(pipe, image_np, retouch_kernel=2):
    np_mask = get_retouch_mask(image_np, retouch_kernel)
    return np_mask


def canny_mask(pipe, image_np, canny_l=10, canny_h=10):
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mask = cv2.Canny(image, threshold1=canny_l, threshold2=canny_h)
    return mask


def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


def get_retouch_mask(img_input: np.ndarray, kernel_size: int) -> np.ndarray:
    '''
    Return the area where the image is retouched.
    Copy from Zhihu.com
    '''
    step = 1
    kernel = (kernel_size, kernel_size)

    img = img_input.astype(np.float32) / 255.0
    sz = img.shape[:2]
    sz1 = (int(round(sz[1] * step)), int(round(sz[0] * step)))
    sz2 = (int(round(kernel[0] * step)), int(round(kernel[0] * step)))
    sI = cv2.resize(img, sz1, interpolation=cv2.INTER_LINEAR)
    sp = cv2.resize(img, sz1, interpolation=cv2.INTER_LINEAR)
    msI = cv2.blur(sI, sz2)
    msp = cv2.blur(sp, sz2)
    msII = cv2.blur(sI * sI, sz2)
    msIp = cv2.blur(sI * sp, sz2)
    vsI = msII - msI * msI
    csIp = msIp - msI * msp
    recA = csIp / (vsI + 0.01)
    recB = msp - recA * msI
    mA = cv2.resize(recA, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR)
    mB = cv2.resize(recB, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR)

    gf = mA * img + mB
    gf -= img
    gf *= 255
    gf = gf.astype(np.uint8)
    gf = gf.clip(0, 255)
    gf = gf.astype(np.float32) / 255.0
    return gf


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst