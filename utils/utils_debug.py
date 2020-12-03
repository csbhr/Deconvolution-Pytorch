import numpy as np
import math
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def pre_process(img):
    img = torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float().mul_(1. / 255)
    img = img.unsqueeze(0)
    return img


def post_process(img):
    img = img.mul(255.).clamp(0, 255).round()
    img = np.transpose(img[0].detach().data.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img


def pre_process_kernel(kernel):
    kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()
    return kernel


def post_process_kernel(kernel):
    mi, ma = torch.min(kernel), torch.max(kernel)
    kernel = (kernel - mi) / (ma - mi)
    kernel = kernel.mul(255.).clamp(0, 255).round()
    kernel = torch.cat([kernel, kernel, kernel], dim=1)
    kernel = np.transpose(kernel[0].detach().data.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return kernel


def PSNR(img1, img2):
    '''
    img1, img2: [0, 255]
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def get_lr_blurdown(img_gt, kernel, scale, dowsammple_method='nearest'):
    img_gt = np.array(img_gt).astype('float32')
    gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).float()

    kernel_size = kernel.shape[0]
    psize = kernel_size // 2
    gt_tensor = F.pad(gt_tensor, (psize, psize, psize, psize), mode='replicate')

    gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) // 2), bias=False)
    nn.init.constant_(gaussian_blur.weight.data, 0.0)
    gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

    blur_tensor = gaussian_blur(gt_tensor)
    blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]
    blur_tensor = blur_tensor.clamp(0, 255).round()
    blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('float32')

    if dowsammple_method == 'nearest':
        blurdown = blur[::scale, ::scale, :]
    elif dowsammple_method == 'bicubic':
        blurdown = cv2.resize(blur, dsize=(0, 0), fx=1. / scale, fy=1. / scale, interpolation=cv2.INTER_CUBIC)
    else:
        raise Exception('Not support downsample method "{}"'.format(dowsammple_method))

    blur = blur.astype('uint8')
    blurdown = blurdown.astype('uint8')

    return blur, blurdown


def image_shift_numpy(img, offset_x=0., offset_y=0.):
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    B, C, H, W = img_tensor.size()

    # init flow
    flo = torch.ones(B, 2, H, W).type_as(img_tensor)
    flo[:, 0, :, :] *= offset_x
    flo[:, 1, :, :] *= offset_y

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().type_as(img_tensor)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # Interpolation
    vgrid = vgrid.permute(0, 2, 3, 1)
    output_tensor = F.grid_sample(img_tensor, vgrid, padding_mode='border')

    output = output_tensor.round()[0].detach().numpy().transpose(1, 2, 0).astype(img.dtype)

    return output


def image_shift_tensor(img_tensor, offset_x=0., offset_y=0.):
    B, C, H, W = img_tensor.size()

    # init flow
    flo = torch.ones(B, 2, H, W).type_as(img_tensor)
    flo[:, 0, :, :] *= offset_x
    flo[:, 1, :, :] *= offset_y

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().type_as(img_tensor)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # Interpolation
    vgrid = vgrid.permute(0, 2, 3, 1)
    output_tensor = F.grid_sample(img_tensor, vgrid, padding_mode='border')

    return output_tensor
