import torch.nn.functional as F
import torch
import numpy as np


def downsample(x, scale, method='directly'):
    if method == 'directly' or method == 'nearest':
        x_down = x[:, :, ::scale, ::scale]
    elif method == 'bicubic':
        x_down = F.interpolate(x, scale_factor=1. / scale, mode='bicubic', align_corners=False)
    else:
        raise Exception('Not support sample method {}'.format(method))

    return x_down


def upsample(x, scale, method='directly'):
    if method == 'directly':
        b, c, h, w = x.size()
        x_up = torch.zeros(b, c, h * scale, w * scale).type_as(x)
        x_up[:, :, ::scale, ::scale] = x
    elif method == 'nearest':
        x_repeat = x.repeat(1, scale * scale, 1, 1)
        x_up = F.pixel_shuffle(x_repeat, scale)
    elif method == 'bicubic':
        x_up = F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False)
    else:
        raise Exception('Not support sample method {}'.format(method))

    return x_up


def flip_kernel(kernel):
    kernel_numpy = np.ascontiguousarray(kernel.detach().cpu().numpy()[:, :, ::-1, ::-1])
    kernel_flip = torch.from_numpy(kernel_numpy).type_as(kernel)
    return kernel_flip


def conv_func(x, kernel, padding='same'):
    b, c, h, w = x.size()
    _, _, _, ksize = kernel.size()

    if padding == 'same':
        padding = ksize // 2
    elif padding == 'valid':
        padding = 0
    else:
        raise Exception("not support padding flag!")

    kernel_c = torch.zeros(c, c, ksize, ksize).type_as(kernel)
    for i in range(c):
        kernel_c[i, i, :, :] = kernel[0, 0, :, :]
    conv_result = F.conv2d(x, kernel_c, bias=None, stride=1, padding=padding)

    return conv_result


def inner_product(x1, x2):
    b, c, h, w = x1.size()
    x1 = x1.view(b, -1)
    x2 = x2.view(b, -1)
    re = x1 * x2
    re = torch.sum(re, dim=1)
    re = re.view(b, 1, 1, 1)
    return re


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    output = output * mask

    return output
