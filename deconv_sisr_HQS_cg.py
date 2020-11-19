import torch.nn.functional as F
import torch
import numpy as np


def flip_kernel(kernel):
    kernel_numpy = np.ascontiguousarray(kernel.cpu().numpy()[:, :, ::-1, ::-1])
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


def dual_conv(x, kernel, mask=None, scale=1, psize=0):
    kernel_flip = flip_kernel(kernel)

    x = conv_func(x, kernel_flip, padding='same')

    if mask is not None:
        x = x * mask

    if scale != 1:
        x_down = x[:, :, psize:-psize:scale, psize:-psize:scale]  # downsample

        x_up = torch.zeros(x.shape).type_as(x)  # upsample
        x_up[:, :, psize:-psize:scale, psize:-psize:scale] = x_down
        x = x_up

    x = conv_func(x, kernel, padding='same')

    return x


def inner_product(x1, x2):
    b, c, h, w = x1.size()
    x1 = x1.view(b, -1)
    x2 = x2.view(b, -1)
    re = x1 * x2
    re = torch.sum(re, dim=1)
    re = re.view(b, 1, 1, 1)
    return re


def deconv_sisr_HQS_cg(y, xp, kernel, scale, max_iter=80, alpha=0.01):
    kernel = flip_kernel(kernel)

    _, _, _, ksize = kernel.size()
    psize = ksize // 2
    assert ksize % 2 == 1, "only support odd kernel size!"

    b, c, h, w = y.size()
    y_up = torch.zeros(b, c, h * scale, w * scale).type_as(y)
    y_up[:, :, ::scale, ::scale] = y

    y_up = F.pad(y_up, (psize, psize, psize, psize), mode='replicate')
    xp = F.pad(xp, (psize, psize, psize, psize), mode='replicate')
    mask = torch.zeros_like(y_up).type_as(y_up)
    mask[:, :, psize:-psize, psize:-psize] = 1.

    b = conv_func(y_up * mask, kernel, padding='same') + alpha * xp

    x = y_up
    Ax = dual_conv(x, kernel, mask, scale, psize)
    Ax = Ax + alpha * x

    r = b - Ax
    for i in range(max_iter):
        rho = inner_product(r, r)
        if i == 0:
            p = r
        else:
            beta = rho / rho_1
            p = r + beta * p

        Ap = dual_conv(p, kernel, mask, scale, psize)
        Ap = Ap + alpha * p

        q = Ap
        alp = rho / inner_product(p, q)
        x = x + alp * p
        r = r - alp * q
        rho_1 = rho

    deconv_result = x[:, :, psize:-psize, psize:-psize]

    return deconv_result
