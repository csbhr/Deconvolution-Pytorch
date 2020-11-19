import torch.nn.functional as F
import torch
import numpy as np


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


def dual_conv(x, kernel, mask=None):
    kernel_flip = flip_kernel(kernel)

    x = conv_func(x, kernel_flip, padding='same')

    if mask is not None:
        x = x * mask

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


def si_deblur_L2_cg(y, kernel, max_iter=80, gamma=0.01):
    kernel = flip_kernel(kernel)
    g1_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).reshape((1, 1, 3, 3))).type_as(kernel)
    g2_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).reshape((1, 1, 3, 3))).type_as(kernel)

    _, _, _, ksize = kernel.size()
    psize = ksize // 2
    assert ksize % 2 == 1, "only support odd kernel size!"

    y = F.pad(y, (psize, psize, psize, psize), mode='replicate')
    mask = torch.zeros_like(y).type_as(y)
    mask[:, :, psize:-psize, psize:-psize] = 1.

    b = conv_func(y * mask, kernel, padding='same')

    x = y
    Ax = dual_conv(x, kernel, mask)
    Ax = Ax + gamma * (dual_conv(x, g1_kernel) + dual_conv(x, g2_kernel))

    r = b - Ax
    for i in range(max_iter):
        rho = inner_product(r, r)
        if i == 0:
            p = r
        else:
            beta = rho / rho_1
            p = r + beta * p

        Ap = dual_conv(p, kernel, mask)
        Ap = Ap + gamma * (dual_conv(p, g1_kernel) + dual_conv(p, g2_kernel))

        q = Ap
        alp = rho / inner_product(p, q)
        x = x + alp * p
        r = r - alp * q
        rho_1 = rho

    deconv_result = x[:, :, psize:-psize, psize:-psize]

    return deconv_result
