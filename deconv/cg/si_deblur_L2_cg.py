import torch.nn.functional as F
import torch
import numpy as np
from deconv.cg.utils_cg import flip_kernel, conv_func, inner_product


def dual_conv(x, kernel, mask=None):
    kernel_flip = flip_kernel(kernel)

    x = conv_func(x, kernel_flip, padding='same')

    if mask is not None:
        x = x * mask

    x = conv_func(x, kernel, padding='same')

    return x


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
