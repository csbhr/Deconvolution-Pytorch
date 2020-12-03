import torch.nn.functional as F
import torch
from deconv.cg.utils_cg import flip_kernel, conv_func, inner_product, downsample, upsample


def dual_conv(x, kernel, mask=None, scale=1, psize=0, sample_method='directly'):
    kernel_flip = flip_kernel(kernel)

    x = conv_func(x, kernel_flip, padding='same')

    if mask is not None:
        x = x * mask

    if scale != 1:
        x_down = downsample(x[:, :, psize:-psize, psize:-psize], scale, method=sample_method)  # downsample

        x_up = torch.zeros(x.shape).type_as(x)  # upsample
        x_up[:, :, psize:-psize, psize:-psize] = upsample(x_down, scale, method=sample_method)
        x = x_up

    x = conv_func(x, kernel, padding='same')

    return x


def sisr_HQS_cg(y, xp, kernel, scale, max_iter=80, alpha=0.01, sample_method='directly'):
    kernel = flip_kernel(kernel)

    _, _, _, ksize = kernel.size()
    psize = ksize // 2
    assert ksize % 2 == 1, "only support odd kernel size!"

    y_up = upsample(y, scale, method=sample_method)

    y_up = F.pad(y_up, (psize, psize, psize, psize), mode='replicate')
    xp = F.pad(xp, (psize, psize, psize, psize), mode='replicate')
    mask = torch.zeros_like(y_up).type_as(y_up)
    mask[:, :, psize:-psize, psize:-psize] = 1.

    b = conv_func(y_up * mask, kernel, padding='same') + alpha * xp

    x = y_up
    Ax = dual_conv(x, kernel, mask, scale, psize, sample_method=sample_method)
    Ax = Ax + alpha * x

    r = b - Ax
    for i in range(max_iter):
        rho = inner_product(r, r)
        if i == 0:
            p = r
        else:
            beta = rho / rho_1
            p = r + beta * p

        Ap = dual_conv(p, kernel, mask, scale, psize, sample_method=sample_method)
        Ap = Ap + alpha * p

        q = Ap
        alp = rho / inner_product(p, q)
        x = x + alp * p
        r = r - alp * q
        rho_1 = rho

    deconv_result = x[:, :, psize:-psize, psize:-psize]

    return deconv_result
