import torch.nn.functional as F
import torch
import numpy as np
from deconv.cg.utils_cg import flip_kernel, conv_func, inner_product, downsample, upsample, warp


def dual_conv(x, kernel, mask=None, flow=None, scale=1, psize=0, sample_method='directly'):
    if flow is not None:  # Fx
        x_warp = warp(x[:, :, psize:-psize, psize:-psize], flow['F'])
        x = F.pad(x_warp, (psize, psize, psize, psize), mode='replicate')

    kernel_flip = flip_kernel(kernel)
    x = conv_func(x, kernel_flip, padding='same')  # KFx

    if mask is not None:
        x = x * mask

    if scale != 1:
        x_down = downsample(x[:, :, psize:-psize, psize:-psize], scale, method=sample_method)  # SKFx

        x_up = torch.zeros(x.shape).type_as(x)
        x_up[:, :, psize:-psize, psize:-psize] = upsample(x_down, scale, method=sample_method)  # STSKFx
        x = x_up

    x = conv_func(x, kernel, padding='same')  # KTSTSKFx

    if flow is not None:  # FTKTSTSKFx
        x_warp = warp(x[:, :, psize:-psize, psize:-psize], flow['FT'])
        x = F.pad(x_warp, (psize, psize, psize, psize), mode='replicate')

    return x


def vsr_flow_L2_cg(y, kernel, adj_list, flow_list, x_init=None, scale=4, max_iter=80, gamma=0.01, wei=0.0001,
                   sample_method='directly'):
    kernel = flip_kernel(kernel)
    g1_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).reshape((1, 1, 3, 3))).type_as(kernel)
    g2_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).reshape((1, 1, 3, 3))).type_as(kernel)

    _, _, _, ksize = kernel.size()
    psize = ksize // 2
    assert ksize % 2 == 1, "only support odd kernel size!"
    assert len(adj_list) == len(flow_list), "every adjacent frame should provide bi-flow!"

    y_up = upsample(y, scale, method=sample_method)

    adj_up_list = []
    for adj in adj_list:
        adj_up = torch.zeros(b, c, h * scale, w * scale).type_as(adj)
        adj_up[:, :, ::scale, ::scale] = adj
        adj_up_list.append(adj_up)

    y_up = F.pad(y_up, (psize, psize, psize, psize), mode='replicate')  # STy
    adj_up_list = [F.pad(adj_up, (psize, psize, psize, psize), mode='replicate') for adj_up in adj_up_list]  # STyi
    mask = torch.zeros_like(y_up).type_as(y_up)
    mask[:, :, psize:-psize, psize:-psize] = 1.

    b = conv_func(y_up * mask, kernel, padding='same')  # KTSTy
    for adj_up, flow in zip(adj_up_list, flow_list):
        adj_k = conv_func(adj_up * mask, kernel, padding='same')  # KTSTyi
        adj_warp = warp(adj_k[:, :, psize:-psize, psize:-psize], flow['FT'])  # FTKTSTyi
        adj_b = F.pad(adj_warp, (psize, psize, psize, psize), mode='replicate')
        b = b + wei * adj_b

    if x_init is not None:
        x = F.pad(x_init, (psize, psize, psize, psize), mode='replicate')
    else:
        x = y_up
    Ax = dual_conv(x, kernel=kernel, mask=mask, scale=scale, psize=psize, sample_method=sample_method)
    for flow in flow_list:
        Ax_f = dual_conv(x, kernel=kernel, mask=mask, flow=flow, scale=scale, psize=psize)
        Ax = Ax + wei * Ax_f
    Ax = Ax + gamma * (dual_conv(x, kernel=g1_kernel) + dual_conv(x, kernel=g2_kernel))

    r = b - Ax
    for i in range(max_iter):
        rho = inner_product(r, r)
        if i == 0:
            p = r
        else:
            beta = rho / rho_1
            p = r + beta * p

        Ap = dual_conv(p, kernel=kernel, mask=mask, scale=scale, psize=psize, sample_method=sample_method)
        for flow in flow_list:
            Ap_f = dual_conv(p, kernel=kernel, mask=mask, flow=flow, scale=scale, psize=psize)
            Ap = Ap + wei * Ap_f
        Ap = Ap + gamma * (dual_conv(p, kernel=g1_kernel) + dual_conv(p, kernel=g2_kernel))

        q = Ap
        alp = rho / inner_product(p, q)
        x = x + alp * p
        r = r - alp * q
        rho_1 = rho

    deconv_result = x[:, :, psize:-psize, psize:-psize]

    return deconv_result
