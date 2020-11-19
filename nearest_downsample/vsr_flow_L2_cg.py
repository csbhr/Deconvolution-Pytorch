import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


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


def dual_conv(x, kernel, mask=None, flow=None, scale=1, psize=0):
    if flow is not None:  # Fx
        x_warp = warp(x[:, :, psize:-psize, psize:-psize], flow['F'])
        x = F.pad(x_warp, (psize, psize, psize, psize), mode='replicate')

    kernel_flip = flip_kernel(kernel)
    x = conv_func(x, kernel_flip, padding='same')  # KFx

    if mask is not None:
        x = x * mask

    if scale != 1:
        x_down = x[:, :, psize:-psize:scale, psize:-psize:scale]  # SKFx

        x_up = torch.zeros(x.shape).type_as(x)
        x_up[:, :, psize:-psize:scale, psize:-psize:scale] = x_down  # STSKFx
        x = x_up

    x = conv_func(x, kernel, padding='same')  # KTSTSKFx

    if flow is not None:  # FTKTSTSKFx
        x_warp = warp(x[:, :, psize:-psize, psize:-psize], flow['FT'])
        x = F.pad(x_warp, (psize, psize, psize, psize), mode='replicate')

    return x


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


def vsr_flow_L2_cg(y, kernel, adj_list, flow_list, x_init=None, scale=4, max_iter=80, gamma=0.01, wei=0.0001):
    kernel = flip_kernel(kernel)
    g1_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).reshape((1, 1, 3, 3))).type_as(kernel)
    g2_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).reshape((1, 1, 3, 3))).type_as(kernel)

    _, _, _, ksize = kernel.size()
    psize = ksize // 2
    assert ksize % 2 == 1, "only support odd kernel size!"
    assert len(adj_list) == len(flow_list), "every adjacent frame should provide bi-flow!"

    b, c, h, w = y.size()
    y_up = torch.zeros(b, c, h * scale, w * scale).type_as(y)
    y_up[:, :, ::scale, ::scale] = y

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
    Ax = dual_conv(x, kernel=kernel, mask=mask, scale=scale, psize=psize)
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

        Ap = dual_conv(p, kernel=kernel, mask=mask, scale=scale, psize=psize)
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
