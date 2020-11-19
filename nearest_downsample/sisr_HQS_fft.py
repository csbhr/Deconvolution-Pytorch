from utils.utils_fft import *


def sisr_HQS_fft(y, xp, k, scale, alpha=0.1):
    '''
    y: LR image, tensor, NxCxWxH
    xp: SR image from pre-step, tensor, NxCx(W*scale)x(H*scale)
    k: kernel, tensor, Nx(1,3)xwxh
    scale: int
    alpha: float
    '''

    # warp boundary
    w_ori, h_ori = y.shape[-2:]
    img = upsample_nearest(y, scale)
    img = wrap_boundary_tensor(img, [int(np.ceil(scale * w_ori / 8 + 2) * 8), int(np.ceil(scale * h_ori / 8 + 2) * 8)])
    img_wrap = img[:, :, ::scale, ::scale]
    img_wrap[:, :, :w_ori, :h_ori] = y
    y = img_wrap

    img = wrap_boundary_tensor(xp, [int(np.ceil(scale * w_ori / 8 + 2) * 8), int(np.ceil(scale * h_ori / 8 + 2) * 8)])
    img[:, :, :scale * w_ori, :scale * h_ori] = xp
    xp = img

    # initialization & pre-calculation
    w, h = y.shape[-2:]
    FB = p2o(k, (w * scale, h * scale))
    FBC = cconj(FB, inplace=False)
    F2B = r2c(cabs2(FB))
    STy = upsample(y, sf=scale)
    FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))

    FR = FBFy + torch.rfft(alpha * xp, 2, onesided=False)
    x1 = cmul(FB, FR)
    FBR = torch.mean(splits(x1, scale), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, scale), dim=-1, keepdim=False)
    invWBR = cdiv(FBR, csum(invW, alpha))
    FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, scale, scale, 1))
    FX = (FR - FCBinvWBR) / alpha
    Xest = torch.irfft(FX, 2, onesided=False)

    Xest = Xest[:, :, :scale * w_ori, :scale * h_ori]

    return Xest
