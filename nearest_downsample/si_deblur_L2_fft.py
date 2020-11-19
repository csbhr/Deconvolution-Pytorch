from utils.utils_fft import *


def si_deblur_L2_fft(y, k, gamma=0.02):
    '''
    y: LR image, tensor, NxCxWxH
    k: kernel, tensor, Nx(1,3)xwxh
    gamma: float
    '''

    # warp boundary
    w_ori, h_ori = y.shape[-2:]
    img_wrap = wrap_boundary_tensor(y, [int(np.ceil(w_ori / 8 + 2) * 8), int(np.ceil(h_ori / 8 + 2) * 8)])
    img_wrap[:, :, :w_ori, :h_ori] = y
    y = img_wrap

    # initialization & pre-calculation
    w, h = y.shape[-2:]
    FB = p2o(k, (w, h))
    FBC = cconj(FB, inplace=False)
    F2B = r2c(cabs2(FB))

    g1_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).reshape((1, 1, 3, 3))).type_as(k)
    g2_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).reshape((1, 1, 3, 3))).type_as(k)
    FG1 = p2o(g1_kernel, (w, h))
    FG2 = p2o(g2_kernel, (w, h))
    F2G1 = r2c(cabs2(FG1))
    F2G2 = r2c(cabs2(FG2))

    FBFy = cmul(FBC, torch.rfft(y, 2, onesided=False))
    invW = F2B + gamma * (F2G1 + F2G2)
    invWBR = cdiv(FBFy, invW)
    Xest = torch.irfft(invWBR, 2, onesided=False)

    Xest = Xest[:, :, :w_ori, :h_ori]

    return Xest
