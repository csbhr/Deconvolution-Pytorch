import cv2
import pickle
import deconv
from utils.utils_debug import pre_process, post_process, pre_process_kernel, post_process_kernel, PSNR, get_lr_blurdown


if __name__ == '__main__':
    scale = 4
    img_GT = cv2.imread('./demo_data/baby_gt.png')
    kernel = pickle.load(open('./demo_data/baby_kernel.pkl', 'rb'))

    # get blur image and low-resolution image
    img_Blur, img_L = get_lr_blurdown(img_GT, kernel, scale)
    # for Half Quadratic Splitting (HQS)
    img_Xp = cv2.resize(img_L, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    img_Blur = pre_process(img_Blur.astype('float32')).cuda()
    img_L = pre_process(img_L.astype('float32')).cuda()
    img_Xp = pre_process(img_Xp.astype('float32')).cuda()
    kernel = pre_process_kernel(kernel).cuda()

    shave = 4  # for calculate PSNR

    # single image deblurring using CG with Gaussian prior
    img_E_deblur_cg_L2 = deconv.cg.si_deblur_L2(y=img_Blur, kernel=kernel, max_iter=80, gamma=0.001)
    img_E_deblur_cg_L2 = post_process(img_E_deblur_cg_L2)
    psnr_deblur_cg_L2 = PSNR(img_E_deblur_cg_L2[shave:-shave, shave:-shave, :], img_GT[shave:-shave, shave:-shave, :])
    print('PSNR-deblur-cg-L2:', psnr_deblur_cg_L2)
    cv2.imwrite('./demo_data/baby_deconv_deblur_cg_L2.png', img_E_deblur_cg_L2)

    # single image deblurring using FFT with Gaussian prior
    img_E_deblur_fft_L2 = deconv.fft.si_deblur_L2(y=img_Blur, kernel=kernel, gamma=0.001)
    img_E_deblur_fft_L2 = post_process(img_E_deblur_fft_L2)
    psnr_deblur_fft_L2 = PSNR(img_E_deblur_fft_L2[shave:-shave, shave:-shave, :], img_GT[shave:-shave, shave:-shave, :])
    print('PSNR-deblur-fft-L2:', psnr_deblur_fft_L2)
    cv2.imwrite('./demo_data/baby_deconv_deblur_fft_L2.png', img_E_deblur_fft_L2)

    # single image sr using CG with Gaussian prior
    img_E_sisr_cg_L2 = deconv.cg.sisr_L2(y=img_L, kernel=kernel, scale=scale, max_iter=80, gamma=0.001)
    img_E_sisr_cg_L2 = post_process(img_E_sisr_cg_L2)
    psnr_sisr_cg_L2 = PSNR(img_E_sisr_cg_L2[shave:-shave, shave:-shave, :], img_GT[shave:-shave, shave:-shave, :])
    print('PSNR-sisr-cg-L2:', psnr_sisr_cg_L2)
    cv2.imwrite('./demo_data/baby_deconv_sisr_cg_L2.png', img_E_sisr_cg_L2)

    # single image sr using FFT with Gaussian prior
    img_E_sisr_fft_L2 = deconv.fft.sisr_L2(y=img_L, kernel=kernel, scale=scale, gamma=0.001)
    img_E_sisr_fft_L2 = post_process(img_E_sisr_fft_L2)
    psnr_sisr_fft_L2 = PSNR(img_E_sisr_fft_L2[shave:-shave, shave:-shave, :], img_GT[shave:-shave, shave:-shave, :])
    print('PSNR-sisr-fft-L2:', psnr_sisr_fft_L2)
    cv2.imwrite('./demo_data/baby_deconv_sisr_fft_L2.png', img_E_sisr_fft_L2)

    # single image sr using CG with Half Quadratic Splitting (HQS)
    img_E_sisr_cg_HQS = deconv.cg.sisr_HQS(y=img_L, xp=img_Xp, kernel=kernel, scale=scale, max_iter=80, alpha=0.001)
    img_E_sisr_cg_HQS = post_process(img_E_sisr_cg_HQS)
    psnr_sisr_cg_HQS = PSNR(img_E_sisr_cg_HQS[shave:-shave, shave:-shave, :], img_GT[shave:-shave, shave:-shave, :])
    print('PSNR-sisr-cg-HQS:', psnr_sisr_cg_HQS)
    cv2.imwrite('./demo_data/baby_deconv_sisr_cg_HQS.png', img_E_sisr_cg_HQS)

    # single image sr using FFT with Half Quadratic Splitting (HQS)
    img_E_sisr_fft_HQS = deconv.fft.sisr_HQS(y=img_L, xp=img_Xp, kernel=kernel, scale=scale, alpha=0.001)
    img_E_sisr_fft_HQS = post_process(img_E_sisr_fft_HQS)
    psnr_sisr_fft_HQS = PSNR(img_E_sisr_fft_HQS[shave:-shave, shave:-shave, :], img_GT[shave:-shave, shave:-shave, :])
    print('PSNR-sisr-fft-HQS:', psnr_sisr_fft_HQS)
    cv2.imwrite('./demo_data/baby_deconv_sisr_fft_HQS.png', img_E_sisr_fft_HQS)

    img_L = post_process(img_L)
    kernel_img = post_process_kernel(kernel)
    cv2.imwrite('./demo_data/baby_lrx{}.png'.format(scale), img_L)
    cv2.imwrite('./demo_data/baby_kernel.png', kernel_img)
