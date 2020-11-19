import cv2
import pickle
from utils.utils_debug import pre_process, post_process, pre_process_kernel, post_process_kernel, PSNR, get_lr_blurdown
from bicubic_downsample.sisr_L2_cg import sisr_L2_cg

if __name__ == '__main__':
    scale = 4
    img_GT = cv2.imread('./demo_data/baby_gt.png')
    kernel = pickle.load(open('./demo_data/baby_kernel.pkl', 'rb'))

    img_L = get_lr_blurdown(img_GT, kernel, scale, dowsammple_method='bicubic')

    img_L = pre_process(img_L.astype('float32')).cuda()
    kernel = pre_process_kernel(kernel).cuda()

    img_E_cg = sisr_L2_cg(y=img_L, kernel=kernel, scale=scale, max_iter=80, gamma=0.001)

    img_L = post_process(img_L)
    img_E_cg = post_process(img_E_cg)
    kernel_img = post_process_kernel(kernel)

    shave = 4
    psnr_cg = PSNR(img_E_cg[shave:-shave, shave:-shave, :], img_GT[shave:-shave, shave:-shave, :])
    print('psnr_cg', psnr_cg)

    cv2.imwrite('./demo_data/baby_lrx{}.png'.format(scale), img_L)
    cv2.imwrite('./demo_data/baby_deconv_cg.png', img_E_cg)
    cv2.imwrite('./demo_data/baby_kernel_img.png', kernel_img)
