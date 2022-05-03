import math
import numpy as np
from ignite.metrics import SSIM


def PSNRsim(image1, image2):
    
    array1=image1.detach().numpy().transpose(1, 2, 0)
    array2=image2.detach().numpy().transpose(1, 2, 0)
    img_diff = array1 - array2
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(1 / rmse)
        return PSNR
    
def SSIM(image1,image2):
    ssim = SSIM(range=1.0)
    ssim.update(image1,image2)
    return ssim.compute()
    
    
    

    

    