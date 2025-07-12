import numpy as np

def control_region_mask(framesize, coronrad, seeinglimited):
    center_x, center_y = framesize[0] // 2, framesize[1] // 2
    mask_ctrl_reg = np.ones(framesize)
    x = np.arange(framesize[0], dtype=float)[None,:] - center_x
    y = np.arange(framesize[1], dtype=float)[:,None] - center_y
    rho2d = np.sqrt(x**2 + y**2)
    mask_ctrl_reg[np.where(rho2d > seeinglimited)] = 0.
    # mask_ctrl_reg[int(ALIGNED_CENTER[0] - owa / 2):int(ALIGNED_CENTER[0] + owa / 2),int(ALIGNED_CENTER[1] - owa / 2):int(ALIGNED_CENTER[1] + owa / 2)] = 0.
    # mask_ctrl_reg[mask_ctrl_reg == 0.] = np.nan
    mask_ctrl_reg[np.where(rho2d < coronrad)] = 0.
    # mask_ctrl_reg = 1 - mask_ctrl_reg
    # mask_ctrl_reg = rotate(mask_ctrl_reg,angle=-62, reshape=False, order=0)
    # mask_ctrl_reg = 1 - mask_ctrl_reg
    mask_ctrl_reg[mask_ctrl_reg > 0.5] = 1
    mask_ctrl_reg[mask_ctrl_reg < 0.5] = 0
    return mask_ctrl_reg