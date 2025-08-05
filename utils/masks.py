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

def make_annular_mask(dimensions, inner_radius, outer_radius, center=None):
    """
    Create a binary annular mask for a 2D array.
    
    The pixels whose distance from the center is between inner_radius and outer_radius
    (inclusive) are set to 1; all other pixels are set to 0.
    
    Args:
        dimensions (tuple): (height, width) of the output mask.
        inner_radius (float): The inner radius (in pixels).
        outer_radius (float): The outer radius (in pixels).
        center (tuple, optional): (row, col) coordinates for the center.
                                  If None, defaults to the center of the array.
    
    Returns:
        jnp.ndarray: A binary mask with shape `dimensions` (1's inside the annulus, 0's outside).
    """
    H, W = dimensions
    if center is None:
        center = (H / 2 - 0.5, W / 2 - 0.5)
    
    # Create coordinate grid
    y, x = np.indices((H, W))
    # Compute the radial distance from the center for each pixel.
    # Note: center is given as (row, col) and x corresponds to column indices.
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    # Create the binary mask: 1 inside the annulus, 0 elsewhere.
    mask = np.where((r >= inner_radius) & (r <= outer_radius), 1, 0)
    return mask