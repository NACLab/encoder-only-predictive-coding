from jax import lax, jit
import jax.numpy as jnp
from functools import partial

'''
Contains supporting co-routines for measuring properties of visual stimuli (images) 
'''

@jit
def measure_NCC(img1, img2):
    ## computes Normalized Cross-Correlation (NCC) over a batch of images; 
    ### Input layout: (B, C, H, W)
    ### Returns: A 1D vector of length B containing scores between [-1.0, 1.0]
    ###          where 1.0 represents perfect structural correlation.

    ## flatten spatial and channel dimensions for vector math : (B, C, H, W) -> (B, C * H * W)
    flat_shape = (img1.shape[0], -1)
    x = jnp.reshape(img1, flat_shape)
    y = jnp.reshape(img2, flat_shape)
    ## compute sample means along the feature dimension for each batch item
    mean_x = jnp.mean(x, axis=1, keepdims=True)
    mean_y = jnp.mean(y, axis=1, keepdims=True)
    ## center vectors (subtract the mean) to remove global baseline brightness
    x_centered = x - mean_x
    y_centered = y - mean_y
    ## compute cross-covariance (dot product of centered vectors) : resulting shape: (B,)
    covariance = jnp.sum(x_centered * y_centered, axis=1)

    ## compute L2 norms of centered vectors to normalize variance
    eps = 1e-12  # Prevent division by zero
    norm_x = jnp.sqrt(jnp.sum(x_centered ** 2, axis=1)) + eps
    norm_y = jnp.sqrt(jnp.sum(y_centered ** 2, axis=1)) + eps

    ## calculate NCC: covariance divided by product of norms
    ncc = covariance / (norm_x * norm_y)
    return jnp.mean(ncc)

@partial(jit, static_argnums=(2, 3))
def measure_SSIM(
    img1, 
    img2, 
    window_size=11, 
    max_val=1.0
):
    ## computes Structural Similarity Index (SSIM) over a batch of images; 
    ### Input layout: (B, C, H, W)
    ### Range: [0, max_val]

    ## define SSIM stabilizing constants
    K1, K2 = 0.01, 0.03
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2
    ## build a normalized uniform spatial window filter for local pooling
    ### Shape: (1, 1, window_size, window_size) to broadcast across channels
    window = jnp.ones((1, 1, window_size, window_size)) / (window_size ** 2)
    ## define standard 2D convolution parameters to map across spatial boundaries
    dn = lax.ConvDimensionNumbers(
        lhs_spec=(0, 1, 2, 3), rhs_spec=(0, 1, 2, 3), out_spec=(0, 1, 2, 3)
    )
    def local_mean(img):
        return lax.conv_general_dilated(
            img, window, window_strides=(1, 1), padding='SAME', dimension_numbers=dn
        )
    ## calculate local statistical moments
    mu1 = local_mean(img1)
    mu2 = local_mean(img2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = local_mean(img1 ** 2) - mu1_sq
    sigma2_sq = local_mean(img2 ** 2) - mu2_sq
    sigma12 = local_mean(img1 * img2) - mu1_mu2
    
    ## compute SSIM map
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    ## reduce across spatial & channel dimensions => 1D vector of batch scores
    return jnp.mean( jnp.mean(ssim_map, axis=(1, 2, 3)) )

