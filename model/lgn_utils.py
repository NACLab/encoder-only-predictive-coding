from jax import lax, jit
import jax.numpy as jnp
from functools import partial

'''
Contains supporting co-routines for LGN glimpser/processor, as well as 
some useful visual metrics for reporting. 
'''

@jit
def measure_NCC(img1, img2):
    # Computes the Normalized Cross-Correlation (NCC) over a batch of images.
    # Input layout: (B, C, H, W)
    # Returns: A 1D vector of length B containing scores between [-1.0, 1.0]
    #          where 1.0 represents perfect structural correlation.

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
    # Computes the Structural Similarity Index (SSIM) over a batch of images.
    # Input layout: (B, C, H, W)
    # Range: [0, max_val]

    ## define standard SSIM stabilizer constants
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

@partial(jit, static_argnums=[1, 2])
def lateral_inhibition_filter(
        saliency_map: jnp.ndarray,
        window_size: int = 3,
        inhibition_strength: float = 0.8
) -> jnp.ndarray:
    # applies lateral competitive inhibition to thin out fat saliency maps : saliency map shape expected: (B, C, H, W)

    ## extract local neighborhood maximums using a max pool
    local_max = lax.reduce_window(
        saliency_map,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, window_size, window_size),
        window_strides=(1, 1, 1, 1),
        padding='SAME'
    )
    ## create an inhibition mask: suppress pixels that fall short of their local peak
    ### if a pixel is much weaker than its neighbor, crush its activity
    is_peak = saliency_map >= (local_max * inhibition_strength)

    ## mask original saliency map
    thinned_saliency = jnp.where(is_peak, saliency_map, 0.0)
    return thinned_saliency

## internal Gaussian kernel (2D only)
def _calc_gaussian_kernel_2D(sigma: float, radius: int) -> jnp.ndarray:
    x = jnp.arange(-radius, radius + 1)
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = kernel / jnp.sum(kernel)
    return kernel[jnp.newaxis, jnp.newaxis, :, :]

@partial(jit, static_argnums=[3, 4, 5, 9, 10])
def lgn_map_filter( ## LGN filter co-routine
        images: jnp.ndarray,  # Input tensor scaled [0, 1] (B, C, H, W)
        sigma_center: float,  # Sensitivity radius (e.g., 0.8)
        sigma_surround: float,  # Inhibition radius (e.g., 2.0)
        kernel_size: int,  # Radius window
        mode='reflect',  # 'reflect', #'edge', # Boundary protection choice
        semi_sat_constant=0.075,  # 0.01, #0.02,      # Scaled precisely for [0, 1] data
        excitation_exp=1.,  # 1.2,          # Mild compression for fractionals
        inhibition_exp=1.,  # 1.0,
        baseline=0.,  # 1e-4, ## baseline neural values for filter
        combine_onoff_codes=True,  ## if True, out = on - off, else, out = [on, off]
        post_lateral_inhibition_strength=0.,
        **kwargs
) -> jnp.ndarray:
    ## in mammalian vision, retina & lateral geniculate nucleus (LGN) do not pass raw light intensities to cortex;
    ## instead, they strip away redundant data & transform sensory info into a sparse, contrast-invariant map of local
    ## spatial variations
    ##
    ## Note on dual on/off stream split:
    ### mammalian retinas split visual inputs into two distinct, parallel pathways:
    ### ON-center cells (which fire when a region is brighter than its background), and
    ### OFF-center cells (which fire when a region is darker than its background)

    ## Edge-artifact padding correction
    padding_config = ((0, 0), (0, 0), (kernel_size, kernel_size), (kernel_size, kernel_size))
    padded_x = jnp.pad(images, padding_config, mode=mode)

    ## Gaussian kernel setup
    k1 = _calc_gaussian_kernel_2D(sigma_center, kernel_size)
    k2 = _calc_gaussian_kernel_2D(sigma_surround, kernel_size)

    dn = lax.ConvDimensionNumbers(
        lhs_spec=(0, 1, 2, 3), rhs_spec=(0, 1, 2, 3), out_spec=(0, 1, 2, 3)
    )
    num_channels = images.shape[1]

    ## spatial blur extraction via independent depthwise XLA convolving
    blur_center = lax.conv_general_dilated(
        padded_x, k1, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=num_channels
    )
    blur_surround = lax.conv_general_dilated(
        padded_x, k2, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=num_channels
    )

    ## generate ON stream Pathway (light updates on dark background)
    rectified_center_on = jnp.maximum(0.0, blur_center)
    rectified_surround_on = jnp.maximum(0.0, blur_surround)
    num_on = jnp.power(rectified_center_on, excitation_exp)
    den_on = jnp.power(rectified_surround_on, inhibition_exp) + (semi_sat_constant ** 2)
    on_stream = num_on / den_on
    ## generate OFF stream Pathway (dark updates on light background)
    rectified_center_off = jnp.maximum(0.0, blur_surround)
    rectified_surround_off = jnp.maximum(0.0, blur_center)
    num_off = jnp.power(rectified_center_off, excitation_exp)
    den_off = jnp.power(rectified_surround_off, inhibition_exp) + (semi_sat_constant ** 2)
    off_stream = num_off / den_off

    ## inject a tiny bit background bias to keep input filter neurons "active"
    on_stream = on_stream + baseline
    off_stream = off_stream + baseline
    if combine_onoff_codes:
        # re-combine on/off streams into one contrast-invariant signed map
        effective_response = on_stream - off_stream
    else:
        ## concatenate to a dual structural layout; shape output: (B, C * 2, H, W)
        effective_response = jnp.concatenate([on_stream, off_stream], axis=1)

    if post_lateral_inhibition_strength > 0.:
        ## apply optional post-filter lateral inhibition (to thin out stimuli values)
        effective_response = lateral_inhibition_filter(
            effective_response,
            window_size=3,  ## 3x3 works for thinning lines
            inhibition_strength=post_lateral_inhibition_strength  # 0.85 ## higher value = thinner, more selective lines
        )
    return effective_response

