from jax import numpy as jnp, random, jit, lax
from functools import partial
import ngclearn
from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.jaxComponent import JaxComponent


def _calc_gaussian_kernel_2D(sigma: float, radius: int) -> jnp.ndarray: ## generates a 2D Gaussian convolutional filter kernel
    x = jnp.arange(-radius, radius + 1)
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = kernel / jnp.sum(kernel)
    return kernel[jnp.newaxis, jnp.newaxis, :, :]


class LGNUnitCell(JaxComponent):  ## approximate LGN cell-units
    """
    Biophysical LGN divisive normalization cell component
    """

    def __init__(
            self,
            name,
            n_patches,  ## P: Number of parallel foveal patches
            patch_shape,  ## tuple/list: (C, H, W) for a single patch
            batch_size=1,
            sigma_center=0.8,
            sigma_surround=2.0,
            kernel_size=6,
            mode='reflect',  ## boundary protection choice ('reflect', 'edge')
            semi_sat_constant=0.075,
            excitation_exp=1.0,  ## non-linear response compression
            inhibition_exp=1.0,  ## shunting pool normalization exponent
            combine_onoff_codes=True, ## if True, results in a contrast-map
            baseline=0.0,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        ## bind static and dynamic variables for ngc-learn JAX graph compilation
        self.batch_size = batch_size
        self.n_patches = n_patches
        self.patch_shape = patch_shape  # (C, H, W)
        self.kernel_size = kernel_size
        self.mode = mode
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        self.semi_sat_constant = semi_sat_constant
        self.excitation_exp = excitation_exp
        self.inhibition_exp = inhibition_exp
        self.combine_onoff_codes = combine_onoff_codes
        self.baseline = baseline
        self.center_outputs = True

        ## unpack structural channel dimensions
        C, H, W = patch_shape
        out_channels = C if combine_onoff_codes else C * 2
        self.out_channels = out_channels
        ## calculate matrix mapping dimensionality for downstream (V1) interfacing
        self.flattened_dim = n_patches * out_channels * H * W
        self.n_units = self.flattened_dim

        ## input Compartment: (B, P, C, H, W)
        init_input = jnp.zeros((batch_size, n_patches, C, H, W))
        self.j = Compartment(init_input, display_name="Input Patch Tensor", units="Luminance")
        ## output Compartment A: (B, P, out_C, H, W)
        init_z = jnp.zeros((batch_size, n_patches, out_channels, H, W))
        self.z = Compartment(init_z, display_name="Structured LGN Tensor Output", units="Rate")
        ## output Compartment B: (B, flattened_dim)
        init_z_flat = jnp.zeros((batch_size, self.flattened_dim))
        self.zF = Compartment(init_z_flat, display_name="Flattened LGN Output Matrix", units="Rate")

    @compilable
    def advance_state(self, dt):
        ## executes parallel shunting divisive normalization using the exact
        ### non-linear power expressions from your original co-routine setup.

        images_5d = self.j.get() ## assumes sensory stimuli come in a 5D tensor
        B = images_5d.shape[0]
        P = self.n_patches
        C, H, W = self.patch_shape

        ## flatten patch and batch dimensions into single axis (B * P, C, H, W);
        ## apply explicit boundary protection mode choice
        padded_x = jnp.pad(
            images_5d.reshape(B * P, C, H, W),
            ((0, 0), (0, 0), (self.kernel_size, self.kernel_size), (self.kernel_size, self.kernel_size)),
            mode=self.mode
        )

        ## setup kernels
        k1 = _calc_gaussian_kernel_2D(self.sigma_center, self.kernel_size)
        k2 = _calc_gaussian_kernel_2D(self.sigma_surround, self.kernel_size)
        dn = lax.ConvDimensionNumbers(
            lhs_spec=(0, 1, 2, 3), rhs_spec=(0, 1, 2, 3), out_spec=(0, 1, 2, 3)
        )

        ## apply spatial blur extraction (via independent depthwise XLA convolution)
        blur_center = lax.conv_general_dilated(
            padded_x, k1, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=C
        )
        blur_surround = lax.conv_general_dilated(
            padded_x, k2, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=C
        )

        ## produce ON stream pathway
        rectified_center_on = jnp.maximum(0.0, blur_center)
        rectified_surround_on = jnp.maximum(0.0, blur_surround)
        num_on = jnp.power(rectified_center_on, self.excitation_exp)
        den_on = jnp.power(rectified_surround_on, self.inhibition_exp) + (self.semi_sat_constant ** 2)
        on_stream = num_on / den_on

        ## produce OFF stream pathway
        rectified_center_off = jnp.maximum(0.0, blur_surround)
        rectified_surround_off = jnp.maximum(0.0, blur_center)
        num_off = jnp.power(rectified_center_off, self.excitation_exp)
        den_off = jnp.power(rectified_surround_off, self.inhibition_exp) + (self.semi_sat_constant ** 2)
        off_stream = num_off / den_off
        ## inject background baseline parameters (neural "noise")
        on_stream = on_stream + self.baseline
        off_stream = off_stream + self.baseline

        ## channel combination / routing
        if self.combine_onoff_codes:
            effective_response = on_stream - off_stream ## output is a contrast "map"
        else:
            effective_response = jnp.concatenate([on_stream, off_stream], axis=1) ## output concats on/off channels
        if self.center_outputs: ## apply local centering (ensuring zero-mean output maps)
            effective_response = effective_response - jnp.mean(effective_response, axis=[2, 3], keepdims=True)

        ## unpack results back to model compartments
        y_structured = effective_response.reshape(B, P, self.out_channels, H, W)
        self.z.set(y_structured)
        y_flattened = effective_response.reshape(B, self.flattened_dim)
        self.zF.set(y_flattened)

    @compilable
    def reset(self): ## resets all internal compartments to zero/default values
        C, H, W = self.patch_shape
        init_input = jnp.zeros((self.batch_size, self.n_patches, C, H, W))
        init_y = jnp.zeros((self.batch_size, self.n_patches, self.out_channels, H, W))
        init_y_flat = jnp.zeros((self.batch_size, self.flattened_dim))

        self.j.set(init_input)
        self.z.set(init_y)
        self.zF.set(init_y_flat)
