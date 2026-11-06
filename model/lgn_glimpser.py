#import numpy as np
import jax
from jax import numpy as jnp, random, nn, jit, lax
from functools import partial as bind
from ngclearn.utils.viz.synapse_plot import visualize
from ngclearn.utils.filters import gaussian_filter, cortical_gaussian_filter as cortical_filter
from .lgn_utils import lgn_map_filter

############# LGN patch-processing co-routines #############

def convert_to_coord(yx_loc, px, py): ## convert [-1,1] coords to [0, px] pixel coord-space
    #yx_raw = jnp.array(jnp.squeeze(jnp.astype(0.5 * ((yx_loc + 1.0) * px), jnp.int32)))
    yx_raw = ((yx_loc + 1.0) / 2.0) * (px - 1)
    return yx_raw

@bind(jax.vmap, in_axes=[0, 0, None], out_axes=0)
def _vmap_slice(img: jax.Array, start_yx: jax.Array, sz: int) -> jax.Array:
    return jax.lax.dynamic_slice(
        img,(start_yx[0], start_yx[1], 0),  # Start indices per image
        (sz, sz, img.shape[-1]) # Size of the slice
    )

@bind(jax.jit, static_argnums=[2,])
def _get_patch(images: jax.Array, locs: jax.Array, sz: int) -> jax.Array:
    B, H, W, C = images.shape
    start_idx = jnp.astype(0.5 * ((locs + 1.0) * H), jnp.int32)
    # pad with zeros to avoid exceeding image boundaries
    padding = (
        (0, 0), # B
        (sz // 2, sz // 2), # H
        (sz // 2, sz // 2), # W
        (0, 0) # C
    )
    images_padded = jnp.pad(images, padding)
    samples = _vmap_slice(images_padded, start_idx, sz)
    return samples

@bind(jax.jit, static_argnums=[1, 2])
def _transform_patch(patch, stat_type=0, normalize=False):
    if stat_type == 1:  ## center patch
        patch = patch - jnp.mean(patch, axis=[1, 2], keepdims=True)
    elif stat_type == 2:  ## standardize patch
        patch = ((patch - jnp.mean(patch, axis=[1, 2], keepdims=True)) /
                   (jnp.std(patch, axis=[1, 2], keepdims=True) + 1e-6))
    if normalize:
        patch_norms = jnp.linalg.norm(patch, ord=2, axis=(1, 2), keepdims=True)
        mask = patch_norms > 0.
        patch_norms = patch_norms * mask + (1. - mask) ## prevent div-by-0
        patch = patch / patch_norms
    return patch

@bind(jax.jit, static_argnums=[1, 4, 5])
def _extract_masked_patch_list(
        images: jax.Array,
        sample_dim: int,
        locs: jax.Array,
        loc_offsets: jax.Array,
        stat_type: int=0,
        normalize = False
): # -> list(jax.Array):
    #fovea_scale = 2.
    #sz = int(fovea_scale * sample_dim) ## parafoveal scale
    #sz = int(fovea_scale * fovea_scale * sample_dim) ## peripheral scale

    patches = []
    n_fovea = loc_offsets.shape[0]
    #n_masks = mask_indices.shape[0] ## TODO: do check that n_msks == n_fovea
    ## extract foveal patches
    for i in range(n_fovea):
        offset_i = loc_offsets[i:i+1, :]
        patch_i = _get_patch(images, locs + offset_i, sample_dim)
        if stat_type > 0:
            patch_i = _transform_patch(
                patch_i, stat_type=stat_type, normalize=normalize
            )
        patches.append(patch_i)  # (B, S, S, 1)
    return patches

@bind(jax.jit, static_argnums=[1, 4, 5, 6])
def extract_patch_set(
    images: jax.Array,
    patch_dim: int, ## patch dim (assumes square patches),
    locs: jax.Array,
    offsets: jax.Array,
    stat_type: int=0,  ## 0: nothing, 1: mean-centering, 2: standardization
    rescale_dim: int = -1,
    normalize=False
) -> (jax.Array, jax.Array):
    B, _, _, C = images.shape
    _rescale_dim = rescale_dim
    if _rescale_dim <= 0:
        _rescale_dim = patch_dim
    ## get patch + mask array lists
    patches = _extract_masked_patch_list(images, patch_dim, locs, offsets)
    # resize the patches to sample_dim
    for i in range(len(patches)):  ## only reshape patches idx >= 1
        # k = patches[i].shape[1] // foveal_dim
        patches[i] = jax.image.resize(
            patches[i], (B, _rescale_dim, _rescale_dim, C), method="bilinear"
        )  ## re-size
        patches[i] = _transform_patch(
            patches[i], stat_type=stat_type, normalize=normalize
        )  ## transform

    # concatenate, flatten and return
    raw_patches = jnp.stack(patches, axis=1)  # (B, P, S, S, C)
    return (
        raw_patches.reshape((raw_patches.shape[0], -1)),
        raw_patches  # (B, P*S*S*C)
    )

@jit
def rescale(x, a=-1., b=1.):
    xmin = jnp.min(x, axis=(1, 2, 3), keepdims=True)
    xmax = jnp.max(x, axis=(1, 2, 3), keepdims=True)
    _x = ((x - xmin) * (b - a))/(xmax - xmin) + a
    return _x

############# epistemic-driven saccade selection routines #############

def _get_bottom_up_salience(
    subtracted_lgn_image: jnp.ndarray, sigma=8.0
) -> jnp.ndarray:
    """Computes baseline bottom-up structural salience from the LGN filter."""
    energy = jnp.sum(jnp.abs(subtracted_lgn_image), axis=0)
    radius = int(sigma * 3)
    if radius % 2 == 0: radius += 1
    x = jnp.arange(-radius // 2 + 1, radius // 2 + 1)
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    kernel = (kernel / jnp.sum(kernel))[None, None, :, :]

    padded = jnp.pad(energy, ((radius//2, radius//2), (radius//2, radius//2)), mode='edge')
    dn = lax.ConvDimensionNumbers((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
    return jnp.squeeze(lax.conv_general_dilated(padded[None, None, :, :], kernel, (1, 1), 'VALID', dimension_numbers=dn))

@bind(jax.jit, static_argnums=[11])
def _plan_next_saccade(
    subtracted_lgn_image: jnp.ndarray,
    history_coords: jnp.ndarray,      ## (max_saccades, 2)
    history_errors: jnp.ndarray,      ## (max_saccades,)
    history_mask: jnp.ndarray,        ## (max_saccades,)
    rng_key: jnp.ndarray,             ## Required for JAX stochastic sampling
    temperature: float = 0.5,         ## Controls stochasticity (0.1 = close to argmax)
    ior_sigma: float = 8.0,           ## TUNED: Increased to widen inhibition area
    error_spread_sigma: float = 12.0,  ## TUNED: Increased to spread error footprint smoothly
    top_down_weight: float = 0.4,     ## TUNED: Shifted slightly for optimal mix
    coord_min: float = -0.95,
    coord_max: float = 0.95,
    gen_new_coords: bool = True       ## Keep as static if passed statically or via closure
) -> jnp.ndarray:
    """
    Active inference saccade planner with optional stochastic foraging (via Boltzmann/Gumbel-Max selection).
    Note that this blends bottom-up salience with top-down free energy error accumulation.

    Args:
        subtracted_lgn_image:
        history_coords:
        history_errors:
        history_mask:
        rng_key:
        temperature:
        ior_sigma:
        error_spread_sigma:
        top_down_weight:
        coord_min:
        coord_max:
        gen_new_coords:

    Returns:
        next glimpse coordinate/action
    """
    H, W = subtracted_lgn_image.shape[-2:]

    ## Coordinate Grid Setup
    y_grid = jnp.linspace(0, H - 1, H)
    x_grid = jnp.linspace(0, W - 1, W)
    yy, xx = jnp.meshgrid(y_grid, x_grid, indexing='ij')

    ## Extract Bottom-Up Structural Landscape
    bu_map = _get_bottom_up_salience(subtracted_lgn_image)

    ## Dynamic Accumulation via lax.scan
    init_td_map = jnp.zeros((H, W))
    init_ior_mask = jnp.ones((H, W))

    def accumulate_history(carry, input_tuple):
        td_map, ior_mask = carry
        coord, error, mask_val = input_tuple

        norm_y = coord[0]
        norm_x = coord[1]

        ## Convert normalized [-1, 1] coordinates back to absolute pixel coordinates
        y_pixel = ((norm_y + 1.0) / 2.0) * (H - 1)
        x_pixel = ((norm_x + 1.0) / 2.0) * (W - 1)

        dist_sq = (yy - y_pixel)**2 + (xx - x_pixel)**2

        error_blob = error * jnp.exp(-0.5 * dist_sq / error_spread_sigma**2)
        current_ior = 1.0 - jnp.exp(-0.5 * dist_sq / ior_sigma**2)

        new_td_map = td_map + (error_blob * mask_val)
        new_ior_mask = ior_mask * jnp.where(mask_val > 0.0, current_ior, 1.0)

        return (new_td_map, new_ior_mask), None

    (td_map, ior_mask), _ = lax.scan(
        accumulate_history,
        (init_td_map, init_ior_mask),
        (history_coords, history_errors, history_mask)
    )

    ## Integrate Pathways: Total Epistemic Value
    combined_salience = (1.0 - top_down_weight) * bu_map + top_down_weight * td_map
    final_score_map = combined_salience * ior_mask

    ## --- JAX Native Spatial Boundary Constraint ---
    norm_yy = (yy / (H - 1)) * 2.0 - 1.0
    norm_xx = (xx / (W - 1)) * 2.0 - 1.0

    within_bounds = (
        (norm_yy >= coord_min) & (norm_yy <= coord_max) &
        (norm_xx >= coord_min) & (norm_xx <= coord_max)
    )

    # We use -1e5 here because adding log-Gumbel noise needs a safe boundary
    # that doesn't trigger NaN overflows but remains an unselectable floor.
    bounded_score_map = jnp.where(within_bounds, final_score_map, -1e5)

    if gen_new_coords:
        noise_m = (temperature > 0.)
        temp_m = noise_m * temperature + (1. - noise_m)
        # --- THE GUMBEL-MAX TRICK FOR STOCHASTIC FORAGING ---
        # Flatten for sampling logic
        flat_scores = bounded_score_map.flatten()

        # Draw uniform noise and transform to standard Gumbel noise
        uniform_noise = jax.random.uniform(rng_key, shape=flat_scores.shape, minval=1e-7, maxval=1.0)
        gumbel_noise = -jnp.log(-jnp.log(uniform_noise))

        # Apply temperature-scaled stochastic sampling.
        # Out-of-bounds metrics masked with -1e5 remain heavily suppressed.
        stochastic_logits = (flat_scores / temp_m) + (gumbel_noise * noise_m)
        stochastic_logits = jnp.where(within_bounds.flatten(), stochastic_logits, -1e10)

        # Select peak index from the perturbed space
        flat_idx = jnp.argmax(stochastic_logits)
        next_y_idx = flat_idx // W
        next_x_idx = flat_idx % W

        ## Normalize selected pixel back to [-1, 1] framework
        norm_y = (next_y_idx / (H - 1)) * 2.0 - 1.0
        norm_x = (next_x_idx / (W - 1)) * 2.0 - 1.0
    else:
        norm_y = norm_x = 0.

    return jnp.array([norm_y, norm_x]), bu_map, final_score_map

# Updated vmap signature mapping uniquely over batch positions.
# Notice the '0' added for the 5th argument (rng_key). Hyperparams remain None (broadcasted).
plan_next_saccade = jax.vmap(
    _plan_next_saccade,
    in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None)
)

def get_sigm_temperature(current_step, total_glimpses): ## sigmoidal temperature
    ## adaptive 2-phase sigmoid temperature schedule
    progress = current_step / (total_glimpses - 1)
    tau_max, tau_min = 0.012, 0.0012 ##1.2, 0.1
    midpoint, steepness = 0.35, 10. # 0.35, 10.0
    temp = float(tau_min + (tau_max - tau_min) / (1.0 + jnp.exp(steepness * (progress - midpoint))))
    return temp

def get_exp_temperature(current_step, total_glimpses): ## exponential decay temperature
    # current_step runs from 0 to T-1
    progress = current_step / (total_glimpses - 1)
    tau_max, tau_min = 0.012, 0.0012 ## 1.2, 0.1
    steepness = 3.5
    return tau_min + (tau_max - tau_min) * jnp.exp(-steepness * progress)

def get_temperature(current_step, total_glimpses, schedule="none"): #"sigmoid"): #"exp"):
    if "sigmoid" in schedule:
        return get_sigm_temperature(current_step, total_glimpses)
    elif "exp" in schedule: ## exp schedule
        return get_exp_temperature(current_step, total_glimpses)
    elif "constant" in schedule:
        return 0.05
    return 0.

## LGN => Lateral Geniculate Nucleus core processor
class LGN_Glimpser():
    """

    Args:
        dkey:
        image_shape:
        batch_size:
        foveal_shape:
        parafoveal_shape:
        peripheral_shape:
        glimpse_bounds:
        dxy:
        center_patches:
        patch_stat_type:
        filter_type:
            :Note: supported filters include - "difference_of_gaussians", "ratio_of_gaussians"

    """

    def __init__(
            self,
            dkey,
            data_fname,
            image_shape,
            batch_size,
            foveal_shape,
            parafoveal_shape,
            peripheral_shape,
            glimpse_bounds=0.35,
            dxy=0.15, ## glimpse/glance differential
            center_patches=False,
            patch_stat_type=0,
            input_filter:str=None, ## apply diff-of-gauss filter to sampled images?
            input_scale:float=1., ## post-image scale factor to apply before extracting patches
            use_fine_grained_filter=True,
            max_saccades=10,
            **kwargs
    ):
        self.dkey, *subkeys = random.split(dkey, 5)
        self.image_shape = image_shape
        self.foveal_shape = foveal_shape
        self.parafoveal_shape = parafoveal_shape
        self.peripheral_shape = peripheral_shape
        self.batch_size = batch_size
        self.glimpse_bounds = glimpse_bounds #0.25
        self.dxy = dxy #0.05
        self.foveal_shift = 0.25 #0.3
        self.center_patches = center_patches
        self.max_saccades = max_saccades

        self.glimpse_setup = 1 ## 0 or 1

        self.filter_inhibition = 0.
        self.input_scale = input_scale
        self.apply_filter = False
        self.use_ratio_filter = False
        self.use_fine_grained_filter = use_fine_grained_filter
        self.filter_fx = gaussian_filter
        self.input_filter = input_filter
        if input_filter == "difference_of_gaussians" or input_filter == "DoG":
            print(">> Applying DoG filter!")
            self.apply_filter = True
        elif input_filter == "ratio_of_gaussians" or input_filter == "RoG":
            print(">> Applying RoG filter!")
            self.apply_filter = self.use_ratio_filter = True
        elif input_filter == "cortical_DoG":
            print(">> Applying Cortical-DoG filter")
            self.apply_filter = True
            self.filter_fx = cortical_filter
        elif input_filter == "cortical_RoG":
            print(">> Applying Cortical-RoG filter")
            self.apply_filter = self.use_ratio_filter = True
            self.filter_fx = cortical_filter
        elif "lgn" in input_filter: ## most cortically-realistic
            print(">> Applying LGN filter")
            self.apply_filter = self.use_ratio_filter = True
            self.filter_fx = lgn_map_filter
            self.filter_inhibition = 0. #0.00005 #0.00015 # 0. # 0.005 ## 0.85

        self.normalize_patches = False
        self.stat_type = patch_stat_type ## 0, 1, 2
        if center_patches:
            self.stat_type = 1 ## this auto-sets to 1
        if patch_stat_type == 3: ## normalize patches
            self.stat_type = 0 ## this auto-sets to 0
            self.normalize_patches = True #False  ## normalize patches by Frobenius norm?

        self.foveal_offsets = self.parafoveal_offsets = self.peripheral_offsets = None
        if self.glimpse_setup == 0:
            self.foveal_offsets = jnp.array(  ## overlapped ring structures
                [
                    ## topology
                    [-1., -1.],
                    [-1.,  1.],
                    [ 1., -1.],
                    [ 1.,  1.]
                ],
                dtype=jnp.float32
            ) * self.foveal_shift
            self.parafoveal_offsets = self.foveal_offsets
            self.peripheral_offsets = None
            nfovea = npfovea = self.foveal_offsets.shape[0]
            nperi = 0
        else:
            self.foveal_offsets = jnp.array(  ## overlapped ring structures
                [
                    ## topology
                    [-1., -1.],
                    [-1.,  1.],
                    #[ 0.,  0.],
                    [ 1., -1.],
                    [ 1.,  1.]
                ],
                dtype=jnp.float32
            ) * self.foveal_shift  # - 0.01
            nfovea = self.foveal_offsets.shape[0]
            npfovea = nperi = 0
            if self.parafoveal_shape is not None:
                self.parafoveal_offsets = jnp.array(
                    [[0., 0.]]
                )
                npfovea = self.parafoveal_offsets.shape[0]
            if self.peripheral_shape is not None:
                self.peripheral_offsets = jnp.array(
                    [[0., 0.]]
                )
                nperi = self.peripheral_offsets.shape[0]

        self.n_streams = nfovea + npfovea + nperi

        ## load in images
        self.images = jnp.load(data_fname)
        self.dataset_name = "norb"
        if "mnist" in data_fname:
            self.dataset_name = "mnist"


        n_rows = self.images.shape[0]
        remainder = n_rows % self.batch_size
        if remainder != 0:
            n_rows, n_cols = self.images.shape
            remainder = n_rows % batch_size
            pad_rows = batch_size - remainder ## calculate how many extra rows are needed
            ## use jax.lax.select to handle the conditional logic inside JAX transformations
            ### if remainder is 0, we need 0 pad rows instead of a full batch_size
            pad_rows = jax.lax.select(remainder == 0, 0, pad_rows)
            ## generate random indices from the valid row range
            ## replace=True behavior is native to jax.random.choice
            random_indices = jax.random.choice(subkeys[1], n_rows, shape=(pad_rows,), replace=True)
            self.images = jnp.concatenate([self.images, self.images[random_indices]], axis=0)
            print(f">> Dataset < (batch_size = {self.batch_size}); Expanded dataset to shape: ", self.images.shape)

        self.ptrs = None
        self._batch_ptrs = None
        self._image_batch = None
        self._patches = None
        self.g_step = 0
        self._xy_batch = None

        ## for epistemic saccade process
        self.temperature = 0.5 ## controls stochasticity (0.1 = close to argmax)
        ## purely deterministic saccade process settings
        # self.ior_sigma = 4.0
        # self.error_spread_sigma = 6.0
        # self.top_down_weight = 0.5
        self.ior_sigma = 5.0 #6.0
        self.error_spread_sigma = 7.5 #10.0 #5.0
        self.top_down_weight = 0.5 #0.3 #0.4

        ## foraging book-keeping variables
        self.history_coords = jnp.zeros((self.batch_size, self.max_saccades, 2)) ## (B, max_saccades, 2)
        self.history_errors = jnp.zeros((self.batch_size, self.max_saccades)) ## (B, max_saccades)
        self.history_mask = jnp.zeros((self.batch_size, self.max_saccades)) ## (B, max_saccades)
        self._k = 0 ## internal tracker of saccade step
        self.bu_salience_map = None
        self.score_map = None

    def _sample_images(self, batch_ptrs):
        batch_size = batch_ptrs.shape[0]
        img_batch = self.images[batch_ptrs, :]
        px, py = self.image_shape
        self._xy_batch = jnp.zeros((batch_size, 2))
        self._image_batch = jnp.reshape(img_batch, (img_batch.shape[0], 1, px, py))

        if self.apply_filter:
            self._image_batch = jnp.reshape(img_batch, (img_batch.shape[0], 1, px, py))
            if "lgn" in self.input_filter:
                ## natural-image (e.g., NORB) saliency config
                lgn_config = {
                    "sigma_center": 0.8,
                    "sigma_surround":2.0,
                    "kernel_size": 6,
                    "semi_sat_constant": 0.075,
                    "excitation_exp": 1.0,
                    "inhibition_exp": 1.0,
                    "mode": "reflect"
                }
                if "mnist" in self.dataset_name:
                    ## un-natural image (e.g., MNIST) saliency config
                    lgn_config = {
                        "sigma_center": 0.35,  ## sharpen the center; target pixel-level edges
                        "sigma_surround": 0.75,  ## pull in the inhibition window tightly
                        "kernel_size": 3,  #
                        "semi_sat_constant": 0.25,  ## raise to suppress background noise amplification
                        "excitation_exp": 1.2,  ## sparsification - boost strong edges, crush weak ones
                        "inhibition_exp": 1.0,
                        "mode": "constant"  ## zero-pad since un-natural image backgrounds are generally zero
                    }
                s1 = lgn_config["sigma_center"]
                s2 = lgn_config["sigma_surround"]
                kw = lgn_config["kernel_size"]
                excitation_exp = lgn_config["excitation_exp"]
                inhibition_exp = lgn_config["inhibition_exp"]
                mode = lgn_config["mode"]
            else:
                kw = 4
                if self.use_fine_grained_filter:
                    s1, s2 = 1.0, 1.6 ## finer-grained filter
                else:
                    s1, s2 = 1.0, 5.0 ## coarse-grained filter
            ## run actual filter over image batch
            self._image_batch = self.filter_fx( #gaussian_filter(
                self._image_batch,
                sigma_center=s1,
                sigma_surround=s2,
                kernel_size=kw, ## Kernel radius (kernel window size is 2*r + 1)
                use_ratio=self.use_ratio_filter,
                excitation_exp=excitation_exp,
                inhibition_exp=inhibition_exp,
                baseline=0.,
                mode=mode, 
                post_lateral_inhibition_strength=self.filter_inhibition
            )
            B, C, H, W = self._image_batch.shape ## pull out filtered batch shape
            self._image_batch = jnp.transpose(self._image_batch, (0, 2, 3, 1))
            #self._image_batch = jnp.expand_dims(jnp.squeeze(self._image_batch, axis=1), axis=3)
        else:
            self._image_batch = jnp.reshape(img_batch, (img_batch.shape[0], px, py, 1))
        self._image_batch = self._image_batch * self.input_scale # xsc

    def step_epistemic_saccade(self, temperature_schedule="none"):
        ## moves "LGN" glimpser in accordance w/ epistemic foraging
        img_batch = self._image_batch ## get current LGN image

        temperature = get_temperature(
            self._k, self.max_saccades, schedule=temperature_schedule
        )
        #print(f"{self._k}: T = {temperature}")
        self.dkey, batch_key = jax.random.split(self.dkey)
        ## split batch_key into subkeys to match batch size (e.g., batch_size=200)
        subkeys = jax.random.split(batch_key, self.batch_size)
        next_xy_coords, bu_map, score_map = plan_next_saccade(
            jnp.transpose(img_batch, (0, 3, 1, 2)),
            self.history_coords,
            self.history_errors,
            self.history_mask,
            subkeys,
            temperature, #self.temperature,
            self.ior_sigma,
            self.error_spread_sigma,
            self.top_down_weight,
            -self.glimpse_bounds,
            self.glimpse_bounds,
            True
        )
        #print("Coords: ",next_xy_coords[0])
        self.bu_salience_map = bu_map
        self.score_map = score_map
        actions = self._xy_batch = next_xy_coords
        self.history_coords = self.history_coords.at[:, self._k].set(actions) ## store coordinates at step _k
        self.history_mask = self.history_mask.at[:, self._k].set(1.) ## update history of masks

        flat_patches, *_ = self.step(self._xy_batch)
        return flat_patches, actions


    def step_random_saccade(self, track_stats=False):
        ## moves "LGN" glimpser randomly (in accordance w/ a random saccade)
        if track_stats: ## NOTE: turning this on will slow down random saccade a tiny bit...
            self.dkey, batch_key = jax.random.split(self.dkey)
            ## split batch_key into subkeys to match batch size (e.g., batch_size=200)
            subkeys = jax.random.split(batch_key, self.batch_size)            
            _, bu_map, score_map = plan_next_saccade(
                jnp.transpose(self._image_batch, (0, 3, 1, 2)),
                self.history_coords,
                self.history_errors,
                self.history_mask,
                subkeys,
                self.temperature,
                self.ior_sigma,
                self.error_spread_sigma,
                self.top_down_weight,
                -self.glimpse_bounds,
                self.glimpse_bounds, 
                False
            )
            self.bu_salience_map = bu_map
            self.score_map = score_map

        ## do a step of a random walk
        self.dkey, *subkeys = random.split(self.dkey, 5)
        if self.dxy > 0.:
            eps = random.uniform(
                subkeys[0], (self.batch_size, 2), minval=-1., maxval=1.
            )
            self._xy_batch = jnp.clip(self._xy_batch + eps * self.dxy, -self.glimpse_bounds, self.glimpse_bounds)
        else:
            self._xy_batch = random.uniform(
                subkeys[0], (self.batch_size, 2), minval=-1., maxval=1.
            ) * self.glimpse_bounds

        actions = self._xy_batch
        self.history_coords = self.history_coords.at[:, self._k].set(actions) ## store coordinates at step _k
        self.history_mask = self.history_mask.at[:, self._k].set(1.) ## update history of masks
        #self._k = self._k + 1

        flat_patches, *_ = self.step(self._xy_batch)
        return flat_patches, actions

    def step_saccade(self, policy="random", track_stats=False): ## general movement routine for LGN glimpser
        if policy == "deterministic": ## engage in deterministic foraging
            return self.step_epistemic_saccade() ## default is "none" temperature schedule -> yielding deterministic
        elif policy == "stochastic": ## engage in stochastic foraging
            return self.step_epistemic_saccade(temperature_schedule="sigmoid")
        ## else, default is random saccade policy
        return self.step_random_saccade(track_stats=track_stats)

    def update_glimpser(self, energy): ## call this externally to step glimpse counter by 1 in discrete movement time
        self.history_errors = self.history_errors.at[:, self._k].set(jnp.squeeze(energy, axis=1))
        self._k = self._k + 1 ## advance saccade counter

    def step(self, action, concat_as_arrays=False): ## obtains processed visual input given current saccade action/coordinate
        #yx_loc = jnp.zeros((xs_img.shape[0], 2))
        ## get current images to examine
        img_batch = self._image_batch

        patches = []
        flat_patches = []
        self._xy_batch = action #+ 0
        ## pull out relevant dynamic patches from x-y location
        fovea_flat, fovea = extract_patch_set( ## foveal patches
            img_batch,
            patch_dim=self.foveal_shape[0],
            locs=action, ## yx_loc
            offsets=self.foveal_offsets,
            stat_type=self.stat_type,
            rescale_dim=self.foveal_shape[0],
            normalize=self.normalize_patches
        )
        flat_patches.append(fovea_flat)
        patches.append(fovea)
        if self.parafoveal_offsets is not None:
            parafovea_flat, parafovea = extract_patch_set( ## parafoveal patches
                img_batch,
                patch_dim=self.parafoveal_shape[0],
                locs=action, ## yx_loc
                offsets=self.parafoveal_offsets,
                stat_type=self.stat_type,
                rescale_dim=self.foveal_shape[0],
                normalize=self.normalize_patches
            )
            flat_patches.append(parafovea_flat)
            patches.append(parafovea)
        if self.peripheral_offsets is not None:
            periphera_flat, periphera = extract_patch_set( ## peripheral patches
                img_batch,
                patch_dim=self.peripheral_shape[0],
                locs=action, ## yx_loc
                offsets=self.peripheral_offsets,
                stat_type=self.stat_type,
                rescale_dim=self.foveal_shape[0],
                normalize=self.normalize_patches
            )
            flat_patches.append(periphera_flat)
            patches.append(periphera)
        out_patches = patches
        if concat_as_arrays:
            ## combine patch types into one "view" ("glance")
            #flat_patches = jnp.concat([fovea_flat, parafovea_flat, periphera_flat], axis=1)
            flat_patches = jnp.concat(flat_patches, axis=1)
            patches = jnp.concat(patches, axis=1)
            #patches = jnp.concat([fovea, parafovea, periphera], axis=1)
            out_patches = flat_patches
        else:
            for p in range(len(out_patches)):
                patch_p = out_patches[p]
                if patch_p.shape == 4:
                    patch_p = jnp.expand_dims(patch_p, axis=1)
                out_patches[p] = patch_p
            out_patches = jnp.concatenate(out_patches, axis=1)  ## yields 5D tensor
            # for p in range(len(out_patches)):
            #     print(out_patches[p].shape)
            #     B, H, W, C = out_patches[p].shape
            #     out_patches[p] = jnp.reshape(out_patches[p], (B, C, H, W))

        self._patches = patches
        self.g_step += 1 ## move glimpse counter

        return out_patches, 0., False, {}  ## format: obs(t), reward, done, {information}

    def render(self, output_dir, preserve_img_id=True, sample_idx=-1): ## NOTE: expensive function if batch_size > 1
        px, py = self.foveal_shape
        batch_ptrs = self._batch_ptrs
        if sample_idx >= 0:
            batch_ptrs = batch_ptrs[sample_idx:sample_idx+1]
        for j in range(batch_ptrs.shape[0]): ## loop through each patch w/in current patch batch
            ptr_j = int(batch_ptrs[j]) ## get image id
            pats = []
            for i in range(self.n_streams): ## loop through every stream patch w/in patch sample
                stream_i = self._patches[j, i]
                if len(stream_i.shape) > 2:
                    for k in range(stream_i.shape[2]):
                        pat_k = stream_i[:, :, k]
                        pat_k = jnp.reshape(pat_k, (1, px * py))
                        pats.append(pat_k)
                else:
                    stream_i = jnp.reshape(stream_i, (1, px * py))
                    pats.append(stream_i)
            pats = jnp.concat(pats, axis=0).T
            #print(jnp.linalg.norm(pats))
            out_fname = f"{output_dir}view{self.g_step}"
            if preserve_img_id:
                out_fname = f"{output_dir}img{ptr_j}_view{self.g_step}"
            visualize(
                [pats], [(px, py)], out_fname
            )

    def reset(self, batch_pointers=None):
        if batch_pointers is not None:
            self._batch_ptrs = batch_pointers
        else:
            self.dkey, *subkeys = random.split(self.dkey, 5)
            self._batch_ptrs = random.randint(subkeys[0], (self.batch_size,), minval=0, maxval=self.images.shape[0])

        batch_size = self._batch_ptrs.shape[0]
        self._sample_images(self._batch_ptrs)
        self._xy_batch = jnp.zeros((batch_size, 2))
        self.g_step = 0

        ## reset epistemic saccade process
        self.history_coords = jnp.zeros((self.batch_size, self.max_saccades, 2)) ## (B, max_saccades, 2)
        self.history_errors = jnp.zeros((self.batch_size, self.max_saccades)) ## (B, max_saccades)
        self.history_mask = jnp.zeros((self.batch_size, self.max_saccades)) ## (B, max_saccades)
        self._k = 0 ## internal tracker of saccade step
        self.bu_salience_map = None
        self.score_map = None
