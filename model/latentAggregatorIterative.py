import jax
from jax import numpy as jnp, random, nn, jit, lax
from functools import partial #as bind
from ngclearn.utils.io_utils import makedir
import jax
import jax.numpy as jnp
from functools import partial


def init_aggregator_state(batch_size, feature_dim, grid_dims):
    ## initializes accumulator state dictionary
    return {
        "k": 0.0,
        "sum_features": jnp.zeros((batch_size, feature_dim)),
        "sum_sq_features": jnp.zeros((batch_size, feature_dim)),
        "sum_weights": jnp.zeros((batch_size, feature_dim)),
        "running_centroids": jnp.zeros((batch_size, feature_dim, grid_dims))
    }


@partial(jax.jit, static_argnums=[3])
def update_global_grid_cell_code_step(
        z_state,
        glimpse_feature,
        coordinate,
        num_frequencies=6
):
    k_prev = z_state["k"]
    k_curr = k_prev + 1.0

    sum_f = z_state["sum_features"] + glimpse_feature
    sum_sq_f = z_state["sum_sq_features"] + (glimpse_feature ** 2)

    powers = jnp.arange(num_frequencies, dtype=jnp.float32)
    frequencies = (2.0 ** powers) * jnp.pi

    phase_angles = coordinate[..., None] * frequencies
    sin_phases = jnp.sin(phase_angles)
    cos_phases = jnp.cos(phase_angles)

    grid_dims = 2 * num_frequencies * 2
    grid_coordinate_step = jnp.concatenate([sin_phases, cos_phases], axis=-1).reshape(
        (coordinate.shape[0], grid_dims)
    )

    feature_weight_step = jnp.abs(glimpse_feature)
    sum_w = z_state["sum_weights"] + feature_weight_step

    ## conduct outer-product contraction (emulate einsum logic)
    centroid_update = feature_weight_step[..., None] * grid_coordinate_step[:, None, :]
    running_centroids = z_state["running_centroids"] + centroid_update

    updated_state = {
        "k": k_curr,
        "sum_features": sum_f,
        "sum_sq_features": sum_sq_f,
        "sum_weights": sum_w,
        "running_centroids": running_centroids
    }

    mean_code = sum_f / k_curr
    var_code = jnp.maximum((sum_sq_f / k_curr) - (mean_code ** 2), 0.0)
    std_code = jnp.sqrt(var_code)

    ## normalize centroids using accumulated step-weights
    grid_centroids = running_centroids / (sum_w[..., None] + 1e-8)

    ## reshape to (batch-size, D * grid_dims) to mirror batch-einsum shape
    grid_centroids_flat = grid_centroids.reshape((glimpse_feature.shape[0], -1))

    mean_code = mean_code / (jnp.linalg.norm(mean_code, axis=-1, keepdims=True) + 1e-8)
    std_code = std_code / (jnp.linalg.norm(std_code, axis=-1, keepdims=True) + 1e-8)
    grid_centroids_flat = grid_centroids_flat / (jnp.linalg.norm(grid_centroids_flat, axis=-1, keepdims=True) + 1e-8)

    global_code = jnp.concatenate([mean_code, std_code, grid_centroids_flat], axis=-1)

    return updated_state, global_code


class IterativeLatentAggregator(): ## iterative sketch-pad memory for MPC

    def __init__(
            self,
            dkey,
            latent_feature_dim=1,
            batch_size=1,
            **kwargs
    ):
        self.dkey, *subkeys = jax.random.split(dkey, 5)
        self.batch_size = batch_size
        self.latent_feature_dim = latent_feature_dim
        self.num_frequencies = 6 ## affects grid-cell properties
        self.grid_dims = 2 * self.num_frequencies * 2

        self._global_state = init_aggregator_state(
            self.batch_size,
            self.latent_feature_dim,
            grid_dims=self.grid_dims
        )
        ## match dimensions of concatenated output layout (pin as data members)
        lat_dim = self.latent_feature_dim  # mean_code
        lat_dim += self.latent_feature_dim  # std_code
        lat_dim += (self.latent_feature_dim * self.grid_dims)  ## grid_centroids_flat
        ## global code "sketch-pad"
        self.global_latent_codes = jnp.zeros((batch_size, lat_dim))

    def update(
            self,
            glimpse_code,
            coordinates_code
    ):
        ## update current memory "etch" of global representational code
        updated_state, global_code = update_global_grid_cell_code_step(
            self._global_state,
            glimpse_code,
            coordinates_code,
            num_frequencies=self.num_frequencies
        )
        self._global_state = updated_state
        self.global_latent_codes = global_code

    def compute_code(self): ## retrieves currently pinned sketch-pad global code
        return self.global_latent_codes

    def reset(self): ## clears sketch-pad global code
        self._global_state = init_aggregator_state(
            self.batch_size,
            self.latent_feature_dim,
            grid_dims=self.grid_dims
        )
        self.global_latent_codes = self.global_latent_codes * 0.0

