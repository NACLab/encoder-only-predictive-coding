from jax import numpy as jnp, random, jit
from functools import partial

from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.jaxComponent import JaxComponent


class GridRateCell(JaxComponent):
    """
    Rate-coded grid cell(s) performing Entorhinal-Hippocampal coordinate mapping
    under a multi-scale Fourier basis.
    This component acts as an online global "sketch-pad", continuously aggregating local
    sensory glimpses and spatial coordinates into a unified, phase-locked macro-representation.
    """

    def __init__(
            self,
            name,
            latent_feature_dim,
            num_frequencies=6, ## 6 is a good default
            batch_size=1,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.batch_size = batch_size
        self.latent_feature_dim = latent_feature_dim
        self.num_frequencies = num_frequencies
        self.grid_dims = 2 * num_frequencies * 2

        ## calculate the fixed, exact dimensionality of the concatenated global macro-code
        ## [mean_code (D) + std_code (D) + grid_centroids_flat (D * grid_dims)]
        self.global_code_dim = self.n_units = latent_feature_dim + latent_feature_dim + (latent_feature_dim * self.grid_dims)
        self.shape = (self.global_code_dim,)

        ## input stimulus compartments
        self.glimpse = Compartment(
            jnp.zeros((batch_size, latent_feature_dim)),
            display_name="Sensory Glimpse Latents (f_t)", units="dimensionless"
        )
        self.coordinate = Compartment(
            jnp.zeros((batch_size, 2)),
            display_name="Continuous Spatial Coordinates (y, x)", units="position"
        )

        ## stateful internal accumulator compartments
        self.k = Compartment(
            jnp.zeros((batch_size, 1)),
            display_name="Glimpse Counter Trace", units="count"
        )
        self.sum_features = Compartment(
            jnp.zeros((batch_size, latent_feature_dim)),
            display_name="Running Feature Sum", units="dimensionless"
        )
        self.sum_sq_features = Compartment(
            jnp.zeros((batch_size, latent_feature_dim)),
            display_name="Running Squared Feature Sum", units="dimensionless"
        )
        self.sum_weights = Compartment(
            jnp.zeros((batch_size, latent_feature_dim)),
            display_name="Running Feature Weight Accumulator Denominator", units="dimensionless"
        )
        self.running_centroids = Compartment(
            jnp.zeros((batch_size, latent_feature_dim, self.grid_dims)),
            display_name="Weighted Conjugate Grid-by-Feature Synaptic Phase Tensor", units="phase"
        )

        ## output readout compartment
        self.global_code = Compartment(
            jnp.zeros((batch_size, self.global_code_dim)),
            display_name="Global Macro-Representation Code Vector", units="dimensionless"
        )

    @compilable
    def advance_state(self, dt):
        ## processes the current step inputs (glimpse + coordinate) to update internal
        ### rate-coded synaptic traces and project the newly mutated global sketch-pad vector

        ## gather current inputs + state traces
        glimpse_t = self.glimpse.get()
        coord_t = self.coordinate.get()
        k_prev = self.k.get()
        sum_f_prev = self.sum_features.get()
        sum_sq_f_prev = self.sum_sq_features.get()
        sum_w_prev = self.sum_weights.get()
        centroids_prev = self.running_centroids.get()

        ## increment step counters (and update running statistics)
        k_curr = k_prev + 1.0
        sum_f_curr = sum_f_prev + glimpse_t
        sum_sq_f_curr = sum_sq_f_prev + (glimpse_t ** 2)

        ## construct spatial Fourier phases (grid module scales)
        powers = jnp.arange(self.num_frequencies, dtype=jnp.float32)
        frequencies = (2.0 ** powers) * jnp.pi  # Shape: (num_frequencies,)

        ## broadcast spatial vectors against frequency scaling modules
        phase_angles = coord_t[..., None] * frequencies
        sin_phases = jnp.sin(phase_angles)
        cos_phases = jnp.cos(phase_angles)
        ## flatten coordinate representation into uniform periodic grid-basis vector
        grid_coordinate_step = jnp.concatenate([sin_phases, cos_phases], axis=-1).reshape(
            (coord_t.shape[0], self.grid_dims)
        )

        ## perform conjunctive feature gating (feature-locked grid-cell phase-centroids)
        feature_weight_step = jnp.abs(glimpse_t)
        sum_w_curr = sum_w_prev + feature_weight_step
        ## NOTE: outer product to bind identity ("what") to coordinate manifold ("where")
        centroid_update = feature_weight_step[..., None] * grid_coordinate_step[:, None, :]
        centroids_curr = centroids_prev + centroid_update

        ## extract analytical descriptive metrics
        mean_code = sum_f_curr / k_curr
        var_code = jnp.maximum((sum_sq_f_curr / k_curr) - (mean_code ** 2), 0.0)
        std_code = jnp.sqrt(var_code)
        ## normalize phase tensor using cumulative structural step weights
        grid_centroids = centroids_curr / (sum_w_curr[..., None] + 1e-8)
        grid_centroids_flat = grid_centroids.reshape((glimpse_t.shape[0], -1))

        ## apply global predictive-coding standard L2 normalizations
        mean_code = mean_code / (jnp.linalg.norm(mean_code, axis=-1, keepdims=True) + 1e-8)
        std_code = std_code / (jnp.linalg.norm(std_code, axis=-1, keepdims=True) + 1e-8)
        grid_centroids_flat = grid_centroids_flat / (
                    jnp.linalg.norm(grid_centroids_flat, axis=-1, keepdims=True) + 1e-8)
        ## assemble unified global structural descriptor vector
        global_code_curr = jnp.concatenate([mean_code, std_code, grid_centroids_flat], axis=-1)

        ## Mutate internal state compartments safely
        self.k.set(k_curr)
        self.sum_features.set(sum_f_curr)
        self.sum_sq_features.set(sum_sq_f_curr)
        self.sum_weights.set(sum_w_curr)
        self.running_centroids.set(centroids_curr)
        self.global_code.set(global_code_curr)

    @compilable
    def reset(self):
        ## wipe sketch-pad and clear all accumulators back to baseline resting potentials
        ### clear transient step inputs if they are not bound directly as targets
        if not self.glimpse.targeted:
            self.glimpse.set(jnp.zeros((self.batch_size, self.latent_feature_dim)))
        if not self.coordinate.targeted:
            self.coordinate.set(jnp.zeros((self.batch_size, 2)))

        ## clear historical state traces
        self.k.set(jnp.zeros((self.batch_size, 1)))
        self.sum_features.set(jnp.zeros((self.batch_size, self.latent_feature_dim)))
        self.sum_sq_features.set(jnp.zeros((self.batch_size, self.latent_feature_dim)))
        self.sum_weights.set(jnp.zeros((self.batch_size, self.latent_feature_dim)))
        self.running_centroids.set(jnp.zeros((self.batch_size, self.latent_feature_dim, self.grid_dims)))
        ## clear macro-representation readout
        self.global_code.set(jnp.zeros((self.batch_size, self.global_code_dim)))
