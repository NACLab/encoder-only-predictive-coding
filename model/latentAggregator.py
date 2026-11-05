import jax
from jax import numpy as jnp, random, nn, jit, lax
from functools import partial #as bind
#from ngclearn.utils.io_utils import makedir

#cov_fx = jax.vmap(lambda x: jnp.cov(x, rowvar=False)) ## create vmapped-covar routine over 3D tensors

@partial(jit, static_argnums=[1])
def create_statistical_codes(z_i_list, normalize_maxcode=True):
    z_i = jnp.array(z_i_list)
    zmu = jnp.mean(z_i, axis=0)
    zsig = jnp.std(z_i, axis=0)
    z_maxpool = jnp.max(z_i, axis=0)
    if normalize_maxcode: ## Before concatenation:
        # zmu = zmu / (jnp.linalg.norm(zmu, axis=-1, keepdims=True) + 1e-8)
        # zsig = zsig / (jnp.linalg.norm(zsig, axis=-1, keepdims=True) + 1e-8)
        # z_maxpool = z_maxpool / (jnp.linalg.norm(z_maxpool, axis=-1, keepdims=True) + 1e-8)

        ## Now it safely shares the exact scale profile of other features/dimensions in code-space
        ## standardize alternative
        z_maxpool = ((z_maxpool - jnp.mean(z_maxpool, axis=-1, keepdims=True)) /
                     (jnp.std(z_maxpool, axis=-1, keepdims=True) + 1e-8))
        z_i = jnp.concatenate([zmu, z_maxpool, zsig], axis=-1)
        z_i = z_i / (jnp.linalg.norm(z_i, axis=-1, keepdims=True) + 1e-8)
    else:
        z_i = jnp.concat([zmu, z_maxpool, zsig], axis=1) ## dim = 2 * emb_dim
    ## old covariance code -- this proved to be too big to flatten
    #zSig = cov_fx(jnp.transpose(z_i, (1, 0, 2)))
    #zSig = jnp.reshape(zSig, (zSig.shape[0], zSig.shape[1] * zSig.shape[2]))
    #tril_indices = jnp.tril_indices(zSig.shape[1], k=0)
    #zTri = zSig[:, tril_indices[0], tril_indices[1]]
    return z_i

@jit
def create_global_spatial_codes( #compress_codes
        glimpse_features,
        coordinates
):
    """
    Compiles 40 glimpse codes and coordinates into a single spatial global vectors.

    Args:
        glimpse_features: tensor of shape (B, T, D) representing latents

        coordinates: tensor of shape (B, T, 2) representing continuous (y, x) positions in range [-1, 1]

    Returns:
        a 1D global representation array
    """
    ## compute standard statistics over 40 glimpses (axis=1)
    mean_code = jnp.mean(glimpse_features, axis=1)  # Shape: (Batch, 216)
    #max_code = jnp.max(glimpse_features, axis=1)  # Shape: (Batch, 216)
    std_code = jnp.std(glimpse_features, axis=1)  # Shape: (Batch, 216)
    mean_code = mean_code / (jnp.linalg.norm(mean_code, axis=-1, keepdims=True) + 1e-8)
    #max_code = max_code / (jnp.linalg.norm(max_code, axis=-1, keepdims=True) + 1e-8)
    std_code = std_code / (jnp.linalg.norm(std_code, axis=-1, keepdims=True) + 1e-8)

    ## compute spatial centroids for each feature channel
    ### use absolute activation magnitudes as weights s.t. positive/negative features both count
    feature_weights = jnp.abs(glimpse_features)  # Shape: (Batch, 40, 216)

    ## normalize weights across T glimpses so they sum to 1.0 per channel
    weight_denom = jnp.sum(feature_weights, axis=1, keepdims=True) + 1e-8
    normalized_weights = feature_weights / weight_denom  # Shape: (Batch, 40, 216)

    ## extract separate Y + X coordinates, expanding dimensions to broadcast against channels
    y_coords = coordinates[..., 0:1]  # Shape: (Batch, 40, 1)
    x_coords = coordinates[..., 1:2]  # Shape: (Batch, 40, 1)
    ## Compute the weighted average Y and X coordinate for every individual feature channel
    centroid_y = jnp.sum(normalized_weights * y_coords, axis=1)  # Shape: (B, n_channels)
    centroid_x = jnp.sum(normalized_weights * x_coords, axis=1)  # Shape: (B, n_channels)
    centroid_y = centroid_y / (jnp.linalg.norm(centroid_y, axis=-1, keepdims=True) + 1e-8)
    centroid_x = centroid_x / (jnp.linalg.norm(centroid_x, axis=-1, keepdims=True) + 1e-8)

    ## concatenate everything along feature axis into final 1D vector codes
    global_code = jnp.concatenate([
        mean_code,
        #max_code,
        std_code,
        centroid_y,
        centroid_x
    ], axis=-1)
    return global_code

@jit
def create_global_attention_code(
        glimpse_features,
        coordinates
):
    """
    Compiles K glimpse codes and coordinates into a single 1D vector via parameter-free attention.

    Args:
        glimpse_features:  tensor of shape (B, T, D) representing latents

        coordinates: tensor of shape (B, T, 2) representing continuous (y, x) positions in range [-1, 1]

    Returns:
        a 1D global representation array
    """
    batch_size, num_glimpses, feature_dim = glimpse_features.shape

    ## Compute standard baseline tracking statistics
    mean_code = jnp.mean(glimpse_features, axis=1)  # Shape: (B, D)
    #max_code = jnp.max(glimpse_features, axis=1)  # Shape: (B, D)
    std_code = jnp.std(glimpse_features, axis=1)  # Shape: (B, D)

    mean_code = mean_code / (jnp.linalg.norm(mean_code, axis=-1, keepdims=True) + 1e-8)
    # max_code = max_code / (jnp.linalg.norm(max_code, axis=-1, keepdims=True) + 1e-8)
    std_code = std_code / (jnp.linalg.norm(std_code, axis=-1, keepdims=True) + 1e-8)

    ## track semantic feature cross-attention values
    ### compute similarity between all glimpses: (B, T, D) x (B, D, T); this acts as raw query-key dot product matrix
    attn_logits_feat = jnp.einsum('btf,bsf->bts', glimpse_features, glimpse_features)
    ## scale by standard square root of dimension to stabilize variance
    attn_logits_feat = attn_logits_feat / jnp.sqrt(feature_dim)
    attn_weights_feat = jax.nn.softmax(attn_logits_feat, axis=-1)  # Shape: (Batch, 40, 40)

    ## pool features based on semantic similarity weights (acts as value aggregator)
    ### (B, T, T) x (B, T, D) -> (B, T, D)
    attended_features = jnp.einsum('bts,bsf->btf', attn_weights_feat, glimpse_features)
    ## average across glimpse pool dimension to arrive at a global 1D token
    global_attn_features = jnp.mean(attended_features, axis=1)  # Shape: (B, D)

    ## track spatial coordinate cross-attention
    ### compute similarity between glimpses based on spatial proximity in [-1, 1]
    ### negative squared distance creates a Gaussian-like radial basis function (RBF) kernel, tracking spatial closeness
    coord_diffs = coordinates[:, :, None, :] - coordinates[:, None, :, :]  # (B, T, T, 2)
    sq_distances = jnp.sum(coord_diffs ** 2, axis=-1)  # (B, T, T)
    ## scale logit weights (acts like setting a standard RBF length-scale variance)
    attn_logits_coord = -0.5 * sq_distances
    attn_weights_coord = jax.nn.softmax(attn_logits_coord, axis=-1)  # Shape: (B, T, T)
    ## pool features based on how physically close they are to each other
    ## (B, T, T) x (B, T, D) -> (B, T, D)
    spatially_grouped_features = jnp.einsum('bts,bsf->btf', attn_weights_coord, glimpse_features)
    global_attn_coords = jnp.mean(spatially_grouped_features, axis=1)  # Shape: (B, D)

    ## normalize attention features + coordinate features
    global_attn_features = global_attn_features / (
            jnp.linalg.norm(global_attn_features, axis=-1, keepdims=True) + 1e-8
    )
    global_attn_coords = global_attn_coords / (
            jnp.linalg.norm(global_attn_coords, axis=-1, keepdims=True) + 1e-8
    )

    ## create final global latent codes
    global_codes = jnp.concatenate([
        mean_code,  # 216 dims
        #max_code,  # 216 dims
        std_code,  # 216 dims
        global_attn_features,  # 216 dims (Feature-relational pooled vectors)
        global_attn_coords  # 216 dims (Spatial-relational pooled vectors)
    ], axis=-1)  # Total: 1080 dims

    return global_codes

@partial(jit, static_argnums=[2])
def create_global_grid_cell_codes(
        glimpse_features,
        coordinates,
        num_frequencies=6 #5 #4
):
    """
    Creates global vector by projecting coordinates into grid-cell-like periodic phases.

    Args:
        glimpse_features:  tensor of shape (B, T, D) representing latents

        coordinates: tensor of shape (B, T, 2) representing continuous (y, x) positions in range [-1, 1]

        num_frequencies: integer controlling the number of geometric frequency scales;
            this must be static/constant across runs

    Returns:
        a 1D global representation array
    """
    n_glimpses = glimpse_features.shape[1] ## get num glimpses

    mean_code = jnp.mean(glimpse_features, axis=1)
    #max_code = jnp.max(glimpse_features, axis=1)
    std_code = jnp.std(glimpse_features, axis=1)
    mean_code = mean_code / (jnp.linalg.norm(mean_code, axis=-1, keepdims=True) + 1e-8)
    #max_code = max_code / (jnp.linalg.norm(max_code, axis=-1, keepdims=True) + 1e-8)
    std_code = std_code / (jnp.linalg.norm(std_code, axis=-1, keepdims=True) + 1e-8)

    ## dynamically generate geometric frequencies: [1.0, 2.0, 4.0, 8.0, 16.0, ...]
    powers = jnp.arange(num_frequencies, dtype=jnp.float32)
    frequencies = (2.0 ** powers) * jnp.pi  # Shape: (num_frequencies,)
    ## broadcast & compute spatial frequencies
    # (B, T, 2, 1) * (num_frequencies,) -> (B, T, 2, num_frequencies)
    phase_angles = coordinates[..., None] * frequencies
    sin_phases = jnp.sin(phase_angles)
    cos_phases = jnp.cos(phase_angles)

    ## flatten periodic coordinates to shape: (B, T, 2 * num_frequencies * 2)
    ### e.g., for num_frequencies=5, grid_dims = 2 * 5 * 2 = 20 dimensions per feature channel.
    grid_dims = 2 * num_frequencies * 2
    grid_coordinates = jnp.concatenate([sin_phases, cos_phases], axis=-1).reshape(
        (coordinates.shape[0], n_glimpses, grid_dims)
    )
    ## compute feature weight distributions
    feature_weights = jnp.abs(glimpse_features)
    weight_denom = jnp.sum(feature_weights, axis=1, keepdims=True) + 1e-8
    normalized_weights = feature_weights / weight_denom
    ## compute phase-centroids (via Einstein summation)
    ### contract batch-time-feature and batch-time-phase into batch-feature-phase
    grid_centroids = jnp.einsum('btf,btp->bfp', normalized_weights, grid_coordinates)
    grid_centroids_flat = grid_centroids.reshape((glimpse_features.shape[0], -1))
    grid_centroids_flat = grid_centroids_flat / (jnp.linalg.norm(grid_centroids_flat, axis=-1, keepdims=True) + 1e-8)

    ## final output global code creation
    global_code = jnp.concatenate(
        [
            mean_code,
            #max_code,
            std_code,
            grid_centroids_flat
        ],
        axis=-1
    )
    return global_code

class LatentAggregator(): ## crafts global latent codes from glimpses

    def __init__(
            self,
            dkey,
            mode="grid_cell", ## OPTIONS: statistical, allocentric_reference, grid_cell
            compress_latents=False,
            **kwargs
    ):
        self.dkey, *subkeys = random.split(dkey, 5)
        self.mode = mode
        ## metadata associated w/ this aggregator
        self.glimpse_codes = []
        self.coordinate_codes = []
        self.compress_latents = compress_latents

    def update(self, glimpse_code, coordinates_code):
        self.glimpse_codes.append(glimpse_code)
        self.coordinate_codes.append(coordinates_code)

    def compute_code(self): ## compute current code given current set of glimpses
        if self.compress_latents:
            glimpse_features = jnp.array(self.glimpse_codes)
            glimpse_features = jnp.transpose(glimpse_features, (1, 0, 2))  # Becomes (B, 40, 216)
            coordinates = jnp.array(self.coordinate_codes)
            coordinates = jnp.transpose(coordinates, (1, 0, 2))  # Becomes (B, 40, 2)

            if self.mode == "attention": ## cross-attention aggregation
                global_codes = create_global_attention_code(glimpse_features, coordinates)
            elif self.mode == "grid_cell": ## grid-cell spatial central aggregation
                global_codes = create_global_grid_cell_codes(glimpse_features, coordinates)
            elif self.mode == "allocentric_reference": ## allocentric reference map spatial centroid aggregation
                global_codes = create_global_spatial_codes(glimpse_features, coordinates)
            else: ## simple mean+max-pool+std-dev statistical aggregation
                global_codes = create_statistical_codes(glimpse_features)
        else:
            global_codes = jnp.concat(self.glimpse_codes, axis=1) ## create super-vector codes
        return global_codes

    def reset(self):
        self.glimpse_codes = []
        self.coordinate_codes = []
