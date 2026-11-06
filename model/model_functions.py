import jax
from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn.utils.viz.synapse_plot import visualize
from ngclearn.utils.metric_utils import measure_sparsity, measure_gini_index
from .agent import Agent

'''
General modeling functions/co-routines
'''

def eval_model( ## generic model evaluation loop call
        model : Agent, testData: jax.Array, batch_size:int, collect_codes:bool=False
):
    F = Ns = 0.
    z_metrics = jnp.zeros((1, 3)) ## mean-sparsity, gini(z), max(z)
    z_codes = []
    n_batches = testData.shape[0] // batch_size
    for j in range(n_batches):
        x_j = testData[j * batch_size:(j+1) * batch_size, :]
        Ns += x_j.shape[0]

        F_j, z_stats = model.process(x_j, adapt_synapses=False)
        zF2 = z_stats[-1]
        _spars = float(jnp.sum(measure_sparsity(zF2)))
        _gini = measure_gini_index(zF2)
        _gini = float(jnp.sum(_gini))
        _mean_maxval = float(jnp.sum(jnp.max(zF2, axis=1)))
        z_metrics = z_metrics + jnp.array([[_spars, _gini, _mean_maxval]])

        F = F_j + F ## update free energy estimate

        if collect_codes:
            print(f"\r F = {F/Ns:.4f}; Collected {int(Ns)} codes", end="")
            #z1, zF1, z2, zF2 = z_stats
            z_codes.append(zF2)
    if collect_codes:
        print()
        z_codes = jnp.concat(z_codes, axis=0)
    else: ## empty array
        z_codes = jnp.array(z_codes)
    z_metrics = z_metrics / testData.shape[0]
    return F/Ns, z_metrics, z_codes

@jit
def calc_num_blanks(latents): ## compute number of "blank" latents
    _latents = jnp.abs(latents)
    nblanks = jnp.sum(jnp.sum(_latents, axis=1, keepdims=True) <= 0., dtype=jnp.int32) ## number blanks
    return nblanks

@jit
def calc_rf_diversity(W):
    ## W: matrix of shape (num_neurons, flattened_features)

    ## Normalize receptive fields to unit length - ensures diversity based on
    ## pattern shape, not magnitude.
    norms = jnp.linalg.norm(W, axis=1, keepdims=True)
    W_norm = W / (norms + 1e-8)
    ## Pairwise Cosine Distance - Compute dot product matrix (N x N)
    cosine_sim = jnp.dot(W_norm, W_norm.T)
    cosine_dist = 1.0 - cosine_sim
    ## Extract off-diagonal elements (unique pairs)
    n = W.shape[0]
    mask = jnp.eye(n) == 0
    avg_dist = jnp.sum(cosine_dist * mask) / (n * (n - 1))
    ## Spectral Entropy (Effective Dimension)
    ## Measures how many orthogonal directions the RFs cover.
    _, s, _ = jnp.linalg.svd(W_norm, full_matrices=False)
    variance_explained = s**2
    prob = variance_explained / jnp.sum(variance_explained)
    entropy = -jnp.sum(prob * jnp.log(prob + 1e-10))
    return avg_dist, entropy

@partial(jit, static_argnums=[1])
def calc_sparsity(codes, flip_measurement=False):
    dim = codes.shape[1]
    nonzero = jnp.sum(codes > 0., axis=1, keepdims=True)
    sparsity = nonzero/dim
    if flip_measurement:
        sparsity = 1. - sparsity
    return jnp.mean(sparsity) ## compute mean sparsity


def get_stats(param: jax.Array): ## basic stat/range computer
    stats = f"[{jnp.amin(param):.4f}, {jnp.amax(param):.4f}] N: {jnp.linalg.norm(param):.4f}"
    return stats

def viz_filters( ## block-matrix filter visualization routine
        W: jax.Array, 
        fname:str, 
        patch_shape:tuple[int, int], 
        stream_dim:int, 
        n_streams:int,
        n_plot_per_row:list[int]=None, 
        ndup=0 #1 #0
):
    Wset = []
    shift = 0
    for i in range(n_streams):
        W_i = W[
            shift * (patch_shape[0] * patch_shape[1]) : (shift + 1) * (patch_shape[0] * patch_shape[1]),
            i * stream_dim:(i + 1) * stream_dim
        ]
        Wset.append(W_i)
        if ndup > 0:
            W_i = W[
                (shift + 1) * (patch_shape[0] * patch_shape[1]) : (shift + 2) * (patch_shape[0] * patch_shape[1]),
                i * stream_dim:(i + 1) * stream_dim
            ]
            shift += 2
            Wset.append(W_i)
        else:
            shift += 1
        #Wset.append(W_i)
    if n_plot_per_row is not None:
        sptr = 0
        for p in range(len(n_plot_per_row)):
            n_p = n_plot_per_row[p]
            Wsubset = Wset[sptr:sptr+n_p]
            visualize(Wsubset, [patch_shape] * len(Wsubset), fname+f"_{p}") #, contrast_by_data=True)
            sptr += n_p
    else:
        visualize(Wset, [patch_shape] * len(Wset), fname) #, contrast_by_data=True)

