import json
from jax import numpy as jnp, random, nn, jit
import numpy as np
import sys, getopt as gopt, optparse
from ngclearn.utils.io_utils import makedir
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_sparsity
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, extract_pca_latents, plot_latents
from model.model_functions import calc_num_blanks

config = {}
trial_id = ""
data_fname = labels_fname = ""
## obtain correct directories
model_dir = ""
codebookName = "latent_codes"
suffix = "jpg" # "pdf"

# read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:], '',
    [
        "subset=",
        "labels_fname=",
        "data_fname=",
        "trial_id=",
        "config_fname="
    ]
)
# external dataset arguments
for opt, arg in options:
    if opt in ("--labels_fname"):
        labels_fname = arg.strip()
    elif opt in ("--data_fname"):
        data_fname = arg.strip()
    elif opt in ("--trial_id"):
        trial_id = arg.strip()
    elif opt in ("--config_fname"):
        config_fname = arg.strip()
        with open(config_fname, 'r') as file:
            ## Parse file content into a dictionary
            config = json.load(file)
######################################################################################
print("> Applying t-SNE to ", data_fname)

model_dir = config.get(trial_id, {}).get("modelDir")
if model_dir is None:
    model_dir = "tsne_outs/"
    makedir(model_dir)
latents = jnp.load(data_fname)
if len(latents.shape) > 2:
    latents = jnp.reshape(latents, (latents.shape[0], latents.shape[1] * latents.shape[2]))
Y = jnp.load(labels_fname) ## load in label design matrix

print(" >> Constructing tSNE visualization of model (rate-)codes...")
tsneDir = "{}/tsne/".format(model_dir)
codes_fname = "{}{}_tsne.npy".format(tsneDir, codebookName)
makedir(tsneDir)

lmin = jnp.amin(latents) ## latent min val
lmax = jnp.amax(latents) ## latent max val
lmu = jnp.mean(latents) ## latent mu
lstd = jnp.std(latents) ## latent sigma
nblanks = calc_num_blanks(latents)
print(
    f"Rate-code latents.shape = {latents.shape}; NB: {nblanks:.1f}  "
    f"min: {lmin:.4f}  max {lmax:.4f} x: {lmu:.4f} +/- {lstd:.4f}"
)
rho = measure_sparsity(latents)
print(f"Sparsity = {jnp.mean(rho):.4f} +/- {jnp.std(rho):.4f}")
n_pca_comp = 50 #30 #400 #60 #200 #400
codes = extract_tsne_latents(np.asarray(latents), perplexity=30, n_pca_comp=n_pca_comp, batch_size=500)
#codes = jnp.load("exp_model/tsne/rpc_latents_tsne.npy")
print("tSNE-codes.shape = ", codes.shape)
jnp.save(codes_fname, codes) ## save tSNE codes to disk

## produce tSNE plot of final projected codes
alpha = 0.55 #1. # set to value > 0 and < 1 if more transparent clusters are desired
plot_latents(
    codes,
    Y,
    plot_fname=f"{tsneDir}tsne_codes.{suffix}", #.pdf jpg
    alpha=alpha,
    xaxis_title="X Coordinate (tSNE units)",
    yaxis_title="Y Coordinate (tSNE units)"
) ## .png

