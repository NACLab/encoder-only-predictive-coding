import sys, getopt as gopt, optparse, time, json
from jax import numpy as jnp, random, jit
from ngclearn.utils.metric_utils import measure_sparsity, measure_gini_index
from model.agent import Agent
from model import LGN_Glimpser, MPCEncoder

# I/O header message for this module
msg = ("#################################################################\n"
       "## Executing Latent Code Extraction Process\n"
       "#################################################################")
print(msg)

seed = 42
model_dir = "exp_out/"
config_fname = ""
data_fname = f"../../../data/mnist/testX.npy"
px = py = 28
trial_id = ""
config = {}
compress_latents = False
n_glimpses = 40 # 50 #40 #20
glimpse_bounds = 0.25 #0.2
saccade_policy_override = False
latents_fname = "latentsX"
glimpse_policy = "random"
verbose = False

######################################################################################
## read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:],
    '',
    [
        "data_fname=",
        "exp_dir=",
        "trial_id=",
        "config_fname=",
        "seed=",
        "compress_latents=",
        "n_glimpses=",
        "glimpse_bounds=",
        "glimpse_policy=",
        "latents_fname="
    ]
)
for opt, arg in options:
    if opt in ("--data_fname"):
        data_fname = arg.strip()
        if "kmnist" in data_fname:
            dataset = "kmnist"
            px = py = 28
        elif "mnist" in data_fname:
            dataset = "mnist"
            px = py = 28
        elif "norb" in data_fname:
            dataset = "norb"
            px = py = 96
        elif "olivetti" in data_fname:
            px = py = 64
        elif "coil20" in data_fname:
            dataset = "coil20" ## (B, 1, 128, 128)
            px = py = 128
        elif "eth80" in data_fname:
            dataset = "eth80" ## (B, 1, 64, 64)
            px = py = 64 #256
        elif "cifar" in data_fname or "svhn" in data_fname:  ## >50,000 (~70,000) images
            ## NOTE: try zca-whitening cifar/svhn before using mpc (?)
            px = py = 32
            # foveal_dim = patch_dim = 8
            # parafoveal_dim = 16
            # peripheral_dim = 24
        elif "natural" in data_fname:
            dataset = "natural"
            px = py = 512
        elif "vanHateren" in data_fname:
            dataset = "vanHateren"
            px = py = 512
    elif opt in ("--exp_dir"):
        exp_dir = arg.strip()
    elif opt in ("--verbose"):
        verbose = (arg.strip().lower() == "true")
    elif opt in ("--trial_id"):
        trial_id = arg.strip()
    elif opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--n_glimpses"):
        n_glimpses = int(arg.strip())
    elif opt in ("--glimpse_bounds"):
        glimpse_bounds = float(arg.strip())
    elif opt in ("--glimpse_policy"):
        glimpse_policy = arg.strip().lower()
        saccade_policy_override = True
    elif opt in ("--compress_latents"):
        compress_latents = (arg.strip().lower() == "true" )
    elif opt in ("--latents_fname"):
        latents_fname = arg.strip()
    elif opt in ("--config_fname"):
        config_fname = arg.strip()
        with open(config_fname, 'r') as file:
            ## Parse file content into a dictionary
            config = json.load(file)
######################################################################################

model_dir = config.get(trial_id, {}).get("modelDir")
model_type = config.get("modelType")

batch_size = int(config.get("hyperParameters", {}).get("batch_size")) # 200 #1000
foveal_shape = patch_shape = tuple(config.get("hyperParameters").get("foveal_shape"))
parafoveal_shape = tuple(config.get("hyperParameters").get("parafoveal_shape"))
peripheral_shape = tuple(config.get("hyperParameters").get("peripheral_shape"))
input_filter = config.get("hyperParameters").get("input_filter")
input_scale = config.get("hyperParameters").get("input_scale")
use_fine_grained_filter = bool(config.get("hyperParameters").get("use_fine_grained_filter"))
if not saccade_policy_override:
    #use_epistemic_saccades = bool(config.get("hyperParameters").get("use_epistemic_saccades"))
    glimpse_policy = config.get("hyperParameters").get("glimpse_policy")
else:
    print(" > Overriding saccade policy with ", glimpse_policy)
print(f"Glimpse params: T: {n_glimpses}  bound: [{-glimpse_bounds}, {glimpse_bounds}] (Policy: {glimpse_policy})")


dist_fname = f"{model_dir}{latents_fname}_distances.npy"
lat_fname = f"{model_dir}{latents_fname}.npy"
print(lat_fname, "\n", dist_fname)

key = random.PRNGKey(seed)
eyeball = LGN_Glimpser(
    key,
    data_fname,
    image_shape=(px, py),
    batch_size=batch_size,
    foveal_shape=foveal_shape,
    parafoveal_shape=parafoveal_shape,
    peripheral_shape=peripheral_shape,
    n_glimpses=n_glimpses,
    dxy=0.,
    glimpse_bounds=glimpse_bounds,
    center_patches=False, #True,
    input_filter=input_filter,
    input_scale=input_scale,
    use_fine_grained_filter=use_fine_grained_filter, 
    max_saccades=n_glimpses 
)
print(" DATA.SOURCE.shape = ", eyeball.images.shape)

if "mpc" in model_type:
    model: Agent = MPCEncoder(
        key,
        model_config=config,
        trial_id=trial_id
    )
else:
    print(f"Error: unsupported model type ({model_type})")
    exit(1)

if verbose:
    print(
        f"{model.get_param_stats()}"
    )
W = model.model.get_components("W1").weights.get().T

lat_dim = model.model.get_components("z2").n_units
chunk_size = -1 ## TODO: use this parameter to process data from disk?
if eyeball.images.shape[0] > 10000:
    chunk_size = 5000

print(f" >>> Collecting latents over: {data_fname}")
test_data = jnp.load(data_fname)
print(" >>> X.shape: ", test_data.shape, " Data.Norm: ", jnp.linalg.norm(test_data))


### TODO: currently assumes number of data-points divisible by batch-size
n_batches = eyeball.images.shape[0] // batch_size #test_data.shape[0] // batch_size

z_codes = []
latent_distances = []
Ns = Ng = 0 ## number of samples, number of glimpses
F = mean_sparsity = mean_gini = max_val = 0.
s_ptr = 0
e_ptr = batch_size
sim_t = time.time()
for i in range(n_batches):
    batch_ptrs = jnp.arange(s_ptr, e_ptr, 1)
    s_ptr += batch_size
    e_ptr += batch_size
    eyeball.reset(batch_pointers=batch_ptrs)
    Ns += batch_ptrs.shape[0]

    _sparsity = 0.
    _gini = 0.
    z_i = [] ## latent codes for this series of glimpses
    actions = []
    z_dist = [] ## latent code distances, i.e., D(z_g-1, z_g)
    z_gm1 = None
    model.reset_scene_level_gridcells() ## clears out grid-cells of MPC model
    for g in range(n_glimpses): ## process input scene thru series of MPC-selected glimpses
        x_g, a_g = eyeball.step_saccade(policy=glimpse_policy)
        Ng += x_g.shape[0]

        #eyeball.render(output_dir="tmp/")
        r_L, F_j, F_j_batch, z_stats = model.process(
            x_g, action=a_g, adapt_synapses=False
        )
        if "olivetti" in data_fname:
            F_j = jnp.sum(F_j_batch[0:test_data.shape[0],:])
        z_g = z_stats[-1]
        if g > 0:
            z_dist.append( jnp.linalg.norm(z_g - z_gm1, axis=1, keepdims=True) )
        z_gm1 = z_g
        eyeball.update_glimpser(-F_j_batch)

        _sparsity = jnp.sum(measure_sparsity(z_g, preserve_batch=False)) + _sparsity
        _gini = jnp.sum(measure_gini_index(z_g, preserve_batch=False)) + _gini
        #actions.append(a_g)

        F = F_j + F ## track current cross-glimpse free-energy
    ## create batch of B x T (T = # glimpses, B = batch size)
    z_dist = jnp.concat(z_dist, axis=1)
    latent_distances.append(z_dist)

    z_i = model.model.get_components("grid_cells").global_code.get() ## extract grid-cell scene activity codes

    max_val = float(jnp.maximum(max_val, jnp.max(z_i)))
    z_codes.append(z_i)
    ## calc final glimpse-averaged scores
    _sparsity = _sparsity/n_glimpses
    mean_sparsity += _sparsity
    _gini = _gini/n_glimpses
    mean_gini += _gini
    if verbose:
        print(
            f"\r {i} / {n_batches}: E = {F / Ng:.4f} Sparsity = {mean_sparsity/(i+1):.4f} "
            f"Gini.index: {mean_gini/(i+1):.4f}  Max.Lat = {max_val:.4f} ({int(Ns)} samples; {int(Ng)} saccades)",
            end=""
        )
    else:
        print(
            f"\r {i} / {n_batches}: E = {F / Ng:.4f} Sparsity = {mean_sparsity / (i + 1):.4f} "
            f"Gini.index: {mean_gini / (i + 1):.4f}  ({int(Ns)} samples; {int(Ng)} saccades)",
            end=""
        )
print()
mean_sparsity = mean_sparsity/n_batches
mean_gini = mean_gini / n_batches

latent_distances = jnp.concat(latent_distances, axis=0) ## create distance data matrix
z_codes = jnp.concat(z_codes, axis=0) ## create final latent code set
sim_t = float(jnp.round(time.time() - sim_t, 4))
print(f" >> Extraction.Time: {sim_t:.2f} s ({(sim_t/60.):.2f} m)")

if compress_latents:
    _mean_sparsity = float(measure_sparsity(z_codes, preserve_batch=False))
    _mean_gini = float(measure_gini_index(z_codes, preserve_batch=False))
    print(f" Compressed.Mean Sparsity = {_mean_sparsity:.4f}  Gini = {_mean_gini:.4f}")
else:
    print(f" Mean Sparsity = {mean_sparsity:.4f}  Gini = {mean_gini:.4f}")
print(
    f" Z.shape: {z_codes.shape}; Range: {float(jnp.min(z_codes))}, {float(jnp.max(z_codes))} "
    f" Stat: {float(jnp.mean(z_codes))} +/ {float(jnp.std(z_codes))}"
)

## save latents to disk
### 1st, check that latents collected match original dataset size
#if "olivetti" in data_fname:
if z_codes.shape[0] != test_data.shape[0]:
    #print("OLD SHAPE: ", z_codes.shape, "  ", latent_distances.shape)
    #print(" >>> Shaving to ", test_data.shape[0])
    z_codes = z_codes[0:test_data.shape[0], :]
    latent_distances = latent_distances[0:test_data.shape[0], :]
    #print("NEW SHAPE: ",z_codes.shape, "  ", latent_distances.shape)

print(f"Save latent distances {latent_distances.shape} to: ", dist_fname)
jnp.save(dist_fname, latent_distances)
print(f"Save latents {z_codes.shape} to: ", lat_fname)
jnp.save(lat_fname, z_codes)

## update trial-record with measurements
_trial = config.get(f"{seed}")
if _trial is not None:
    _trial["test_sparsity"] = float(mean_sparsity)
    _trial["test_gini"] = float(mean_gini)
    config[f"{seed}"] = _trial

## save updated trial / config dictionary
with open(config_fname, "w", encoding="utf-8") as out_fd:
    json.dump(config, out_fd, indent=4)
