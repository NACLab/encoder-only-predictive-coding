import sys, getopt as gopt, optparse, time, json
from jax import numpy as jnp, random, nn, jit, lax
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.metric_utils import measure_sparsity
from model import LGN_Glimpser, Agent, MPCEncoder
from model.model_functions import viz_filters, calc_num_blanks

# I/O header message for this module
msg = ("#################################################################\n"
       "## Executing Model Training Process\n"
       "#################################################################")
print(msg)

seed = 42
save_filter_seq = False # True
config_fname = ""
config = {}
exp_dir = "exp_out/"
makedir(exp_dir)
data_fname = f"../../../data/mnist/trainX.npy"
dataset = "mnist"
px = py = 28
glimpse_bounds = 0.25 
n_glimpses = 10
verbose = False

######################################################################################
## read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:],
    '',
    [
        "data_fname=", 
        "exp_dir=", 
        "config_fname=", 
        "seed=", 
        "n_glimpses=",
        "glimpse_bounds="
        "verbose="
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
            # foveal_dim = patch_dim = 24  # 14 #24
            # parafoveal_dim = 48  # 28 #48
            # peripheral_dim = 72  # 60 #56 #60
        elif "olivetti" in data_fname or "faces" in data_fname:
            dataset = "olivetti"
            px = py = 64
            # foveal_dim = patch_dim = 12  # 16 # 16 #8
            # parafoveal_dim = 24  # 28 #32 #16
            # peripheral_dim = 48  # 48 #24
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
    elif opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--n_glimpses"):
        n_glimpses = int(arg.strip())
    elif opt in ("--glimpse_bounds"):
        glimpse_bounds = float(arg.strip())
    elif opt in ("--config_fname"):
        config_fname = arg.strip()
        with open(config_fname, 'r') as file:
            config = json.load(file) ## parse file content into a dictionary
######################################################################################

model_type = config.get("modelType")
model_dir = f"{exp_dir}{model_type}/{dataset}/{seed}/"
if verbose:
    print(">> ", model_dir)
makedir(model_dir) ## create model directory if not available

probe_mod = 100 # 100000
save_filter_seq = bool(config.get("hyperParameters").get("save_filter_seq"))
n_iter = int(config.get("hyperParameters").get("n_iter")) #20 #10 #50
batch_size = int(config.get("hyperParameters").get("batch_size"))
foveal_shape = patch_shape = tuple(config.get("hyperParameters").get("foveal_shape"))
parafoveal_shape = tuple(config.get("hyperParameters").get("parafoveal_shape"))
peripheral_shape = tuple(config.get("hyperParameters").get("peripheral_shape"))
patch_dim = patch_shape[0]
input_filter = config.get("hyperParameters").get("input_filter")
input_scale = config.get("hyperParameters").get("input_scale")
use_fine_grained_filter = bool(config.get("hyperParameters").get("use_fine_grained_filter"))
glimpse_policy = config.get("hyperParameters").get("glimpse_policy")

if verbose:
    print(f"Glimpse params: T: {n_glimpses}  bound: [{-glimpse_bounds}, {glimpse_bounds}]")

key = random.PRNGKey(seed)
key, *subkeys = random.split(key, 15)
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
    glimpse_bounds=glimpse_bounds, #0.2,
    center_patches=True,
    input_filter=input_filter,
    input_scale=input_scale,
    use_fine_grained_filter=use_fine_grained_filter, 
    max_saccades=n_glimpses
)

## set up model
model : Agent = None
if "mpc" in model_type:
    print(" >> Constructing encoder-only model")
    model = MPCEncoder(
        key,
        model_config=config,
        saveDir=model_dir
    )
else:
    print(f"Error: unsupported model type ({model_type})")
    exit(1)
model.save_to_disk()
if config.get(f"{seed}") is None:
    config.setdefault(
        f"{seed}", {"modelDir" : model_dir}
    )
    ## save updated trial / config dictionary
    with open(config_fname, "w", encoding="utf-8") as out_fd:
        json.dump(config, out_fd, indent=4)

if verbose:
    print("+++++++++++++++++++\n", model.get_param_stats(), "\n+++++++++++++++++++")

ftag = "i"
if save_filter_seq:
    ftag = "0"
_W1_recfields =  model.params[0].weights.get()
if "gpc" in model_type:
    _W1_recfields = _W1_recfields.T
viz_filters(
    _W1_recfields, f"{model_dir}filters_{ftag}", patch_shape, model.Nz1, model.n_streams1
)

energies = []
F_window = []
Fint_window = []
win_len = 100 #50 # 100
tick = 0
Fint = F = 0.
Ns = Ng = 0 ## number samples, number glimpses
mean_sparsity = 0.
max_val = 0.
nblanks = 0
sim_t = time.time()
for i in range(n_iter):
    eyeball.reset()
    Ns += eyeball._image_batch.shape[0]

    _sparsity = 0.
    F_i = Fint_i = 0.
    for g in range(n_glimpses):
        x_g, a_g = eyeball.step_saccade(policy=glimpse_policy)
        Ng += x_g.shape[0]

        #eyeball.render(output_dir="tmp/", sample_idx=0)
        #exit()
        Fint_j, F_j, F_j_batch, z_stats = model.process(x_g, action=a_g, adapt_synapses=True)
        eyeball.update_glimpser(-F_j_batch)

        zF2 = z_stats[-1]
        _sparsity = jnp.sum(measure_sparsity(zF2, preserve_batch=False)) + _sparsity
        max_val = float(jnp.maximum(max_val, jnp.max(zF2)))
        nblanks = calc_num_blanks(zF2) + nblanks
        F_i = F_j + F_i ## sum raw energies
        Fint_i = Fint_j + Fint_i

    F_int_i = Fint_i / (n_glimpses * batch_size)
    Fint_window.append(F_int_i)
    if len(Fint_window) > win_len:
        Fint_window.pop(0)
    Fint = jnp.mean(jnp.array(Fint_window))

    F_i = F_i/(n_glimpses * batch_size)
    F_window.append(F_i)
    if len(F_window) > win_len:
        F_window.pop(0)
    F = jnp.mean(jnp.array(F_window))


    _sparsity = _sparsity/n_glimpses
    mean_sparsity += _sparsity
    if verbose:
        print(
            f"\r {i}/{n_iter}: E(in) = {Fint:.4f} E = {F:.4f} Sparsity = {mean_sparsity / (i + 1):.4f} "
            f"Max.Lat = {max_val:.4f} Nblanks: {nblanks} ({int(Ns)} samples; {int(Ng)} saccades)",
            end=""
        )
    else:
        print(
            f"\r {i}/{n_iter}: E(in) = {Fint:.4f} E = {F:.4f} "
            f"Sparsity = {mean_sparsity / (i + 1):.4f} ({int(Ns)} samples; {int(Ng)} saccades)",
            end=""
        )

    if i > 0 and i % probe_mod == 0: #if Ns % probe_mod == 0:
        if verbose:
            print()
        energies.append(F) #energies.append(F/Ng)
        tick += 1
        if verbose:
            print("+++++++++++++++++++\n", model.get_param_stats(), "\n+++++++++++++++++++")
        model.save_to_disk(params_only=True)
        ftag = "f"
        if save_filter_seq:
            ftag = f"{tick}"
        _W1_recfields = model.params[0].weights.get()
        if "gpc" in model_type:
            _W1_recfields = _W1_recfields.T
        viz_filters(
            _W1_recfields, f"{model_dir}filters_{ftag}", patch_shape, model.Nz1, model.n_streams1
        )
if verbose:
    print()
sim_t = float(jnp.round(time.time() - sim_t, 4))
print(f" >> Full Simulation.Time: {sim_t:.2f} s ({(sim_t/60.):.2f} m)")
energies = jnp.array(energies)
jnp.save(f"{model_dir}energy.npy", jnp.array(energies))

_trial = config.get(f"{seed}")
if _trial is not None:
    _trial["train_time_in_sec"] = sim_t
    if Ns <= 0.:
        Ns = 1.
    _trial["train_energy_in_nats"] = float(jnp.round(F, 4))
    config[f"{seed}"] = _trial
## save updated trial / config dictionary
with open(config_fname, "w", encoding="utf-8") as out_fd:
    json.dump(config, out_fd, indent=4)

