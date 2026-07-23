import sys, getopt as gopt, optparse, time, json
from jax import numpy as jnp, random
from ngclearn.utils.analysis import KNNProbe
from ngclearn.utils.metric_utils import measure_ACC, analyze_scores 

# I/O header message for this module
msg = ("#################################################################\n"
       "## Executing KNN-Probe Downstream Evaluation Process\n"
       "#################################################################")
print(msg)

seed = 42
metric_tag = config_fname = data_fname = lab_fname = test_data_fname = test_lab_fname = ""
######################################################################################
## read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:],
    '',
    [
        "data_fname=",
        "lab_fname=",
        "test_data_fname=",
        "test_lab_fname=",
        "config_fname=",
        "seed=",
        "metric_tag=" ## adds in a string post-fix tag to stored measurements in JSON config
    ]
)
for opt, arg in options:
    if opt in ("--data_fname"):
        data_fname = arg.strip()
    elif opt in ("--lab_fname"):
        lab_fname = arg.strip()
    if opt in ("--test_data_fname"):
        test_data_fname = arg.strip()
    elif opt in ("--test_lab_fname"):
        test_lab_fname = arg.strip()
    elif opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--metric_tag"):
        metric_tag = arg.strip()
    elif opt in ("--config_fname"):
        config_fname = arg.strip()
        with open(config_fname, 'r') as file:
            ## Parse file content into a dictionary
            config = json.load(file)
######################################################################################

X = jnp.load(data_fname)
Y = jnp.load(lab_fname)
vX = jnp.load(test_data_fname)
vY = jnp.load(test_lab_fname)

#batch_size = int(config.get("hyperParameters", {}).get("batch_size"))
batch_size = 100
if "olivetti" in data_fname:
    batch_size = 10
elif "eth80" in data_fname:
    batch_size = 2
print(" >> Batch.size = ", batch_size)
print(X.shape, " ", Y.shape)
print(vX.shape, " ", vY.shape)

D = X.shape[1]
C = Y.shape[1]
dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 3)
knn = KNNProbe(
    subkeys[0],
    source_seq_length=1,
    input_dim=D,
    out_dim=C,
    num_neighbors=6, #8, #6, ##3 (used 3 for norb/mnist/kmnist; 6 for eth80)
    vote_style="mode",
    #distance_function=("cosine", 1)
    distance_function=("minkowski", 1) #1) ## L1-dist
)

## "training step"
print(" >> Fitting model!")
knn.update(X, Y) ## fit KNN to data

## evaluation / inference step
print(" >> Evaluation!")
n_batches = vX.shape[0] // batch_size
acc = 0.
Ns = 0.
mu_matrix = []
for i in range(n_batches):
    xi = vX[i*batch_size:(i+1)*batch_size, :]
    yi = vY[i*batch_size:(i+1)*batch_size, :]
    y_hat = knn.process(xi)
    #print(i,"\n",jnp.sum(jnp.sum(yi, axis=1) <= 0),"\n-----\n")
    mu_matrix.append(y_hat)
    acc_i = measure_ACC(y_hat, yi) * xi.shape[0] ## un-normalize acc(i)
    acc = acc_i + acc
    Ns += xi.shape[0]
    print(f"\rAcc = {acc/Ns:.5f} ({int(Ns)})", end="") ## should construct the (smeared) identity matrix, exactly same as Y
print()
#acc = acc / Ns ## compute overall probe accuracy
mu_matrix = jnp.concat(mu_matrix, axis=0)
print(mu_matrix.shape)
print(vY.shape)

conf_matrix, precision, recall, misses, acc, adj_acc = analyze_scores(
    mu_matrix, vY, extract_label_indx=True
)
print("--- KNN-Probe Results --- ")
print(conf_matrix)
print(f"Acc = {acc:.5f} Acc.adj = {adj_acc:.5f} (Misses = {misses})")
print(f"Prec = {precision:.5f} Recall = {recall:.5f}")

## update trial-record with measurements
if config is not None:
    _trial = config.get(f"{seed}")
    if _trial is not None:
        model_dir = _trial.get("modelDir")
        jnp.save(f"{model_dir}test_confusion.npy", conf_matrix)
        _trial[f"test_knn_acc{metric_tag}"] = float(acc)
        _trial[f"test_knn_precision{metric_tag}"] = float(precision)
        _trial[f"test_knn_recall{metric_tag}"] = float(recall)
        config[f"{seed}"] = _trial

        ## save updated trial / config dictionary
        with open(config_fname, "w", encoding="utf-8") as out_fd:
            json.dump(config, out_fd, indent=4)

