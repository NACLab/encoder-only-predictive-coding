# Encoder-only Neuroscience-informed Self-Supervised Learning: Meta-Representational Predictive Coding

This is the supporting code for the paper 
<a href="https://arxiv.org/html/2503.21796v2">
"Meta-Representational Predictive Coding: Neuroscience-Informed Self-Supervised Learning"
</a> (Ororbia, Friston, Rao, 2025). <br>
Note that this code was written on/run on an Ubuntu 22.04.2 LTS and 
assumes that you have Python 3.10.12, jax/jaxlib 0.4.28 (for Cuda-12), and 
ngclearn 3.1.0 (with ngcsimlib 3.1.0) successfully installed on your system.

## Installation

If you have Python 3.10.12 installed, you can automatically configure the needed dependencies 
via pip. It is recommended that you first create a separate Python virtual environment (VM) to 
act as a playground for this code (and protect your working environment) like so:

```consolve
python3.10 -m venv env_mpc  ## create a playground VM
source env_mpc/bin/activate  ## activate/enter the playground VM
```

You can then install the required libraries/modules in your Python VM via the pip package installer like so:

```console
pip install -r requirements.txt  ## install required libraries in your playground VM
```

<i>Note:</i> Running the above pip command will ensure that you have the GPU-enabled variants of 
JAX and NGC-Sim-Lib/NGC-Learn.

## Running the Model Simulation 

In order to run the simulation, make sure you unzip the mnist data prepared for you in 
the `/data/` folder (unzip `/data/mnist.zip` and place it inside of `/data/`).
To train an MPC circuit model on MNIST, run the following prepared BASH script:
```console
./exec_mnist.sh
```

To train an MPC circuit model on small NORB  (unzip `/data/norb.zip` and place it inside of `/data/`)., run the following 
prepared BASH script:
```console
./exec_norb.sh
```

<i>Note</i>: To use (small) Norb, you either want to re-use/re-purpose your current arrays or use the optional 
fetching script provided in `data/`; run the script like so:
```console
$ cd data/
$ python fetch_norb.py
```

This will download and process the small NORB arrays for you, creating the expected `trainX.npy`, `trainY.npy`, 
`testX.npy`, and `testY.npy` arrays in the sub-folder `data/norb/`.

The above training scripts will simulate an MPC circuit model on the MNIST/NORB databases for you (on the 
GPU with identifier 0; if you want to run a different GPU, choose another appropriate integer identifier within the script).
Furthermore, the script will generate the model structure (in ngc-learn JSON format) as well as
store NPZ files containing your best found parameters during training. 
All of this will be stored, if you run the script in its default mode (i.e., w/o modifying
its arguments) to a folder `exp_out/mpc/` which contains your saved
ngc-learn MPC model. Note that your experimental trial metrics/data will be stored within the appropriate 
JSON config file in `configs/`, i.e., NORB information will be stored in `configs/norb_mpc.json` and 
MNIST information will be recorded within `configs/mnist_mpc.json`.

<i>Note:</i> You can safely ignore the warnings collected in auto-generated `logging/` directoy as these 
are simply where ngc-learn/sim-lib store library messages.

## Model Evaluation / Analysis

Running the above scripts will further, after model training simulation is completed, run a few additional 
analysis scripts; notably, the above will collect/extract latent codes from your trained MPC models and 
then run a KNN downstream analysis probe to obtain your test generalization ability. 

If you want a t-SNE visualization of your model's collected latent codes, after the above training script(s) finish, you 
can directly run the following BASH script to produce a latent code visualization:
```console
./run_tsne.sh 0 mnist mpc 55 ## for trial (seed) 55, run t-SNE, on the MNIST mpc model
```

which will create a sub-directory called `exp_out/mpc/mnist/55/tsne/`. 
It is in here that you will find a t-SNE plot of your model's extracted latent codes (as well as a numpy array 
containing the tSNE embedding codes). 
<i>Note</i>: for NORB, if you want to visualize MPC's latent codes, create a sub-set of the test-set (with 2000 random samples 
per class) and use this as the test set for the t-SNE script to visualize to avoid memory issues (since NORB's full 
test-set is quite large).

## Citation / Reference

If you use this code or model mathematics in any form in your project(s), please cite its source
article:
<pre>
@article{ororbia2025meta,
  title={Meta-representational predictive coding: biomimetic self-supervised learning},
  author={Ororbia, Alexander and Friston, Karl and Rao, Rajesh PN},
  journal={arXiv preprint arXiv:2503.21796},
  year={2025}
}
</pre>

