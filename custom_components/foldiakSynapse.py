from jax import random, numpy as jnp, jit

from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.synapses import DenseSynapse


class FoldiakSynapse(DenseSynapse):

    def __init__(
            self,
            name,
            shape,
            eta=0.,
            rho=0.,  ## NOTE: 0 triggers vanilla anti-Hebbian learning
            exc=0.,
            inh=0.05,
            inh_bound=-1.,
            threshold_scale=1.,
            weight_init=None,
            bias_init=None,
            sign_value=1.,
            p_conn=1.,
            resist_scale=1.,
            batch_size=1,
            **kwargs
    ):
        super().__init__(
            name,
            shape=shape,
            weight_init=weight_init,
            bias_init=bias_init,
            resist_scale=resist_scale,
            p_conn=p_conn,
            batch_size=batch_size, **kwargs
        )

        ## synaptic plasticity properties and characteristics
        self.shape = shape
        self.Rscale = resist_scale
        self.eta = eta
        self.sign_value = sign_value
        self.rho = rho
        self.exc = exc
        self.inh = inh
        self.inh_bound = inh_bound
        self.inh_scale = -1.  ## < 0 to ensure inhibitory pressure
        self.threshold_scale = threshold_scale  ## <= 0 disables threshold bias
        self.make_inputs_nonnegative = True ## NOTE: ensures input activites are positive-only

        ## exc gets flipped to negative to ensure subtraction yields
        ## self-excitation along main diagonal
        I = jnp.eye(N=shape[0], M=shape[1])
        self.exc_weights = I * -exc
        self.weights.set(self.weights.get() * (1. - I) * inh)  # + I * -exc )

        ## Set up compartments (state of the cell, parameters, will be updated through stateless calls)
        ## NOTE: apply mask immediately at initialization
        self.weights.set(self.weights.get() * self.mask.get())

        self.initial_weights = Compartment(self.weights.get())  ## store synaptic initial conditions (never updated)
        if self.rho > 0. and self.threshold_scale > 0.:
            self.biases.set(jnp.zeros((1, shape[1])))  # + 0.5)
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre = Compartment(preVals)
        self.post = Compartment(postVals)
        self.dWeights = Compartment(jnp.zeros(shape))
        self.dBiases = Compartment(jnp.zeros(shape[1]))

    @compilable
    def advance_state(self):
        inputs = self.inputs.get()
        if self.make_inputs_nonnegative:
            inputs = jnp.abs(inputs)
        weights = self.weights.get() + self.exc_weights
        thresholds = self.biases.get()  ## subtractive "biases"
        # weights = weights * self.mask.get()
        inh_scale = self.inh_scale
        ## NOTE: the output of a Foldiak synaptic projection is negative (to serve as inhibitory pressure)
        self.outputs.set(
            (jnp.matmul(inputs, weights) * self.Rscale) * inh_scale + thresholds * self.threshold_scale * inh_scale
        )

    @compilable
    def evolve(self, dt):
        # Get the variables
        pre = self.pre.get() ## NOTE: generally this is also "self.inputs.get()"
        if self.make_inputs_nonnegative:
            pre = jnp.abs(pre)
        post = self.post.get()
        weights = self.weights.get()
        thresholds = biases = self.biases.get()
        # opt_params = self.opt_params.get()
        key, *subkeys = random.split(self.key.get(), 3)  ## generate keys for noise samples
        self.key.set(key)

        if self.eta > 0.:
            I = jnp.eye(weights.shape[0])  ## obtain this component's identity matrix
            # exc_vals = weights * I ## extract self-excitation values

            B = pre.shape[0]  ## get batch-length
            ################################################################################################################
            ## core synaptic update model
            dThresholds = biases * 0
            if self.rho > 0.:
                dWeights = jnp.matmul(pre.T, post) - (self.rho ** 2)
                dWeights = dWeights * self.eta * (1. / B)
                weights = weights + dWeights * self.sign_value
                weights = jnp.maximum(0., weights * (1.0 - I))

                # if self.inh_bound > 0.:
                #    weights = jnp.clip(weights, 0., self.inh_bound)

                if self.threshold_scale > 0.:  # self.bias_init is not None
                    dThresholds = jnp.sum((post - self.rho), axis=0, keepdims=True) * self.eta * (1. / B)
                    thresholds = thresholds + dThresholds * self.sign_value * self.threshold_scale
                    ## TODO: clip thresholds correctly to scale?
            else:
                dWeights = jnp.matmul(post.T, post)
                dWeights = dWeights - jnp.diag(jnp.diag(dWeights))
                dWeights = dWeights * self.mask.get() * self.eta * (1. / B)

                weights = weights + dWeights
                weights = jnp.maximum(0., (weights + weights.T) / 2.)  # * self.mask.get()

            if self.inh_bound > 0.:
                weights = jnp.clip(weights, 0., self.inh_bound)
                thresholds = jnp.clip(thresholds, 0., self.inh_bound)
            ################################################################################################################

            # weights = weights + exc_vals ## put back in self-excitation values
            ## Apply/enforce masked sparsity pattern
            weights = weights * self.mask.get()

            self.dWeights.set(dWeights)
            self.dBiases.set(dThresholds)

        # Update compartments
        self.weights.set(weights)
        self.biases.set(thresholds)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)  # outputs
        self.pre.set(preVals)  # pre
        self.post.set(postVals)  # post
        self.dWeights.set(jnp.zeros(self.shape))  # dW
        self.dBiases.set(jnp.zeros((1, self.shape[1])))  # db / dThresholds
