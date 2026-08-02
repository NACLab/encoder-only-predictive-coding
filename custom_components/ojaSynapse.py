from jax import random, numpy as jnp, jit

from ngclearn import compilable  # from ngcsimlib.parser import compilable
from ngclearn import Compartment  # from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils.model_utils import normalize_block_matrix


class OjaSynapse(DenseSynapse):

    def __init__(
            self,
            name,
            shape,
            eta=0.,
            stability_factor=1.,  ## if <= 0, will disable Oja's correction term
            block_shape=None,
            use_block_norm=True,
            normalize=False,
            norm_axis=1,
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
        self.stability_factor = stability_factor
        self.sign_value = sign_value

        self.normalize = normalize
        self.norm_axis = norm_axis #1
        self.block_shape = None
        self.use_block_norm = use_block_norm  # True ## apply norm per block?  #False
        if block_shape is not None:
            self.block_shape = tuple(block_shape)  ## has to be a tuple to work with jit
        else:
            self.use_block_norm = False

        ## Set up compartments (state of the cell, parameters, will be updated through stateless calls)
        ## NOTE: apply mask immediately at initialization
        self.weights.set(self.weights.get() * self.mask.get())

        self.norm_trigger = Compartment(jnp.zeros((1, 1)))
        if self.normalize:
            ## pre-normalize synapses if norm-constraints are to be used
            self.norm_trigger.set(self.norm_trigger.get() + 1)
            weights = self.weights.get()
            weights = OjaSynapse._normalize(
                weights, self.norm_axis, self.use_block_norm, self.block_shape
            )
            weights = weights * self.mask.get()
            self.weights.set(weights)

        self.initial_weights = Compartment(self.weights.get())  ## store synaptic initial conditions (never updated)
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre = Compartment(preVals)
        self.post = Compartment(postVals)
        self.dWeights = Compartment(jnp.zeros(shape))
        self.dBiases = Compartment(jnp.zeros(shape[1]))

    @compilable
    def advance_state(self):
        weights = self.weights.get()  # * self.mask.get()
        self.outputs.set((jnp.matmul(self.inputs.get(), weights) * self.Rscale) + self.biases.get())

    @staticmethod
    def _normalize(weights, norm_axis, use_block_norm, block_shape, norm_target=1., norm_order=2):
        if use_block_norm:
            _weights = normalize_block_matrix(
                weights, block_shape, order=norm_order, axis=norm_axis, norm_targ=norm_target
            )
        else:
            _weights = weights * (
                    norm_target / (jnp.linalg.norm(weights, ord=norm_order, axis=norm_axis, keepdims=True) + 1e-8)
            )
        return _weights

    @compilable
    def evolve(self, dt):
        # Get the variables
        pre = self.pre.get()
        post = self.post.get()
        weights = self.weights.get()
        # biases = self.biases.get()
        key, *subkeys = random.split(self.key.get(), 3)  ## generate keys for noise samples
        self.key.set(key)

        B = pre.shape[0]  ## get batch-length
        ################################################################################################################
        ## Perform a single step of Oja's rule
        dWeights_hebb = pre.T @ post  ## classical Hebbian term
        dWeights_oja = 0.
        if self.stability_factor > 0.:
            dWeights_oja = -(post[:, None] ** 2 * weights)  ## Oja-correction / stability term
            dWeights_oja = jnp.sum(dWeights_oja, axis=0)  ## calculate full (tensor) Oja's correction
            # dWeights_oja = dWeights_oja * stability_factor
        dWeights = dWeights_hebb + dWeights_oja  ## calculate full (tensor) update
        # dWeights = dWeights * self.mask.get() * self.eta * (1. / B)
        dWeights = dWeights * self.eta * (1. / B)

        weights = weights + dWeights * self.sign_value  ## do Hebbian ascent

        ## Apply a normalization step
        if self.norm_trigger.get() > 0.:
            weights = OjaSynapse._normalize(
                weights, self.norm_axis, self.use_block_norm, self.block_shape
            )
        ## Apply/enforce masked sparsity pattern
        weights = weights * self.mask.get()

        # Update compartments
        self.weights.set(weights)
        self.dWeights.set(dWeights)

    @compilable
    def reset(self):  ## reset core components/statistics
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)  # outputs
        self.pre.set(preVals)  # pre
        self.post.set(postVals)  # post
        self.dWeights.set(jnp.zeros(self.shape))  # dW
        self.dBiases.set(jnp.zeros(self.shape[1]))  # db
