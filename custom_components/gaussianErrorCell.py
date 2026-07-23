from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

class GaussianErrorCell(JaxComponent):

    def __init__(self, name, n_units, batch_size=1, sigma=1., shape=None, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        sigma_shape = (1,1)
        if not isinstance(sigma, float) and not isinstance(sigma, int):
            sigma_shape = jnp.array(sigma).shape
        self.sigma_shape = sigma_shape
        self.shape = shape
        self.batch_size = batch_size
        self.n_units = n_units

        ## Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.L = Compartment(0., display_name="Gaussian Log likelihood", units="nats") # loss compartment
        self.L_batch = Compartment(jnp.zeros((self.batch_size, 1)))
        self.mu = Compartment(restVals, display_name="Gaussian mean") # mean/mean name. input wire
        self.dmu = Compartment(restVals) # derivative mean
        _Sigma = jnp.zeros(sigma_shape)
        self.Sigma = Compartment(_Sigma + sigma, display_name="Gaussian variance/covariance")
        self.dSigma = Compartment(_Sigma)
        self.target = Compartment(restVals, display_name="Gaussian data/target variable") # target. input wire
        self.dtarget = Compartment(restVals) # derivative target
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed
        self.mask = Compartment(restVals + 1.0)

    @staticmethod
    def _eval_log_density(target, mu, Sigma): ## Gaussian log likelihood
        ## NOTE: ln(p) = -(x - mu)^2 * 1/(2 Sigma), where Sigma might be sigma^2 or covariance matrix
        _dmu = (target - mu)
        #log_density = -jnp.sum(jnp.square(_dmu)) * (1./((Sigma ** 2) * 2))
        batch_log_density = -jnp.sum(jnp.square(_dmu), axis=1, keepdims=True) * (1./((Sigma ** 2) * 2))
        log_density = jnp.sum(batch_log_density)
        return log_density, batch_log_density, _dmu ## return density and raw delta

    @compilable
    def advance_state(self, dt): ## compute Gaussian error cell output (fixed-point)
        # Get the variables
        mu = self.mu.get()
        target = self.target.get()
        Sigma = self.Sigma.get()
        modulator = self.modulator.get()
        mask = self.mask.get()

        # Moves Gaussian cell dynamics one step forward. Specifically, this routine emulates the error unit
        # behavior of the local cost functional:
        L, L_batch, _dmu = GaussianErrorCell._eval_log_density(target, mu, Sigma) # L = -jnp.sum(jnp.square(_dmu)) * (0.5 / Sigma)
        ## _dmu => "raw" e (error unit/mis-match) # _dmu = (target - mu)
        dmu = _dmu * (1./ Sigma)  ## obtain precision-scaled e: (target - mu)/Sigma
        dtarget = -dmu  # reverse of e ## -(target - mu)/Sigma
        dSigma = Sigma * 0 + 1.  # no derivative is calculated at this time for Sigma

        dmu = dmu * modulator * mask ## not sure how mask will apply to a full covariance...
        dtarget = dtarget * modulator * mask
        mask = mask * 0. + 1. ## "eat" the mask as it should only apply at time t

        # Update compartments
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.dSigma.set(dSigma)
        self.L.set(jnp.squeeze(L))
        self.L_batch.set(L_batch)
        self.mask.set(mask)

    @compilable
    def reset(self):  ## reset core components/statistics
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)
        dmu = restVals
        dtarget = restVals
        dSigma = jnp.zeros(self.sigma_shape)
        target = restVals
        mu = restVals
        modulator = mu + 1.
        L = 0.  # jnp.zeros((1, 1))
        mask = jnp.ones(_shape)

        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.dSigma.set(dSigma)
        if not self.target.targeted:
            self.target.set(target)
        if not self.mu.targeted:
            self.mu.set(mu)
        self.modulator.set(modulator)
        self.L.set(L)
        self.mask.set(mask)
