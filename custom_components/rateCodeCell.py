from jax import numpy as jnp, random, jit

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.model_utils import create_function, lkwta, d_lkwta
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2, step_rk4

def create_layer_group_masks(n_groups, z_l_dim):
    _masks = []
    for g in range(n_groups):
        row = []
        for gg in range(n_groups):
            if gg == g:
                row.append(jnp.ones((1, z_l_dim)))
            else:
                row.append(jnp.zeros((1, z_l_dim)))
        _masks.append(jnp.concatenate(row, axis=1))
    masks = jnp.concatenate(_masks, axis=0)
    return _masks, masks

def _dfz_internal_laplace(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = jnp.sign(z) ## d/dx of Laplace is signum
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_cauchy(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = (z * 2)/(1. + jnp.square(z))
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_exp(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = jnp.exp(-jnp.square(z)) * z * 2
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_gaussian(z, j, j_td, tau_m, leak_gamma): ## raw dynamics
    z_leak = z # * 2 ## Default: assume Gaussian
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

# @partial(jit, static_argnames=["integType", "priorType"])
def _run_cell(dt, j, j_td, z, tau_m, leak_gamma=0., integType=0, priorType=0):
    _dfz_fns = {
        0: lambda t, z, params: _dfz_internal_gaussian(z, *params),
        1: lambda t, z, params: _dfz_internal_laplace(z, *params),
        2: lambda t, z, params: _dfz_internal_cauchy(z, *params),
        3: lambda t, z, params: _dfz_internal_exp(z, *params),
    }
    _dfz_fn = _dfz_fns.get(priorType, _dfz_internal_gaussian)
    _step_fns = {
        0: step_euler,
        1: step_rk2,
        2: step_rk4,
    }
    _step_fn = _step_fns.get(integType, step_euler)
    params = (j, j_td, tau_m, leak_gamma)
    _, _z = _step_fn(0., z, _dfz_fn, dt, params)
    return _z

class RateCodeCell(JaxComponent): ## Rate-coded/real-valued cell

    def __init__(
            self,
            name,
            n_units,
            tau_m,
            prior=("gaussian", 0.),
            activation_model=("identity", 1, 1), ## (act_fx string, nWinners, n_groups)
            use_deriv_fx=True,
            integration_type="euler",
            batch_size=1,
            resist_scale=1.,
            shape=None,
            **kwargs):
        super().__init__(name, **kwargs)

        ## Set up layer size properties
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size

        ## Set up membrane parameters (affects ODE integration)
        self.use_deriv_fx = use_deriv_fx
        self.tau_m = tau_m ## membrane time constant -- setting to 0 triggers "stateless" mode
        priorType, leakRate = prior
        priorTypeDict = {
            "gaussian": 0,
            "laplacian": 1,
            "cauchy": 2,
            "exp": 3
        }
        self.priorType = priorTypeDict.get(priorType, 0)
        self.priorLeakRate = leakRate ## degree to which rate neurons leak (according to prior)
        self.resist_scale = resist_scale ## a "resistance" scaling factor
        ## integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Set up activation function model
        self.fx, self.dfx = create_function(fun_name="identity")
        self.gamma = 0.
        self.n_groups = 1
        self.nWTA = 1 #()
        self.neuron_masks = None
        act_fx, *_ = activation_model
        self.act_fx = act_fx
        self.act_fx_choice = 0 ## default is standard activation
        if "lkwta" in act_fx:
            self.act_fx_choice = 1
            act_fx, nWinners, n_groups = activation_model
            self.n_groups = n_groups

            self.nWTA = nWinners #tuple(nWinners for k in range(self.n_groups))
            #self.neuron_masks, _ = create_layer_group_masks(self.n_groups, self.n_units // self.n_groups)
            _, self.neuron_masks = create_layer_group_masks(self.n_groups, self.n_units // self.n_groups)
            #self.fx, self.dfx = lkwta, d_lkwta
        else: ## act_fx_choice == 0
            self.fx, self.dfx = create_function(fun_name=act_fx)

        ## Set up compartments (state of the cell & parameters will be updated through stateless calls)
        restVals = jnp.zeros(_shape)
        self.j = Compartment(
            restVals, display_name="Input Stimulus Current", units="mA"
        ) # electrical current
        self.j_scaled = Compartment(restVals) ## temporary modified current variable
        self.zF = Compartment(
            restVals, display_name="Transformed Rate Activity"
        ) # rate-coded output - activity
        self.j_td = Compartment(
            restVals, display_name="Modulatory Stimulus Current", units="mA"
        ) # top-down electrical current - pressure
        self.z = Compartment(
            restVals, display_name="Rate Activity", units="mA"
        ) # rate activity
        ## historical variables (of t-dt)
        self.zF_prior = Compartment(
            restVals, display_name="Prior Transformed Rate Activity (t-dt)"
        )

    @compilable
    def advance_state(self, dt):
        # Get current value of compartments
        j = self.j.get()
        j_td = self.j_td.get()
        z = self.z.get()
        zF = self.zF.get() ## current zF(t)
        self.zF_prior.set(zF) ## store in zF(t-dt)

        if self.tau_m > 0.:
            ## TODO: maybe use Gaussian-decay/prior + langevin here too?
            dfx_val = 1.
            if self.use_deriv_fx:
                if self.act_fx_choice == 1:
                    dfx_val = d_lkwta(z, self.neuron_masks, self.nWTA)
                else:
                    dfx_val = self.dfx(z)
            ## else: ignore derivative of activation w.r.t. dynamics z
            j = (j * self.resist_scale) * dfx_val
            ### run a step of integration over neuronal dynamics
            tmp_z = _run_cell(
                dt, j, j_td, z,
                tau_m=self.tau_m,
                leak_gamma=self.priorLeakRate,
                integType=self.intgFlag,
                priorType=self.priorType
            )
            z = tmp_z ## pre-activation function value(s)
        else:
            ## run in "stateless" mode (when no membrane time constant provided)
            z = j + j_td ## total sum

        ## Set post-activation function value(s)
        if self.act_fx_choice == 1:
            zF = lkwta(z, self.neuron_masks, self.nWTA)
        else:
            zF = self.fx(z)

        ## TODO: try max-pooling?
        # ## compute a filtered version of zF
        # eps = 1e-6
        # z_filter = jnp.sqrt((zF ** 2) + eps)
        # self.z_filter.set(z_filter)

        # Update compartments (move t to t+dt)
        #self.j.set(j)
        self.j_scaled.set(j)
        #self.j_td.set(j_td)
        self.z.set(z)
        self.zF.set(zF)

    @compilable
    def reset(self):  # , batch_size, shape): #n_units
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)
        if not self.j.targeted:
            self.j.set(restVals)
        self.j_scaled.set(restVals)
        if not self.j_td.targeted:
            self.j_td.set(restVals)
        self.z.set(restVals)
        self.zF.set(restVals)
        self.zF_prior.set(restVals)
