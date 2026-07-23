import numpy as np
from jax import numpy as jnp, random, jit
from .agent import Agent

from ngclearn import Context, MethodProcess
from ngcsimlib.global_state import stateManager
from ngclearn.utils.model_utils import create_block_matrix as create_group_matrix
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from custom_components import RateCodeCell as RateCell, OjaSynapse, FoldiakSynapse, GaussianErrorCell as ErrorCell
from ngclearn.components import StaticSynapse
from ngcsimlib.operations import Summation
from ngclearn.utils.model_utils import eye_wrapped


class MPCEncoder(Agent): ## model-builder for proposed MPC (this sub-classes the "Agent" construct)
    """
    This is the full MPC model structure/framework and includes both intra- and inter-stream elements.
    Note: this MPC-encoder uses factorized error neurons for intra and inter stream prediction synapses (resulting
    in two block-mask matrices for intra- and inter-stream pathways.
    """

    def __init__(
            self,
            dkey,
            model_config=None,
            saveDir=None,
            trial_id=None,
            **kwargs
    ):
        super().__init__(dkey, saveDir=saveDir, **kwargs)
        self.dkey, *subkeys = random.split(self.dkey, 5)

        ## controls influence of more distal streams (controls how much inter-stream prediction shapes MPC representations)
        stream_field_sigma = float(model_config.get("hyperParameters").get("stream_field_sigma"))

        self.use_action_syn = True ## use action-conditional syn?
        self.use_cross_stream_syn = False  ## use cross-stream syn?
        self.inter_predict_innovation = True ## should inter-stream synapses predict cross-stream innovations?
        if stream_field_sigma > 0.:
            print(f" >> Using weighted ({stream_field_sigma:0.4f}) inter-stream predictors!")
            self.use_cross_stream_syn = True

        self.n_layers = 3 ## sensory + latent layers
        self.param_names = ["W1", "W2", "V1", "V2", "R1_intra", "R2_intra"]
        if self.use_action_syn:
            self.param_names = self.param_names + ["A1", "A2"]
        if self.use_cross_stream_syn:
            self.param_names = self.param_names + ["R1_inter", "R2_inter"]

        self.normalize = normalize = bool(model_config.get("hyperParameters").get("normalize"))
        self.Na = Na = int(model_config.get("hyperParameters").get("Na"))  ## number action dims
        # self.Nz0 = Nz0 = int(model_config.get("hyperParameters").get("Nz0"))
        foveal_shape = tuple(model_config.get("hyperParameters").get("foveal_shape"))
        self.Nz0 = Nz0 = foveal_shape[0] * foveal_shape[1]
        self.Nz1 = Nz1 = int(model_config.get("hyperParameters").get("Nz1"))
        self.Nz2 = Nz2 = int(model_config.get("hyperParameters").get("Nz2"))
        if model_config.get("model_masks") is not None:
            _m1 = model_config.get("model_masks").get("mask1_fname")
            stream_mask1 = jnp.array(np.loadtxt(_m1, delimiter=","))
            _m2 = model_config.get("model_masks").get("mask2_fname")
            stream_mask2 = jnp.array(np.loadtxt(_m2, delimiter=","))

            self.n_streams0 = n_streams0 = stream_mask1.shape[0]
            self.n_streams1 = n_streams1 = stream_mask1.shape[1]
            self.n_streams2 = n_streams2 = stream_mask2.shape[1]
        else:
            n_streams0 = int(model_config.get("hyperParameters").get("n_streams0"))
            n_streams1 = int(model_config.get("hyperParameters").get("n_streams1"))
            n_streams2 = int(model_config.get("hyperParameters").get("n_streams2"))
            stream_mask1 = jnp.eye(N=n_streams0, M=n_streams1)
            stream_mask2 = jnp.eye(N=n_streams1, M=n_streams2)

            self.n_streams0 = n_streams0
            self.n_streams1 = n_streams1
            self.n_streams2 = n_streams2

        self.batch_size = batch_size = int(model_config.get("hyperParameters").get("batch_size"))
        self.T = T = int(model_config.get("hyperParameters").get("T"))
        self.dt = 1.  ## integration time-constant (ms)
        tau_m = float(model_config.get("hyperParameters").get("tau_m"))
        w_scale = 0.1  ## NOTE: make this configurable?
        eta_intra = float(model_config.get("hyperParameters").get("eta_intra"))
        eta_inter = float(model_config.get("hyperParameters").get("eta_inter"))
        eta_lat = float(model_config.get("hyperParameters").get("eta_lat"))
        eta_div = float(model_config.get("hyperParameters").get("eta_div"))
        z_decay = float(model_config.get("hyperParameters").get("z_decay"))  # 0.0001 #0.001 #0.0005 #0.  # 0.0001
        oja_factor = float(model_config.get("hyperParameters").get("oja_factor"))
        act_fx_model = str(model_config.get("hyperParameters").get("act_fx_model"))
        rho_base = float(model_config.get("hyperParameters").get("rho_base"))  # 0.05
        inh_base = float(model_config.get("hyperParameters").get("inh_base"))
        exc_base = float(model_config.get("hyperParameters").get("exc_base"))
        inh_1 = inh_base
        exc_1 = exc_base
        rho_1 = rho_base
        inh1_bound = float(jnp.maximum(inh_base, exc_base))
        inh_scale = float(model_config.get("hyperParameters").get("inh_scale"))  # 1. #15.
        inh_2 = inh_base * inh_scale
        exc_2 = exc_base * inh_scale
        rho_2 = rho_base * inh_scale
        inh2_bound = float(jnp.maximum(inh_2, exc_2))
        threshold_scale = float(model_config.get("hyperParameters").get("threshold_scale"))
        ## random distillation target factor: < 0 disables this
        self.z_gamma = float(model_config.get("hyperParameters").get("z_gamma"))

        n_z0 = Nz0 * n_streams0  ## full dimensionality of input
        n_z1 = Nz1 * n_streams1
        n_z2 = Nz2 * n_streams2
        self.blk_shape_W1 = blk_shape_W1 = (n_z0 // self.n_streams0, n_z1 // self.n_streams1)
        self.blk_shape_W2 = blk_shape_W2 = (n_z1 // self.n_streams1, n_z2 // self.n_streams2)
        self.blk_shape_R1 = blk_shape_R1 = (n_z1 // self.n_streams1, n_z1 // self.n_streams1)
        self.blk_shape_R2 = blk_shape_R2 = (n_z2 // self.n_streams2, n_z2 // self.n_streams2)

        if "lkwta" in act_fx_model:
            nWTA1 = 5  # 4
            nWTA2 = 7
            act1_model = ("lkwta", nWTA1, n_streams1)
            act2_model = ("lkwta", nWTA2, n_streams2)
        else:
            act1_model = act2_model = ("relu", 1, 1)

        if trial_id is not None:
            self._load_from_disk(model_config.get(trial_id).get("modelDir"))
        else:
            ensemble_scale = 1.  # 1./2. #1. # 1./self.n_streams
            with Context("model") as self.model:  ## set up actual model
                ## Define sensory layer
                a = RateCell(  ## action input
                    name="a",
                    tau_m=0.,
                    n_units=Na,
                    batch_size=batch_size
                )
                z0 = RateCell(  ## z0
                    name="z0",
                    tau_m=0.,
                    n_units=Nz0 * n_streams1,
                    batch_size=batch_size
                )

                ## build motif for layer z1
                hyperparams1 = (
                    n_streams1, Nz0, Nz1, Na, tau_m, z_decay, w_scale, inh_1, exc_1, rho_1, inh1_bound, eta_intra,
                    eta_inter, batch_size, eta_lat, oja_factor, ensemble_scale, act1_model, threshold_scale,
                    normalize, stream_field_sigma
                )
                projection1, dynamics1 = build_modif(
                    subkeys[0],
                    l=1,
                    z_lm1=z0,
                    q_lm1=z0,
                    stream_mask=stream_mask1,
                    block_shapes=(blk_shape_W1, blk_shape_R1),
                    hyperparams=hyperparams1
                )
                q1, Q1, t1 = projection1
                z1, e1, e1_intra, e1_inter, W1, E1, R1_intra, ER1_intra, R1_inter, ER1_inter, V1, A1 = dynamics1

                ## build motif for layer z2
                hyperparams2 = (
                    n_streams2, Nz1, Nz2, Na, tau_m, z_decay, w_scale, inh_2, exc_2, rho_2, inh2_bound,
                    eta_intra / eta_div, eta_inter / eta_div, batch_size, eta_lat / eta_div, oja_factor,
                    ensemble_scale, act2_model, threshold_scale, normalize, stream_field_sigma
                )
                projection2, dynamics2 = build_modif(
                    subkeys[1],
                    l=2,
                    z_lm1=z1,
                    q_lm1=q1,
                    block_shapes=(blk_shape_W2, blk_shape_R2),
                    stream_mask=stream_mask2,
                    hyperparams=hyperparams2
                )
                q2, Q2, t2 = projection2
                z2, e2, e2_intra, e2_inter, W2, E2, R2_intra, ER2_intra, R2_inter, ER2_inter, V2, A2 = dynamics2

                ## globally wire-up the layers
                projection_motifs = [projection1, projection2]
                dynamics_motifs = [dynamics1, dynamics2]
                wire_motifs(
                    z0,
                    a,
                    projection_motifs,
                    dynamics_motifs,
                    z_gamma=self.z_gamma,
                    use_action_syn=self.use_action_syn,
                    use_cross_stream_syn=self.use_cross_stream_syn,
                    inter_predict_innovation=self.inter_predict_innovation
                )
                ## set up command execution path
                evolve_chain = [W1, R1_intra, V1, W2, R2_intra, V2] #, CR1, CR2, A1, A2]
                infer_chain = [z0, Q1, q1, Q2, q2]
                if self.use_cross_stream_syn:
                    advance_chain = [
                        V1, ER1_intra, ER1_inter, V2, E2, ER2_intra, ER2_inter,
                        z1, W1, e1, ## intra-encoder
                        R1_intra, e1_intra, ## intra-prediction
                        R1_inter, e1_inter, ## inter-prediction
                        z2, W2, e2,  ## intra-encoder
                        R2_intra, e2_intra, ## intra-prediction
                        R2_inter, e2_inter, ## inter-prediction
                    ]
                else:
                    advance_chain = [
                        V1, ER1_intra, V2, E2, ER2_intra,
                        z1, W1, e1, ## intra-encoder
                        R1_intra, e1_intra, ## intra-prediction
                        z2, W2, e2, ## intra-encoder
                        R2_intra, e2_intra, ## intra-prediction
                    ]
                reset_chain = [
                    z0, z1, z2, q1, q2,
                    e1, e2, e1_intra, e2_intra,
                    Q1, Q2, W1, W2, R1_intra, R2_intra, V1, V2,
                    E1, E2, ER1_intra, ER2_intra
                ]
                if self.use_action_syn:
                    evolve_chain = evolve_chain + [A1, A2]
                    infer_chain = [a, A1, A2] + infer_chain
                    reset_chain = reset_chain + [a, A1, A2]
                if self.use_cross_stream_syn:
                    evolve_chain = evolve_chain + [R1_inter, R2_inter]
                    reset_chain = reset_chain + [e1_inter, e2_inter, R1_inter, R2_inter, ER1_inter, ER2_inter]

                ## Set up command flow
                self.evolve = MethodProcess("evolve_proc")
                for node in evolve_chain:
                    self.evolve >> node.evolve
                self.infer = MethodProcess("infer_proc")
                for node in infer_chain:
                    self.infer >> node.advance_state
                self.advance = MethodProcess("advance_proc") #, use_jit=False)
                for node in advance_chain:
                    self.advance >> node.advance_state
                self.reset = MethodProcess("reset_proc")
                for node in reset_chain:
                    self.reset >> node.reset

        self.params = self.model.get_components(*self.param_names)

    def get_param_stats(self):
        out_str = ""
        for p in range(len(self.params)):
            param = self.params[p]
            name = param.name
            _param = param.weights.get()
            stats = f"{name}: [{jnp.amin(_param):.4f}, {jnp.amax(_param):.4f}] N: {jnp.linalg.norm(_param):.4f}"
            if p > 0:
                out_str = f"{out_str}\n{stats}"
            else:
                out_str = stats
        return out_str

    def process(self, x, action=None, adapt_synapses=False):
        self.reset.run()  # batch_size=x.shape[0])
        ## get input nodes
        a = self.model.get_components("a")
        z0 = self.model.get_components("z0")
        if action is not None:
            a.j.set(action)

        ## do necessary synapse/weight tying
        for l in range(1, self.n_layers):
            ## tie intra-/encoder synapse feedback to forward pathways
            E_l = self.model.get_components(f"E{l}")
            W_l = self.model.get_components(f"W{l}")
            E_l.weights.set(W_l.weights.get().T)
            ## tie intra-stream synapse feedback to forward pathways
            ER_l = self.model.get_components(f"ER{l}_intra")
            R_l = self.model.get_components(f"R{l}_intra")
            ER_l.weights.set(R_l.weights.get().T)
            ## tie inter-stream synapse feedback to forward pathways
            if self.use_cross_stream_syn:
                ER_l = self.model.get_components(f"ER{l}_inter")
                R_l = self.model.get_components(f"R{l}_inter")
                ER_l.weights.set(R_l.weights.get().T)

        ################################################################################################################
        ## do optional random target projection (distillation)
        rand_targs = []
        if 0. <= self.z_gamma < 1.:
            for l in range(1, self.n_layers):
                Q_l = self.model.get_components(f"Q{l}")
                W_l = self.model.get_components(f"W{l}")
                Q_l.weights.set(W_l.initial_weights.get())

            z0.j.set(x)
            self.infer.run(t=1. * self.dt, dt=self.dt)
            for l in range(1, self.n_layers):
                q_l = self.model.get_components(f"q{l}")
                rand_targs.append(q_l.z.get())
        else:
            for l in range(1, self.n_layers):
                q_l = self.model.get_components(f"q{l}")
                rand_targs.append(q_l.z.get())

        ################################################################################################################
        ## do P-step (projection to get initial conditions)
        for l in range(1, self.n_layers):
            Q_l = self.model.get_components(f"Q{l}")
            W_l = self.model.get_components(f"W{l}")
            Q_l.weights.set(W_l.weights.get())
        z0.j.set(x)
        self.infer.run(t=1. * self.dt, dt=self.dt)
        for l in range(1, self.n_layers):
            z_l = self.model.get_components(f"z{l}")
            q_l = self.model.get_components(f"q{l}")
            z_l.z.set(q_l.z.get())
            z_l.zF.set(q_l.zF.get())

        if self.z_gamma >= 0.:
            for l in range(1, self.n_layers):
                q_targ_l = self.model.get_components(f"q{l}")
                el_intra = self.model.get_components(f"e{l}_intra")
                targ_l = rand_targs[l] * (1. - self.z_gamma) + q_targ_l.z.get() * self.z_gamma
                el_intra.target.set(targ_l)
                if self.use_cross_stream_syn:
                    el_inter = self.model.get_components(f"e{l}_inter")
                    el_inter.target.set(targ_l)
        ################################################################################################################
        ## run iterative settling (E-steps) to get activities
        inputs = jnp.array(self.advance.pack_rows(self.T, t=lambda x: x, dt=self.dt))
        stateManager.state, outputs = self.advance.scan(inputs)  ## outputs[0]

        ################################################################################################################
        ## calculate approximate (free) energy
        energy = 0.
        energy_batch = 0.
        energy_internal = 0.
        for l in range(1, self.n_layers):
            e_l = self.model.get_components(f"e{l}")
            e_l_intra = self.model.get_components(f"e{l}_intra")
            energy_internal += e_l.L.get()
            energy += ( e_l.L.get() + e_l_intra.L.get() )
            energy_batch += ( e_l.L_batch.get() + e_l_intra.L_batch.get() )
            if self.use_cross_stream_syn:
                e_l_inter = self.model.get_components(f"e{l}_inter")
                energy += e_l_inter.L.get()
                energy_batch += e_l_inter.L_batch.get()

        z1 = self.model.get_components("z1").zF.get()
        W1 = self.model.get_components("W1").weights.get()

        ################################################################################################################
        if adapt_synapses: ## do M-step
            self.evolve.run(t=self.T, dt=self.dt)
        self.params = self.model.get_components(*self.param_names)

        z_codes = []
        for l in range(1, self.n_layers):
            z_codes.append(self.model.get_components(f"z{l}").zF.get())

        return energy_internal, energy, energy_batch, z_codes


    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:  ## this condition allows to only write actual parameter values w/in components to disk
            model_dir = "{}/{}/component/custom".format(self.saveDir, "model")
            self.params = self.model.get_components(*self.param_names)
            for param in self.params:
                param.save(model_dir)
            z1, z2, q1, q2 = self.model.get_components(
                "z1", "z2", "q1", "q2"
            )
            nodes = [z1, z2, q1, q2]
            for node in nodes:
                node.save(model_dir)

        else:  ## this saves the whole model form (JSON structure as well as parameter values)
            self.model.save_to_json(self.saveDir, model_name="model", overwrite=True)  ## save model state

    def _load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        # with Context("Circuit") as self.circuit:
        self.model = Context.load(directory=model_directory, module_name="model")
        # self.advance_proc = self.circuit.get_objects("advance_process", objectType="process")
        processes = self.model.get_objects_by_type("process")  ## obtain all saved processes within this context
        self.advance = processes.get("advance_proc")
        self.reset = processes.get("reset_proc")
        self.evolve = processes.get("evolve_proc")
        self.infer = processes.get("infer_proc")
        self.params = self.model.get_components(*self.param_names)

def build_modif(
        dkey,
        l,
        z_lm1: RateCell,
        q_lm1: RateCell,
        block_shapes,
        stream_mask,
        hyperparams
):  ## sub-circuit motif creation co-routine
    _, *subkeys = random.split(dkey, 5)
    (n_streams, Nz_lm1, Nz_l, Na, tau, z_decay, w_scale, inh_0, exc_0, rho,
     inh_bound, eta_intra, eta_inter, batch_size, eta_lat, oja_factor,
     ensemble_scale, act_model, threshold_scale, normalize, stream_field_sigma) = hyperparams
    block_shape_W, block_shape_R = block_shapes
    act_model_type, _, _ = act_model

    z_l_prior = ("gaussian", z_decay)
    if "lkwta" in act_model_type:
        z_l_prior = ("laplacian", z_decay)
    z_l = RateCell(  ## z_l
        name=f"z{l}",
        n_units=Nz_l * n_streams,
        tau_m=tau,
        activation_model=act_model,
        use_deriv_fx=True,
        prior=z_l_prior,
        batch_size=batch_size
    )
    sigma_z = 1.
    e_l = ErrorCell(  ## intra-stream encoder error units
        name=f"e{l}",  ## e1_intra
        n_units=z_l.n_units,
        sigma=sigma_z,  # (sigma_z ** 2),
        batch_size=batch_size
    )
    e_l_intra = ErrorCell(  ## intra-stream prediction error units
        name=f"e{l}_intra",
        n_units=z_l.n_units,
        sigma=sigma_z * (1. / ensemble_scale),  ## scale by inverse ensemble avg
        batch_size=batch_size
    )
    el_inter_scale = 1.
    if stream_field_sigma > 0.:
        el_inter_scale = (1. / stream_field_sigma)
    e_l_inter = ErrorCell(  ## inter-stream prediction error units
        name=f"e{l}_inter",
        n_units=z_l.n_units,
        sigma=sigma_z * el_inter_scale, # * (1. / ensemble_scale),  ## scale by inverse ensemble avg
        batch_size=batch_size
    )
    q_l = RateCell(  ## q_l
        name=f"q{l}",
        n_units=Nz_l * n_streams,
        tau_m=0.,
        activation_model=act_model,
        batch_size=batch_size
    )
    t_l = RateCell(  ## q_l
        name=f"tz{l}",
        n_units=Nz_l * n_streams,
        tau_m=0.,
        activation_model=act_model,
        batch_size=batch_size
    )
    ####################################################################################################################
    ## intra-stream prediction synapses setup
    W_l = OjaSynapse(  ## encoding synapses from layer l-1 to l
        name=f"W{l}",
        shape=(z_lm1.n_units, z_l.n_units),
        eta=eta_intra * batch_size,
        stability_factor=oja_factor,
        block_shape=block_shape_W,
        normalize=normalize,
        weight_init=dist.gaussian(mean=0., std=w_scale),
        mask=create_group_matrix(  ## mask
            stream_mask,
            group_shape=(Nz_lm1, Nz_l)
        ),
        key=subkeys[0],
        batch_size=batch_size
    )
    Q_l = StaticSynapse( ## initial conditions encoding synapses from l-1 to l
        name=f"Q{l}",
        shape=(z_lm1.n_units, z_l.n_units),
        mask=create_group_matrix(  ## mask
            stream_mask,
            group_shape=(Nz_lm1, Nz_l)
        ),
        key=subkeys[0],
        batch_size=batch_size
    )
    E_l = StaticSynapse( ## error synapses from l to l-1
        name=f"E{l}",
        shape=(W_l.shape[1], W_l.shape[0]),
        key=subkeys[0],
        mask=W_l.mask.get().T,
        batch_size=batch_size
    )

    bands = jnp.eye(N=n_streams, M=n_streams)
    T_M = create_group_matrix(  ## topology masking/weighting matrix
        bands,
        group_shape=(Nz_l, Nz_l)
    ) ## identity block-matrix means intra-stream predictions

    R_l_intra = OjaSynapse(  ## intra-stream prediction synapses from layer l to l
        name=f"R{l}_intra",
        shape=(z_l.n_units, z_l.n_units),
        eta=eta_inter * batch_size,
        stability_factor=oja_factor,
        use_block_norm=True,
        block_shape=block_shape_R,
        normalize=normalize,
        resist_scale=ensemble_scale,  ## R outputs should be ensemble average
        weight_init=dist.gaussian(mean=0., std=w_scale),
        mask=T_M,
        key=subkeys[1],
        batch_size=batch_size
    )
    ER_l_intra = StaticSynapse(
        name=f"ER{l}_intra",
        shape=(R_l_intra.shape[1], R_l_intra.shape[0]),
        key=subkeys[1],
        mask=R_l_intra.mask.get().T,
        batch_size=batch_size
    )
    ####################################################################################################################

    ####################################################################################################################
    ## Cross-stream synapses setup
    ## NOTE: el_inter will be weighted by `stream_field_sigma`
    _bands = bands
    C_T_M = jnp.ones((1,1))
    if stream_field_sigma > 0.0:
        ## calculate global weights applied to inter-stream synapses
        C_T_M = 0.
        k0 = 0  ## center-point of stream topology
        _accum = 0.0 ## accumulation stream weights should approach/asymptote to 1.0
        weight_base = 0.5
        for k in range(1, n_streams):  ## this will create a full topology for all K streams
            band_k = eye_wrapped(N=n_streams, k=k, values=jnp.ones((n_streams,)))
            # weight_k = jnp.exp(-((k - k0) ** 2) * (1. / (2 * sigSqr))) * 1./(jnp.sqrt(2. * jnp.pi * sigSqr))
            weight_k = jnp.power(weight_base, int(k - k0))
            _accum += weight_k
            cross_stream_mask_k = create_group_matrix(  ## mask
                band_k,
                group_shape=(Nz_l, Nz_l)
            ) * weight_k  ## band block-matrix means inter-stream predictions
            C_T_M = cross_stream_mask_k + C_T_M
            _bands += band_k * weight_k
        jnp.set_printoptions(precision=3, suppress=True)

    R_l_inter = OjaSynapse(  ## (inter)cross-stream prediction synapses from layer l to l
        name=f"R{l}_inter",
        shape=(z_l.n_units, z_l.n_units),
        eta=eta_inter * batch_size,
        stability_factor=oja_factor,
        use_block_norm=True, 
        block_shape=block_shape_R,
        normalize=normalize,
        resist_scale=ensemble_scale,  ## R outputs should be ensemble average
        weight_init=dist.gaussian(mean=0., std=w_scale),
        mask=C_T_M,
        key=subkeys[2],
        batch_size=batch_size
    )
    ER_l_inter = StaticSynapse(
        name=f"ER{l}_inter",
        shape=(R_l_inter.shape[1], R_l_inter.shape[0]),
        key=subkeys[1],
        mask=R_l_inter.mask.get().T,
        batch_size=batch_size
    )
    ####################################################################################################################

    if "lkwta" in act_model_type:
        V_l = FoldiakSynapse(  ## lateral synapses in layer 1
            name=f"V{l}",
            shape=(z_l.n_units, z_l.n_units),
            eta=0.,
            rho=0.,
            inh_bound=0.,
            exc=0.,
            inh=0.,
            threshold_scale=0.,
            weight_init=dist.constant(value=0.),
            mask=1.,
            key=subkeys[3],
            batch_size=batch_size
        )
    else:
        V_l = FoldiakSynapse(  ## lateral synapses in layer 1
            name=f"V{l}",
            shape=(z_l.n_units, z_l.n_units),
            eta=eta_lat * batch_size,
            rho=rho,
            inh_bound=inh_bound,
            exc=exc_0,
            inh=inh_0,
            threshold_scale=threshold_scale * inh_0,
            weight_init=dist.uniform(low=0., high=1.),
            mask=create_group_matrix(  ## mask
                jnp.eye(N=n_streams, M=n_streams),
                group_shape=(Nz_l, Nz_l)
            ),
            key=subkeys[3],
            batch_size=batch_size
        )

    A_l = OjaSynapse(  ## action-conditional prediction synapses from action to layer l
        name=f"A{l}",
        shape=(Na, z_l.n_units),
        eta=eta_intra * batch_size,
        stability_factor=oja_factor,
        normalize=normalize,
        use_block_norm=False,
        resist_scale=1.,
        weight_init=dist.gaussian(mean=0., std=w_scale),
        key=subkeys[3],
        batch_size=batch_size
    )

    ## just package nodes into lists for external wiring w/in model context
    projection_nodes = (q_l, Q_l, t_l)
    dynamics_nodes = (
        z_l, e_l, e_l_intra, e_l_inter, W_l, E_l, R_l_intra, ER_l_intra, R_l_inter, ER_l_inter, V_l, A_l
    )
    return projection_nodes, dynamics_nodes

def wire_motifs(
        z0,
        a,
        projection_motifs,
        dynamics_motifs,
        z_gamma=-1.,
        use_action_syn=False, ## use action-conditional syn?
        use_cross_stream_syn=False, ## use cross-stream syn?
        inter_predict_innovation=False
):
    ############################################################################
    qlm1 = zlm1 = z0  ## NOTE: seed nodes are same for projection & dynamics circuits
    for l in range(len(dynamics_motifs)):  ## go from lyr 0 to L-1
        ql, Ql, tl = projection_motifs[l]
        zl, el, el_intra, el_inter, Wl, El, Rl_intra, ERl_intra, Rl_inter, ERl_inter, Vl, Al = dynamics_motifs[l]

        ## Wire signal pathways

        ### infernece/projection step
        qlm1.zF >> Ql.inputs
        if use_action_syn:
            a.zF >> Al.inputs
            Summation(Ql.outputs, Al.outputs) >> ql.j
        else:
            Ql.outputs >> ql.j

        ### dynamics/E-step
        zlm1.zF >> Wl.inputs
        if use_action_syn:
            Summation(Wl.outputs, Al.outputs) >> el.mu
        else:
            Wl.outputs >> el.mu
        zl.z >> el.target
        zl.zF_prior >> Rl_intra.inputs
        Rl_intra.outputs >> el_intra.mu ## intra-predictions
        #zl.z_filter >> Rl_inter.inputs
        zl.zF_prior >> Rl_inter.inputs
        Rl_inter.outputs >> el_inter.mu ## inter-predictions
        if z_gamma < 0.:
            zl.z >> el_intra.target
            if use_cross_stream_syn:
                if inter_predict_innovation:
                    el.dmu >> el_inter.target ## innovation is target for cross-stream syn
                else:
                    zl.z >> el_inter.target

        ### Set up message-passing between l and l+1
        el.dtarget >> zl.j_td  ## top-down pressure on z^l
        #Summation(elC.dtarget, el.dtarget) >> zl.j_td
        zl.zF >> Vl.inputs  ## lateral entry
        el_intra.dmu >> ERl_intra.inputs  ## cross-stream entry
        if use_cross_stream_syn:
            el_inter.dmu >> ERl_inter.inputs
        if l != len(dynamics_motifs) - 1: ## wire up interaction from l+1 to l
            ## get level l+1 nodes
            (zlp1, elp1, elp1_intra, elp1_inter, Wlp1, Elp1, Rlp1_intra,
             ERlp1_intra, Rlp1_inter, ERlp1_inter, Vlp1, Alp1) = dynamics_motifs[l + 1]
            elp1.dmu >> Elp1.inputs  ## top-down entry

            ## we sum up cross-stream, top-down, & lateral pressures into z^l
            if use_cross_stream_syn:
                Summation(ERl_intra.outputs, ERl_inter.outputs, Elp1.outputs, Vl.outputs) >> zl.j
            else:
                Summation(ERl_intra.outputs, Elp1.outputs, Vl.outputs) >> zl.j
        else: ## if l == L, then only lateral and cross-stream pressures into z^l
            if use_cross_stream_syn:
                Summation(ERl_intra.outputs, ERl_inter.outputs, Vl.outputs) >> zl.j
            else:
                Summation(ERl_intra.outputs, Vl.outputs) >> zl.j

        ## Hebbian updates to encoder, cross-stream, and lateral synapses
        ### adjust encoder synapses
        zlm1.zF >> Wl.pre
        el.dmu >> Wl.post
        if use_action_syn:
            ### adjust action-conditional synapses
            a.zF >> Al.pre
            el.dmu >> Al.post
        ### adjust intra-stream prediction synapses
        zl.zF_prior >> Rl_intra.pre
        el_intra.dmu >> Rl_intra.post
        if use_cross_stream_syn:
            ### adjust outer cross-stream prediction synapses
            #zl.z_filter >> CRl.pre ##
            zl.zF_prior >> Rl_inter.pre
            el_inter.dmu >> Rl_inter.post
        ### adjust lateral inhibitory synapses
        zl.zF >> Vl.pre
        zl.zF >> Vl.post

        zlm1 = zl
        qlm1 = ql

    ## wiring is now completed
    ############################################################################
