from ngclearn.utils.io_utils import makedir

class Agent(): ## base agent construct for housing models

    def __init__(
            self,
            dkey,
            saveDir=None,
            **kwargs
    ):
        self.dkey = dkey
        self.saveDir = saveDir
        if self.saveDir is not None:
            makedir(self.saveDir)

        self.params = []

    def get_param_stats(self):
        pass

    def process(self, x, action=None, adapt_synapses=False):
        pass

    def save_to_disk(self, params_only=False):
        pass

    def _load_from_disk(self, model_directory):
        pass
