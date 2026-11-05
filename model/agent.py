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

    def visualize_filters(
            self,
            out_dir,
            file_tag,
            use_tensor_lc_format=True,
            w1_macro_grid=(5, 5),
            contrast_by_data=False  # True
    ):
        pass

    def visualize_layer2_macro_fields(
            self,
            pixel_shape,  # tuple (8, 8), size of your raw input pixel patches
            prefix="./macro_filters2",
            suffix=".jpg"
    ):
        pass