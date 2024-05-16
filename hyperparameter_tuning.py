"""Run hyperparameter tuning for TST."""
# TODO: Read whether HP Tuning can be performed automatically somehow

from abc import ABC


class HPSearch(ABC):
    """Abstract class for parameter search strategy."""

    def search(self, **kwargs):
        ...
    

class GridSearch(HPSearch):
    """Perform Grid Search"""

    def __init__(self, **kwargs):
        ...

    def search(self, **kwargs):
        ...

class HyperParameterTuning:

    # HyperParameters: 
    activation_function = ["gelu", "relu"]
    batch_size = [16, 32, 64, 128, 256]
    model_dimensionality = [128, 256, 512, 1024] # Do these need to be 2 potentials?
    feedforward_dim = [128, 256, 512] # NN hidden layers
    dropout = [0.1, 0.2]
    epochs = [100, 200, 300, 400, 500] # TODO: Eig brauchen wir nur fixed? Best model wird dann festgehalten


    end_hint = 0.0 # TODO: Find out what that is
    data_window_len = None # TODO: Find out what that is
    global_reg = False # TODO: Find out what that is
    harden = False # TODO
    l2_reg = [0, 0.1, 0.2] # TODO: Read about sensible parameter range

    learning_rate = [0.01, 0.001, 0.0001]
    learning_rate_factor = [0.1] # TODO:  Find out what that is
    learning_rate_step = [1000000] # TODO:  Find out what that is

    data_class = None # Here custom data class
    comment = None # Here model parameters?
    experiment_name = None # Do as in ROC -> create reference dataframe
    data_dir = None
    freeze = False # What does freeze mean?

    mask_distribution = "geometric" # Always
    mask_mean_length = [3] # Should be much larger later
    mask_features = [0,1] # TODO:  Find out what that is / is this what channels are masked?
    mask_mode = "separate" # TODO:  Find out what that is
    mask_ratio = [0.15]

    normalization = "standardization"
    normalization_layer = ["BatchNorm", "LayerNorm"]

    num_heads = 8
    num_layers= 3

    pos_encoding = ["learnable","static"] # TODO: Look up exact parameter name for static

    subsample_factor = [None, 3, 6]

    optimizer = ["RAdam", "Adam"]
    task = "imputation"

    extract_embeddings = None # Won't need that during HP search



    left_over = {
        "output_dir": "outputs/regression_embeddings_2024-05-09_14-14-48_x37",
        "pattern": "TRAIN",
        "pred_dir": "outputs/regression_embeddings_2024-05-09_14-14-48_x37/predictions",
        "records_file": "regression_records_embeddings.xls",
        "save_dir": "outputs/regression_embeddings_2024-05-09_14-14-48_x37/checkpoints",
        "seed": null,
        
        
        "tensorboard_dir": "outputs/regression_embeddings_2024-05-09_14-14-48_x37/tb_summaries",
        "test_from": null,
        "test_only": null,
        "test_pattern": null,
        "test_ratio": 0,
        "val_interval": 2,
        "val_pattern": null,
        "val_ratio": 0.2
    }

    def __init__(self, search_method: HPSearch):
        ...

    
    #

    def run(self):
        ...

if __name__ == "__main__":
    # Run
    pass


