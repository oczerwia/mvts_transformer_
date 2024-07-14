"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

from src.datasets.dataset import collate_unsuperv
from src.utils import utils

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

import os
import sys
import time

import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_pareto_front
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

# 3rd party packages
import pandas as pd
import torch
from datasets.data import Normalizer, data_factory
from datasets.dataset import collate_unsuperv
from models.loss import get_loss_module
from models.ts_transformer import model_factory
from optimizers import get_optimizer
# Project modules
from options import Options
from running import NEG_METRICS, harden_steps, load_setup, pipeline_factory, setup, validate
from torch.utils.data import DataLoader


def initialize_config(trial):
    """Here we initialize all hyperparameters."""
    args = Options().parse()  # `argsparse` object
    config = load_setup(args)
    
    
    # Hyperparameters
    config["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    config["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    config["epochs"] = trial.suggest_int("epochs", 100, 500)
    config["l2_reg"] = trial.suggest_loguniform("l2_reg", 1e-5, 1e-1)
    config["dropout"] = trial.suggest_uniform("dropout", 0.0, 0.5)
    config["n_layers"] = trial.suggest_categorical("n_layers", [4,6,8,12])
    config["n_heads"] = trial.suggest_categorical("n_heads", [4,6,8,12])
    config["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128, 256, 512, 1024])

    config["activation"] = trial.suggest_categorical("activation", ["relu", "gelu"])
    config["subsample_factor"] = trial.suggest_categorical("subsample_factor", [1, 2, 4])
    config["lr_factor"] = [trial.suggest_float("lr_factor_1", 0.0, 0.9),
                            trial.suggest_float("lr_factor_2", 0.0, 0.9),
                              trial.suggest_float("lr_factor_3", 0.0, 0.9)]
    config["normalization_layer"] = trial.suggest_categorical("normalization_layer", ["LayerNorm", "BatchNorm"])

    config["optimizer"] = trial.suggest_categorical("optimizer", ["RAdam", "Adam"])

    config = setup(config)

    return config

def initialize_logging(config):
    # Create output directory if it does not exist
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    if not os.path.exists(os.path.join(config["output_dir"], "predictions")):
        os.makedirs(os.path.join(config["output_dir"], "predictions"))
    if not os.path.exists(os.path.join(config["output_dir"], "models")):
        os.makedirs(os.path.join(config["output_dir"], "models"))
    if not os.path.exists(os.path.join(config["output_dir"], "embeddings")):
        os.makedirs(os.path.join(config["output_dir"], "embeddings"))

    # Initialize logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # initialize seed
    torch.manual_seed(42)

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)

    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))  # command used to run

    return logger

def initialize_optimizer(config, model):
    weight_decay = config["l2_reg"] if config["l2_reg"] else 0
    optim_class = get_optimizer(config["optimizer"])
    optimizer = optim_class(
        model.parameters(), lr=config["lr"], weight_decay=weight_decay
    )
    return model, optimizer

def evaluate_test_set(config, test_data, model, device, loss_module, logger):
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    test_dataset = dataset_class(test_data, config=config)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate_unsuperv,
    )

    if config["extract_embeddings_only"]:
        embeddings_extractor = runner_class(
            model,
            test_loader,
            device,
            loss_module,
            print_interval=config["print_interval"],
            console=config["console"],
        )
        with torch.no_grad():
            embeddings = embeddings_extractor.extract_embeddings(keep_all=True)
            embeddings_filepath = os.path.join(
                os.path.join(config["output_dir"] + "/embeddings.pt")
            )
            torch.save(embeddings, embeddings_filepath)
        return
    else:
        test_evaluator = runner_class(
            model,
            test_loader,
            device,
            loss_module,
            print_interval=config["print_interval"],
            console=config["console"],
        )
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = "Test Summary: "
        for k, v in aggr_metrics_test.items():
            if v is None:
                v=0
            print_str += f"{k}: {np.round(v, 8)} | "
        # logger.info(print_str)
        return
    
def initialize_dataloader(config, model, my_data, val_data, device, optimizer, loss_module):
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, config=config)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate_unsuperv,
    )

    train_dataset = dataset_class(my_data, config=config)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True, # We can shuffle within
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate_unsuperv,
        prefetch_factor=1 if config["num_workers"] > 0 else None,
    )

    train_evaluator = runner_class(
        model,
        train_loader,
        device,
        loss_module,
        optimizer,
        l2_reg=config["l2_reg"] if config["l2_reg"] else 0,
        print_interval=config["print_interval"],
        console=config["console"],
    )
    val_evaluator = runner_class(
        model,
        val_loader,
        device,
        loss_module,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    return train_evaluator, train_loader, train_dataset, val_evaluator,val_loader, val_dataset

def main(trial):


    config = initialize_config(trial)
    total_epoch_time = 0

    total_start_time = time.time()

    logger = initialize_logging(config)
        

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config["gpu"] != "-1") else "cpu"
    )
    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    

    # Build data
    # logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]
    my_data = data_class(
        config["data_dir"],
        pattern=config["pattern"],
        n_proc=config["n_proc"],
        limit_size=config["limit_size"],
        config=config,
    )

    if config[
        "test_pattern"
    ]:  # used if test data come from different files / file patterns
        test_data = data_class(
            config["data_dir"], pattern=config["test_pattern"], n_proc=-1, config=config
        )
        test_indices = test_data.all_IDs
    if config[
        "val_pattern"
    ]:  # used if val data come from different files / file patterns
        val_data = data_class(
            config["data_dir"], pattern=config["val_pattern"], n_proc=-1, config=config
        )
        val_indices = val_data.all_IDs



    ##################################### MODEL SETUP ############################################
    # Create model
    # .info("Creating model ...")
    model = model_factory(config, my_data)

    if config["freeze"]:
        for name, param in model.named_parameters():
            if name.startswith("output_layer"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # logger.info("Model:\n{}".format(model))
    # logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(utils.count_parameters(model, trainable=True))
    )

    # Initialize optimizer

    model, optimizer = initialize_optimizer(config, model)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config["lr"]  # current learning step
    # Load model and optimizer state
    if config["load_model"]:
        model, optimizer, start_epoch = utils.load_model(
            model,
            config["load_model"],
            optimizer,
            config["resume"],
            config["change_output"],
            config["lr"],
            config["lr_step"],
            config["lr_factor"],
        )
    model.to(device)

    loss_module = get_loss_module(config)



    if config["test_only"] == "testset":  # Only evaluate and skip training
        logger.info("Evaluating on test set ...")
        evaluate_test_set(config, test_data, model, device, loss_module, logger)
        return
    
    train_evaluator, train_loader, train_dataset, val_evaluator,val_loader, val_dataset = initialize_dataloader(
        config, model, my_data, val_data, device, optimizer, loss_module
        )


    best_value = (
        1e16 if config["key_metric"] in NEG_METRICS else -1e16
    )  # initialize with +inf or -inf depending on key metric
    metrics = (
        []
    )  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(
        val_evaluator, config, best_metrics, best_value, epoch=0
    )
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    early_stopping_epochs = 10
    epsilon = 1e-6

    logger.info("Starting training...")
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        mark = epoch if config['save_all'] else 'last'
        # torch.multiprocessing.get_context('spawn').join()
        epoch_start_time = time.time()
        aggr_metrics_train = train_evaluator.train_epoch(
            epoch
        )  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = "Epoch {} Training Summary: ".format(epoch)
        for k, v in aggr_metrics_train.items():
            print_str += "{}: {:8f} | ".format(k, v)
        logger.info(print_str)
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        # evaluate if first or last epoch or at specified interval
        if (
            (epoch == config["epochs"])
            or (epoch == start_epoch + 1)
            or (epoch % config["val_interval"] == 0)
        ):
            aggr_metrics_val, best_metrics, best_value = validate(
                val_evaluator,
                config,
                best_metrics,
                best_value,
                epoch,
            )
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

        utils.save_model(
            os.path.join(config["save_dir"], "model_{}.pth".format(mark)),
            epoch,
            model,
            optimizer,
        )

        # Learning rate scheduling
        if epoch == config["lr_step"][lr_step]:
            utils.save_model(
                os.path.join(config["save_dir"], "model_{}.pth".format(epoch)),
                epoch,
                model,
                optimizer,
            )
            lr = lr * config["lr_factor"][lr_step]
            if (
                lr_step < len(config["lr_step"]) - 1
            ):  # so that this index does not get out of bounds
                lr_step += 1
            logger.info("Learning rate updated to: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Difficulty scheduling

        if config['harden'] and harden_steps(epoch):
            old_ratio = train_loader.dataset.masking_ratio
            old_mask_len = train_loader.dataset.mean_mask_length
            train_loader.dataset.update()
            val_loader.dataset.update()
            logger.info(f"Hardening Masking Ratio: {old_ratio} -> {train_loader.dataset.masking_ratio}")
            logger.info(f"Hardening Masking Length: {old_mask_len} -> {train_loader.dataset.mean_mask_length}")

        if (best_metrics['loss'] <= (prev_loss + epsilon)) \
            and (best_metrics['loss'] >= (prev_loss - epsilon)) \
            and (best_metrics['Correlation'] <= (prev_correlation + epsilon))\
            and (best_metrics['Correlation'] >= (prev_correlation - epsilon)):
                early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        # Check if early stopping condition is met
        if early_stopping_counter >= early_stopping_epochs:
            logger.info("Early stopping condition met. Stopping training.")
            break

        # Update previous loss and correlation
        prev_loss = best_metrics['loss']
        prev_correlation = best_metrics['Correlation']
            

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(
        config["output_dir"], "metrics_" + config["experiment_name"] + ".xls"
    )
    # TODO: Export this as csv, so that I can read it during tuning
    best_predictions = list(
        np.load(
            os.path.join(config["output_dir"], "predictions", "best_predictions.npz"),
            allow_pickle=True,
        )["metrics"]
    )
    metrics_df = pd.DataFrame(metrics, columns=header)
    metrics_filepath = os.path.join(
        config["output_dir"], "metrics_" + config["experiment_name"] + ".csv"
    )
    metrics_df.to_csv(metrics_filepath)

    logger.info(
        "Best {} was {}. Other metrics: {}".format(
            config["key_metric"], best_value, best_metrics
        )
    )

    return best_metrics['loss'], best_metrics['Correlation']

if __name__ == "__main__":

    study = optuna.create_study(directions=["minimize","maximize"],
                                sampler=optuna.samplers.TPESampler(),
                                storage="sqlite:///db.sqlite3")
    
    study.optimize(main, n_trials=5, n_jobs=1, gc_after_trial=True)
    study.trials_dataframe().to_csv("results.csv")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")