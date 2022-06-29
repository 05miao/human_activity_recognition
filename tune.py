import logging
import os
import wandb
import math
from input_pipeline.datasets import load
from models.architectures import *
from train import Trainer
from utils import utils_params, utils_misc


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')  # ['Trainer.learning_rate=xxx', 'Trainer.total_steps=xxx', ...]

        # generate folder structures
        run_paths = utils_params.gen_run_folder(
            ','.join(bindings))  # Join all items in the list into a string, using , as separator

        run.name = run_paths['path_model_id'].split('/')[-1]  # used in Linux. When Windows, use '\\' instead.

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        logging.info(f"------------ run folder {run_paths['path_model_id']} ------------")

        gin.parse_config_files_and_bindings(
            [os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs/config.gin'))],
            bindings, skip_unknown=True)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        ds_train, ds_val, ds_test = load('hapt')

        # model
        model = sequence_LSTM_model(input_shape=[250, 6], n_classes=6)

        trainer = Trainer(model, ds_train, ds_val, n_classes=6, run_paths=run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    'name': 'human_activity_recognition_tune',
    'method': 'bayes',
    'metric': {'name': 'best_val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'Trainer.learning_rate': {'distribution': 'log_uniform', 'min': math.log(1e-6), 'max': math.log(1e-4)},
        'Trainer.total_steps': {'values': [10000]},
        'sequence_LSTM_model.dropout_rate': {'distribution': 'uniform', 'min': 0.3, 'max': 0.5},
        }
}
sweep_id = wandb.sweep(sweep_config, project='human_activity_recognition_tune')
wandb.agent(sweep_id, function=train_func, count=10)
