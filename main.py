import logging
import wandb
from input_pipeline import datasets
from utils import utils_params, utils_misc
from absl import app, flags
from evaluation.eval import evaluate
from train import Trainer
from models.architectures import *
import os


FLAGS = flags.FLAGS
flags.DEFINE_string('train_or_evaluate', 'evaluate',
                    "Specify train/evaluate/train_and_evaluate")
flags.DEFINE_string('data_dir', "/home/data",
                    'Specify the parent dir of HAPT_dataset,when GPU, /home/data')
flags.DEFINE_string('model_name', 'sequence_GRU_model_GlobalMaxPooling', 'Name of the model')
flags.DEFINE_integer('n_classes', 6, '6/12-class classification problem')


def main(argv):
    # generate folder structures 'experiments'
    run_paths = utils_params.gen_run_folder()

    # set loggers, INFO and above will be logged into the file run.log
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    config_file = 'configs/config.gin'
    if FLAGS.train_or_evaluate == 'evaluate':
        file_path = input("In case any change of hyperparameters after tuning or different choices of models,\n "
                          "you'd better to specify the path of config_operative.gin before evaluating, e.g.\n"
                          "xxx/config_operative.gin. Make sure it matches the model you've chosen. If you still\n"
                          " want to use the original config.gin, just press enter key to exit: \n")
        if os.path.isfile(file_path):
            config_file = file_path
            logging.info(f"Using tuned hyperparameters in {file_path}, "
                         f"instead of using hyperparameters in the original config.gin")
    gin.parse_config_files_and_bindings([config_file], [], skip_unknown=True)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    logging.info(f"------------ run folder {run_paths['path_model_id']} ------------")

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load(data_dir=FLAGS.data_dir)

    # model
    if FLAGS.model_name == 'sequence_LSTM_model_GlobalMaxPooling':
        model = sequence_LSTM_model(input_shape=[250, 6], pooling="GlobalMaxPooling", n_classes=FLAGS.n_classes)
    elif FLAGS.model_name == 'sequence_LSTM_model_GlobalAveragePooling':
        model = sequence_LSTM_model(input_shape=[250, 6], pooling="GlobalAveragePooling", n_classes=FLAGS.n_classes)
    elif FLAGS.model_name == 'sequence_GRU_model_GlobalMaxPooling':
        model = sequence_GRU_model(input_shape=[250, 6], pooling="GlobalMaxPooling", n_classes=FLAGS.n_classes)
    elif FLAGS.model_name == 'sequence_GRU_model_GlobalAveragePooling':
        model = sequence_GRU_model(input_shape=[250, 6], pooling="GlobalAveragePooling", n_classes=FLAGS.n_classes)
    else:
        raise ValueError('No such a model available.')
    model.summary(print_fn=logging.info)

    if FLAGS.train_or_evaluate == 'train':
        logging.info("------------ Only train ------------")
        # setup wandb
        wandb.init(project='human_activity_recognition_train', name=run_paths['path_model_id'],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        trainer = Trainer(model, ds_train, ds_val, n_classes=FLAGS.n_classes, run_paths=run_paths)
        for _ in trainer.train():
            continue

    elif FLAGS.train_or_evaluate == 'evaluate':
        logging.info("------------ Only evaluate ------------")
        checkpoint_dir = input("Only evaluate. Muss specify the checkpoint_dir, e.g. xxx/ckpts/train :\n")
        evaluate(model, ds_test, data_dir=FLAGS.data_dir, n_classes=FLAGS.n_classes,
                 run_paths=run_paths, checkpoint_dir=checkpoint_dir)

    elif FLAGS.train_or_evaluate == 'train_and_evaluate':
        logging.info("------------ Train and evaluate ------------")
        # setup wandb
        wandb.init(project='human_activity_recognition_train_and_evaluate', name=run_paths['path_model_id'],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        trainer = Trainer(model, ds_train, ds_val, n_classes=FLAGS.n_classes, run_paths=run_paths)
        for _ in trainer.train():
            continue
        evaluate(model, ds_test, data_dir=FLAGS.data_dir, n_classes=FLAGS.n_classes,
                 run_paths=run_paths, checkpoint_dir=run_paths['path_ckpts_train'])

    else:
        raise ValueError("Muss specify train/evaluate/train_and_evaluate for FLAGS.train_or_evaluate")

    return 0


if __name__ == "__main__":
    app.run(main)
