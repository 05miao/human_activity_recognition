import logging
import os
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt

from evaluation.metrics import ConfusionMatrix
from visualization.visualize import visualize


def evaluate(model, ds_test, data_dir, n_classes, run_paths, checkpoint_dir):

    """

    Args:
        model (keras.Model): model
        ds_test (tf.data.Dataset): test set
        data_dir (str): parent directory of e.g. IDRID_dataset
        n_classes (int): can be 2 or 5
        run_paths (dict): contains the info about the directory structure of 'experiments'. This directory
          structure is automatically generated by utils_params.gen_run_folder().
        checkpoint_dir (str): e.g. 'xxx/ckpts/train'. used to restore a training checkpoint,.
          When starting evaluation soon after training, this dir can be run_paths['path_ckpt_train'].
          When just starting evaluation without training, this dir must be manually specified.

    """

    logging.info("\n------------ Starting evaluation ------------")

    confusion_matrix = ConfusionMatrix(n_classes)
    ckpt = tf.train.Checkpoint(model=model)
    logging.info(f"restore checkpoint: \n{tf.train.latest_checkpoint(checkpoint_dir)}")
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    for test_features, test_labels in ds_test:
        test_predictions = model(test_features, training=False)
        test_predictions = tf.math.argmax(test_predictions, axis=1)
        # confusion matrix
        confusion_matrix.update_state(test_labels, test_predictions)

    logging.info('\n------ Evaluation statistics')

    # confusion matrix
    logging.info(f"Confusion matrix: \n{confusion_matrix.result().numpy()}")

    # precision, sensitivity, f1
    precision, sensitivity, f1 = confusion_matrix.metrics_from_confusion_matrix()
    logging.info(f"Precision: {precision}, Sensitivity: {sensitivity}, F1: {f1}")

    # show the confusion matrix
    plt.figure(dpi=300)
    sn.heatmap(confusion_matrix.result().numpy(), annot=True)
    plt.title('Evaluation - Confusion matrix')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(run_paths['path_plt'], 'Metrics.png'), bbox_inches='tight')
    plt.show()

    # visualize the truth and predict of the time series of each experiment
    visualize(model, data_dir, n_classes, run_paths, checkpoint_dir)
