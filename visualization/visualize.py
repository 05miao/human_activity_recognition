import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import more_itertools as mit
import logging


def visualize(model, data_dir, n_classes, run_paths, checkpoint_dir, window_size=250, batch_size=32):
    logging.info('------------ Starting visualization -------------')

    while True:
        try:
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

            path_rawdata = data_dir + "/HAPT_dataset/RawData/"
            lookup_table_rawdata = pd.read_csv(path_rawdata + "labels.txt",
                                               names=["experiment", "user", "label", "start", "end"], delim_whitespace=True)
            lookup_table_rawdata["label"] -= 1  # activity 1,...,12 -> 0,...,11
            label_name = pd.read_csv(data_dir + "/HAPT_dataset/activity_labels.txt",
                                     names=["label", "name"], delim_whitespace=True)
            label_name["label"] -= 1

            experiment_id = int(input(
                f"Enter an experiment_id of test set, i.e. an integer in [44,55]. Or 'exit' to exit: "))
            file_acc, file_gyro = sorted(glob.glob(path_rawdata + f"*{experiment_id}*.txt"))

            acc = pd.read_csv(file_acc, names=["a_x", "a_y", "a_z"], delim_whitespace=True)
            acc = (acc - acc.mean()) / acc.std()
            gyro = pd.read_csv(file_gyro, names=["g_x", "g_y", "g_z"], delim_whitespace=True)
            gyro = (gyro - gyro.mean()) / gyro.std()
            # df[["acc", "gyro"]]
            df = pd.concat([acc, gyro], axis=1)

            # e.g. lookup_table for experiment 44, label<=5 (6-classification)
            lookup_table = lookup_table_rawdata.loc[
                (lookup_table_rawdata["experiment"] == experiment_id) &
                (lookup_table_rawdata["label"] <= (n_classes - 1))]
            # df["label"]: labels(<=5) of each time step need to be specified
            df["label"] = np.nan
            for label, start, end in zip(lookup_table["label"], lookup_table["start"], lookup_table["end"]):
                df.loc[start:end, "label"] = label

            # nan: time steps which are unlabeled or with label>=6; drop them out
            df = df.dropna()

            # when sliding windows, remain those windows whose samples belong to a simple class
            def filter_func(x):
                y = x["label"].values
                if (y == y[0]).all():  # True when all labels in the window are same
                    return x

            # sliding windows without overlap
            num_of_windows = np.floor(len(df.index) / window_size).astype('int')
            filtered_df = [filter_func(df.iloc[i * window_size: (i + 1) * window_size]) for i in range(num_of_windows)]

            # create dataset; one label each window
            df = pd.concat(filtered_df)

            def make_ds(x):
                return tf.data.Dataset.from_tensor_slices(x).batch(window_size) \
                    .map(lambda d: (d[:, :-1], tf.cast(d[:, -1][0], tf.int64))) \
                    .batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

            ds = make_ds(df)

            # make predictions for each window
            predictions = []
            for test_features, test_labels in ds:
                test_predictions = model(test_features, training=False)
                test_predictions = tf.math.argmax(test_predictions, axis=1)
                predictions += test_predictions.numpy().tolist()
            df["prediction"] = np.repeat(predictions, window_size)
            # df.to_csv(os.path.join(run_paths['path_model_id'],
            #                        f"df_{n_classes}_classification_for_experiment_{experiment_id}.csv"))

            # ----------------------------- plot_truth_or_predict ----------------------------- #
            plt.rcParams.update({'font.size': 7})

            def plot_truth_or_predict(truth_or_predict="truth"):
                fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 4.4), dpi=300,
                                                    gridspec_kw={'height_ratios': [2, 2, 0.4]})

                colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_classes))

                ax0.plot(acc.index, acc, linewidth=0.25)
                ax1.plot(gyro.index, gyro, linewidth=0.25)
                if truth_or_predict == "truth":
                    for l, s, e in zip(lookup_table["label"], lookup_table["start"], lookup_table["end"]):
                        ax0.axvspan(s, e, facecolor=colors[l], alpha=0.3)
                        ax1.axvspan(s, e, facecolor=colors[l], alpha=0.3)
                elif truth_or_predict == "predict":
                    for g in mit.consecutive_groups(df.index):
                        consecutive_time_period = list(g)
                        s = consecutive_time_period[0]
                        e = consecutive_time_period[-1]
                        predict = df.loc[s, "prediction"]
                        ax0.axvspan(s, e, facecolor=colors[predict], alpha=0.3)
                        ax1.axvspan(s, e, facecolor=colors[predict], alpha=0.3)
                else:
                    raise ValueError("No such an option")
                ax0.set_title(f"experiment {experiment_id} - {truth_or_predict} - accelerometer")
                ax1.set_title(f"experiment {experiment_id} - {truth_or_predict} - gyroscope")

                color_index = np.arange(n_classes)  # [0,...,5]
                left_bound = color_index / n_classes  # [0,...,5]/6
                right_bound = (color_index + 1) / n_classes  # [1,...,6]/6
                midpoint = (color_index + 0.5) / n_classes
                for i, l, r in zip(color_index, left_bound, right_bound):
                    ax2.axvspan(l, r, facecolor=colors[i], alpha=0.3)
                ax2.set_title(f"{n_classes}-classification")
                ax2.set_xticks(midpoint)
                ax2.set_xticklabels(label_name["name"][:n_classes])  # rotation=45
                ax2.set_yticks([])
                ax2.margins(x=0)

                fig.tight_layout(rect=[0, 0, 1, 0.9])
                plt.show()

            plot_truth_or_predict("truth")
            plot_truth_or_predict("predict")
            # ----------------------------- plot_truth_or_predict ----------------------------- #
        except ValueError:
            logging.info('Exit the visualization')
            break
