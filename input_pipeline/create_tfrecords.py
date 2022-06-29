import gin
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
import os


@gin.configurable
def create_tfrecords(data_dir, window_size, window_shift):
    """
    create train/test/val TFRecords for HAPT_dataset
    """

    # -------------------------------- Data Preprocessing --------------------------------#
    path_rawdata = data_dir + '/HAPT_dataset/RawData/'

    # --- for each experiment 1-61, get ["a_x","a_y","a_z","g_x","g_y","g_z","label"(<=5)] of each time step ---
    # initialize
    lookup_table_rawdata = pd.read_csv(path_rawdata + 'labels.txt',
                                       names=["experiment", "user", "label", "start", "end"],
                                       delim_whitespace=True)
    lookup_table_rawdata["label"] -= 1  # activity 1,...,12 -> 0,...,11

    files_acc = sorted(glob.glob(path_rawdata + "acc_*.txt"))
    files_gyro = sorted(glob.glob(path_rawdata + "gyro_*.txt"))

    df_train_list, df_val_list, df_test_list = [], [], []

    for file_acc, file_gyro in zip(files_acc, files_gyro):
        # df[["acc", "gyro"]]
        acc = pd.read_csv(file_acc, names=["a_x", "a_y", "a_z"], delim_whitespace=True)
        gyro = pd.read_csv(file_gyro, names=["g_x", "g_y", "g_z"], delim_whitespace=True)
        df = pd.concat([acc, gyro], axis=1)
        # z-score normalization for each experiment before dropping time steps which are unlabeled or with label>=6
        df = (df - df.mean()) / df.std()

        # experiment_id between "exp" "_"; user_id between "user" "."
        experiment_id = int(file_acc.partition("exp")[2].partition("_")[0])
        user_id = int(file_acc.partition("user")[2].partition(".")[0])

        # df["label"]: labels of each time step need to be specified
        df["label"] = np.nan
        # according to tutor's advice:
        # samples of 6 postural transitions are significantly less than samples of 6 basic activities
        # due to the high imbalance of the dataset, only do classification for the 6 basic activities(label 0~5)
        lookup_table = lookup_table_rawdata.loc[
            (lookup_table_rawdata["experiment"] == experiment_id) &
            (lookup_table_rawdata["label"] <= 5)]
        # time steps from "start" to "end" have the same "label"
        for l, start, end in zip(lookup_table["label"], lookup_table["start"], lookup_table["end"]):
            df.loc[start:end, "label"] = l
        # nan: time steps which are unlabeled or with label>=6; drop them out
        df = df.dropna()

        # --- train/val/test split ---
        if user_id <= 21:  # train: user 01 to 21
            df_train_list.append(df)
        elif 22 <= user_id <= 27:  # test: user 22 to 27
            df_test_list.append(df)
        else:  # val: user 28 to 30
            df_val_list.append(df)

    df_train = pd.concat(df_train_list)
    df_test = pd.concat(df_test_list)
    df_val = pd.concat(df_val_list)

    # --- sliding_windows; sequence to label---
    @tf.function
    def sliding_windows(x, size=window_size, shift=window_shift):
        x = tf.data.Dataset.from_tensor_slices(x).window(size, shift, drop_remainder=True)
        x = x.flat_map(lambda window: window.batch(size, drop_remainder=True))
        x = x.map(lambda ds: (ds[:, :-1], tf.cast(ds[:, -1], tf.int64)))
        # only when all samples in a window belong to a single class will this window be left
        # that is, labels of each time step in the window should be equal
        x = x.filter(lambda data, labels: tf.math.reduce_all(labels == labels[0]))
        # remain the single class label for each window, e.g. [5 5 ... 5] -> 5, reduce tensor shape (window_size,) to ()
        x = x.map(lambda data, labels: (data, labels[0]))
        return x

    ds_train = sliding_windows(df_train)
    ds_test = sliding_windows(df_test)
    ds_val = sliding_windows(df_val)

    # -------------------------------- Data Preprocessing --------------------------------#

    # --------------------------------- Create TFRecords -------------------------------- #
    # https://www.tensorflow.org/tutorials/load_data/tfrecord
    # https://www.kaggle.com/ryanholbrook/tfrecords-basics/notebook

    # --- create tf.Examples for train/test/val dataset and write them to each TFRecord ---
    def make_example(data, label):
        """create a single tf.Example to represent a single element (data, label) in tf.Dataset"""
        # serialize tensors with tf.io.serialize_tensor, get the bytestring with the numpy method,
        # and encode them in a BytesList
        data = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(data).numpy()]))
        label = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.serialize_tensor(label).numpy()]))

        # feature dict {feature name: bytes_list} -> Features -> Example
        feature = {'data': data, 'label': label}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()  # serialize Example

    ds_train = ds_train.map(lambda data, label: tf.py_function(make_example, [data, label], tf.string))
    ds_test = ds_test.map(lambda data, label: tf.py_function(make_example, [data, label], tf.string))
    ds_val = ds_val.map(lambda data, label: tf.py_function(make_example, [data, label], tf.string))

    tfdata_dir = os.path.abspath(os.path.dirname(__file__))
    train_filename = tfdata_dir + "/train.tfrecord"
    val_filename = tfdata_dir + "/val.tfrecord"
    test_filename = tfdata_dir + "/test.tfrecord"
    tf.data.experimental.TFRecordWriter(train_filename).write(ds_train)
    tf.data.experimental.TFRecordWriter(val_filename).write(ds_test)
    tf.data.experimental.TFRecordWriter(test_filename).write(ds_val)
    # -------------------------------- Create TFRecords -------------------------------- #
