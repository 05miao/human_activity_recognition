import gin
import tensorflow as tf
import logging
import glob
from input_pipeline.create_tfrecords import create_tfrecords
import os

@gin.configurable
def load(name, data_dir):
    if name == "hapt":
        logging.info(f"------------ Preparing dataset {name}... -----------")

        tfdata_dir = os.path.abspath(os.path.dirname(__file__))
        train_filename = tfdata_dir + "/train.tfrecord"
        val_filename = tfdata_dir + "/val.tfrecord"
        test_filename = tfdata_dir + "/test.tfrecord"

        if not glob.glob(tfdata_dir+"/*.tfrecord"):
            create_tfrecords(data_dir)

        # read TFRecords
        raw_ds_train = tf.data.TFRecordDataset(train_filename)
        raw_ds_test = tf.data.TFRecordDataset(val_filename)
        raw_ds_val = tf.data.TFRecordDataset(test_filename)

        def _parse_function(example):
            feature = {
                'data': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
            }
            example = tf.io.parse_single_example(example, feature)
            data = tf.io.parse_tensor(example["data"], tf.float64)
            label = tf.io.parse_tensor(example["label"], tf.int64)
            return data, label

        ds_train = raw_ds_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = raw_ds_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = raw_ds_val.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    else:
        raise ValueError("No such an dataset available")

    return prepare(ds_train, ds_val, ds_test)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, batch_size, caching):
    buffer_size = 0
    for _ in ds_train:
        buffer_size += 1
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test
