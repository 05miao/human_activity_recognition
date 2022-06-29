from models.layers import *
import tensorflow as tf
import gin


@gin.configurable
def sequence_LSTM_model(input_shape, dropout_rate, pooling, n_classes=6):
    """Defines a LSTM architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        pooling (str): pooling type
        n_classes (int): number of label category (must be 6 or 12)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish feature extraction blocks
    outputs = basic_LSTM(inputs, 256)
    outputs = basic_Dense(outputs, 128)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_LSTM(outputs, 128)
    outputs = basic_Dense(outputs, 64)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_LSTM(outputs, 64)
    if pooling == "GlobalMaxPooling":
        outputs = tf.keras.layers.GlobalMaxPooling1D()(outputs)
    elif pooling == "GlobalAveragePooling":
        outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    else:
        raise ValueError("No such a pooling layer available in the model")
    outputs = basic_Dense(outputs, 32)

    # establish output layers
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_LSTM_model')


@gin.configurable
def sequence_GRU_model(input_shape, dropout_rate, pooling, n_classes=6):
    """Defines a GRU architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        pooling (str): pooling type
        n_classes (int): number of label category (must be 6 or 12)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish feature extraction blocks
    outputs = basic_GRU(inputs, 256)
    outputs = basic_Dense(outputs, 128)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_GRU(outputs, 128)
    outputs = basic_Dense(outputs, 64)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_GRU(outputs, 64)
    if pooling == "GlobalMaxPooling":
        outputs = tf.keras.layers.GlobalMaxPooling1D()(outputs)
    elif pooling == "GlobalAveragePooling":
        outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    else:
        raise ValueError("No such a pooling layer available in the model")
    outputs = basic_Dense(outputs, 32)

    # establish output layers
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_GRU_model')
