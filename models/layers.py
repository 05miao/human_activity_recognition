import tensorflow as tf


def basic_Dense(inputs, units, use_bn=True, use_activation=True):
    """A single dense layer

    Parameters:
        inputs (Tensor): input of the dense layer
        units (int): number of filters used for the dense layer
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single dense layer
    """

    outputs = tf.keras.layers.Dense(units, activation='linear')(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


def basic_LSTM(inputs, units, return_sequences=True, use_bn=True, use_activation=True):
    """A single LSTM layer

    Parameters:
        inputs (Tensor): input of the LSTM layer
        units (int): number of filters used for the LSTM layer
        return_sequences (bool): whether output is sequence or not (Default is True)
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single LSTM layer
    """

    outputs = tf.keras.layers.LSTM(units, return_sequences=return_sequences)(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


def basic_GRU(inputs, units, return_sequences=True, use_bn=True, use_activation=True):
    """A single GRU layer

    Parameters:
        inputs (Tensor): input of the GRU layer
        units (int): number of filters used for the GRU layer
        return_sequences (bool): whether output is sequence or not (Default is True)
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single GRU layer
    """

    outputs = tf.keras.layers.GRU(units, return_sequences=return_sequences)(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs
