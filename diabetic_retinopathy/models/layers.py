import gin
import tensorflow as tf

@gin.configurable
def vgg_block(inputs, number_of_filters,
              kernel_size): # <- configs
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(number_of_filters, kernel_size, padding='same', activation='relu')(inputs)
    out = tf.keras.layers.Conv2D(number_of_filters, kernel_size, padding='same', activation='relu')(out)
    out = tf.keras.layers.MaxPool2D( (2, 2) )(out)

    return out


def inception_module(input, filter_1, filter_2_1, filter_2_2, filter_3_1, filter_3_2, filter_4, kernel_initializer) -> tf.Tensor:
    
    path_1 = tf.keras.layers.Convolution2D(filters=filter_1, kernel_size=1, padding='same', activation='relu',
                                           kernel_initializer=kernel_initializer)(input)

    path_2 = tf.keras.layers.Convolution2D(filters=filter_2_1, kernel_size=1, padding='same', activation='relu',
                                           kernel_initializer=kernel_initializer)(input)
    path_2 = tf.keras.layers.Convolution2D(filters=filter_2_2, kernel_size=3, padding='same', activation='relu',
                                           kernel_initializer=kernel_initializer)(path_2)

    path_3 = tf.keras.layers.Convolution2D(filters=filter_3_1, kernel_size=1, padding='same', activation='relu',
                                           kernel_initializer=kernel_initializer)(input)
    path_3 = tf.keras.layers.Convolution2D(filters=filter_3_2, kernel_size=5, padding='same', activation='relu',
                                           kernel_initializer=kernel_initializer)(path_3)

    path_4 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    path_4 = tf.keras.layers.Convolution2D(filters=filter_4, kernel_size=1, padding='same', activation='relu',
                                           kernel_initializer=kernel_initializer)(path_4)

    return tf.keras.layers.Concatenate(axis=-1)( [path_1, path_2, path_3, path_4] )
