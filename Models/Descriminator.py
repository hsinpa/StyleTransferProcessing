from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, LeakyReLU
from tensorflow.keras.layers import Add, BatchNormalization, UpSampling2D, DepthwiseConv2D, add
import tensorflow as tf
from tensorflow.keras import backend as K
from Models.Modules import _inverted_residual_block, _conv_block, _bottleneck, _make_divisible


class Descriminator:

    def Build(self, input_shape, k, alpha=1.0, data_format='channels_last'):
        """MobileNetv2
        This function defines a MobileNetv2 architectures.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            k: Integer, number of classes.
            alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
        # Returns
            MobileNetv2 model.
        """
        tf.keras.backend.set_image_data_format(data_format)
        inputs = Input(shape=input_shape, name='input_1')
        encode_layer = []

        first_filters = _make_divisible(32 * alpha, 8)
        x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))
        x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=3)

        x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=2)



        pass
