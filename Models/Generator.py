from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, LeakyReLU
from tensorflow.keras.layers import Add, BatchNormalization, UpSampling2D, DepthwiseConv2D, add
import tensorflow as tf
from tensorflow.keras import backend as K
from Models.Modules import _inverted_residual_block, _conv_block, _bottleneck, _make_divisible, transition_layer

class Generator:

    def build_encoder(self, input, alpha=1.0):
        encode_layer = []
        first_filters = _make_divisible(32 * alpha, 8)

        x = _conv_block(input, first_filters, (3, 3), strides=(2, 2))
        x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)

        x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)

        encode_layer.append(x)
        x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=2)

        x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

        return (x, encode_layer)

    def build_decoder(self, x, encode_layer):
        alpha = 1
        denseSetup = [(6, 160, 96), (8, 64, 32), (12, 28, 24), (6, 18, 16)]
        denseSetCount = len(denseSetup)
        for i in range(denseSetCount):
            r, train_f, trans_f = denseSetup[i]

            x = _inverted_residual_block(x, filters=train_f, kernel=3, strides=1, n=r, alpha=alpha, t=2)
            x = transition_layer(x, trans_f)
            if encode_layer is not None:
                x = Add()([x, encode_layer[denseSetCount - i - 1]])

        x = _inverted_residual_block(x, filters=8, kernel=3, strides=1, t=3, n=2, alpha=alpha)
        x = transition_layer(x, 8)

        x = Conv2D(filters=3, kernel_size=9, strides=1, padding='same', name='output_1', activation='tanh')(x)
        return x

    def Build(self, input_shape, alpha=1.0, data_format='channels_last'):
        tf.keras.backend.set_image_data_format(data_format)
        inputs = Input(shape=input_shape, name='g_input_1')

        latent_value, u_layers = self.build_encoder(inputs)
        decoder = self.build_decoder(latent_value, u_layers)

        model = Model(inputs, decoder)
        return model


if __name__ == '__main__':
    model = Generator()
    m = model.Build((128, 128, 3), data_format='channels_last')

    print(m.summary())

    # m.save("../save_model/", save_format="tf")


