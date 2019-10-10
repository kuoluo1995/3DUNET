from keras import Input, Model
from keras.layers import BatchNormalization, concatenate, Conv3D, Activation, MaxPooling3D, UpSampling3D, \
    Deconvolution3D
from keras.optimizers import Adam
from keras import backend as K

K.set_image_dim_ordering('th')
K.set_image_data_format("channels_first")


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def convolution_block(input_layer, n_filters, kernel=(3, 3, 3), padding='same', strides=(1, 1, 1)):
    block = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    layer = BatchNormalization(axis=1)(block)
    return Activation('relu')(layer)


def unet_3d(input_shape, n_base_filters=32):
    _input = Input(input_shape)
    _block = _input
    bridge_list = list()

    for layer_depth in range(4):
        _block = convolution_block(input_layer=_block, n_filters=n_base_filters * (2 ** layer_depth))
        _block = convolution_block(input_layer=_block, n_filters=n_base_filters * (2 ** layer_depth))
        if layer_depth < 4 - 1:
            bridge_list.append(_block)
            _block = MaxPooling3D(pool_size=(2, 2, 2))(_block)

    for layer_depth in reversed(range(4)):
        if layer_depth < 4 - 1:
            _block = UpSampling3D(size=(2, 2, 2))(_block)  # or change ti Deconvolution3D
            # _block = Deconvolution3D(filters=n_base_filters * (2 ** layer_depth), kernel_size=(2, 2, 2),
            #                          strides=(2, 2, 2))
            _block = concatenate([_block, bridge_list[layer_depth]], axis=1)
        _block = convolution_block(input_layer=_block, n_filters=n_base_filters * (2 ** layer_depth))
        _block = convolution_block(input_layer=_block, n_filters=n_base_filters * (2 ** layer_depth))

    final_convolution = Conv3D(1, (1, 1, 1), activation='sigmoid')(_block)
    model = Model(inputs=_input, outputs=final_convolution)
    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coefficient_loss, metrics=[dice_coefficient])
    return model
