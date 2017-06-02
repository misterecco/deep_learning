from .basic import conv_2d, deconv_2d, max_pool_2d, relu, batch_norm


def conv(signal, out_channels):
    return conv_2d(signal, 3, 1, out_channels)


def upconv(signal, out_channels):
    return deconv_2d(signal, 3, 2, out_channels)


def convout(signal):
    return conv_2d(signal, 1, 1, 3)


def max_pool(signal):
    return max_pool_2d(signal, 2, 2)


def bn_conv_relu(signal, out_channels):
    return relu(conv(batch_norm(signal), out_channels))


def bn_upconv_relu(signal, out_channels):
    return relu(upconv(batch_norm(signal), out_channels))
    