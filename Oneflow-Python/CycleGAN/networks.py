import oneflow as flow
import oneflow.python.framework.id_util as id_util

def instance_norm(input, name_prefix, trainable = True, reuse = True):
    (mean, variance) = flow.nn.moments(input, [2, 3], keepdims = True)
    gamma = flow.get_variable(
        name_prefix + "_gamma",
        shape = (1, input.shape[1], 1, 1),
        dtype=input.dtype,
        initializer = flow.ones_initializer(),
        trainable = trainable,
        reuse = reuse
    )
    beta = flow.get_variable(
        name_prefix + "_beta",
        shape = (1, input.shape[1], 1, 1),
        dtype=input.dtype,
        initializer = flow.zeros_initializer(),
        trainable = trainable,
        reuse = reuse
    )
    epsilon = 1e-3
    normalized = (input - mean) / flow.math.sqrt(variance + epsilon)
    return gamma * normalized + beta

def conv2d_layer(
    name,
    input,
    out_channel,
    kernel_size = 3,
    strides = 1,
    padding = "SAME", # or [[], [], [], []]
    data_format = "NCHW",
    dilation_rate = 1,
    use_bias = True,
    weight_initializer = flow.random_normal_initializer(mean = 0.0, stddev = 0.02),
    bias_initializer = flow.zeros_initializer(),
    trainable = True,
    reuse = True 
):   
    weight_shape = (out_channel, input.shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "_weight",
        shape = weight_shape,
        dtype = input.dtype,
        initializer = weight_initializer,
        trainable = trainable,
        reuse = reuse
    )
    output = flow.nn.conv2d(input, weight, strides, padding, data_format, dilation_rate)
    if use_bias:
        bias = flow.get_variable(
            name + "_bias",
            shape = (out_channel,),
            dtype = input.dtype,
            initializer = bias_initializer,
            trainable = trainable
        )
        output = flow.nn.bias_add(output, bias, data_format)
    return output

def resBlock(input, channel, name_prefix, trainable = True, reuse = True):
    out = conv2d_layer(name_prefix + "_conv1", input, channel, kernel_size = 3, strides = 1, trainable = trainable, reuse = reuse)
    out = instance_norm(out, name_prefix + "_in1", trainable = trainable, reuse = reuse)
    out = flow.nn.relu(out)
    out = conv2d_layer(name_prefix + "_conv2", out, channel, kernel_size = 3, strides = 1, trainable = trainable, reuse = reuse)
    out = instance_norm(out, name_prefix + "_in2", trainable = trainable, reuse = reuse)
    return out + input

def deconv(input, out_channel, name_prefix, kernel_size = 4, strides = [2, 2], trainable = True, reuse = True):
    weight = flow.get_variable(
        name_prefix + "_weight",
        shape = (input.shape[1], out_channel, kernel_size, kernel_size),
        dtype = flow.float,
        initializer = flow.random_normal_initializer(mean = 0.0, stddev = 0.02),
        trainable = trainable,
        reuse = reuse
    )
    return flow.nn.conv2d_transpose(
                input,
                weight,
                strides = strides,
                padding = "SAME",
                output_shape = (input.shape[0], out_channel, input.shape[2] * strides[0], input.shape[3] * strides[1]))

def define_G(input, net_name_prefix, ngf = 64, n_blocks = 9, trainable = True, reuse = True):
    with flow.scope.namespace(net_name_prefix):
        conv = conv2d_layer("conv_0", input, ngf, kernel_size = 7, strides = 1, trainable = trainable, reuse = reuse)
        conv = instance_norm(conv, "ins_0", trainable = trainable, reuse = reuse)
        conv = flow.math.relu(conv)

        n_downsampling = 2
        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            conv = conv2d_layer("conv_s2_%d" % i, conv, ngf * mult * 2, kernel_size = 3, strides = 2, trainable = trainable, reuse = reuse)
            conv = instance_norm(conv, "ins_s2_%d" % i, trainable = trainable, reuse = reuse)
            conv = flow.math.relu(conv)
        
        mult = 2 ** n_downsampling
        # add ResNet blocks
        for i in range(n_blocks):       
            conv = resBlock(conv, ngf * mult, "resblock_%d" % i, trainable = trainable, reuse = reuse)
        
        # add upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            conv = deconv(conv, int(ngf * mult / 2), "deconv_u2_%d" % i, kernel_size = 3, strides = [2, 2], trainable = trainable, reuse = reuse)
            conv = instance_norm(conv, "ins_u2_%d" % i, trainable = trainable, reuse = reuse)
            conv = flow.math.relu(conv)

        conv = conv2d_layer("last_conv", conv, 3, kernel_size = 7, strides = 1, trainable = trainable, reuse = reuse)
        out = flow.math.tanh(conv)
    return out

def define_D(input, net_name_prefix, ndf = 64, n_layers_D = 3, trainable = True, reuse = True):
    with flow.scope.namespace(net_name_prefix):
        kernel_size = 4
        padding = [[0, 0], [0, 0], [1, 1], [1, 1]]
        conv = conv2d_layer("conv_0", input, ndf, kernel_size = kernel_size, strides = 2, padding = padding, trainable = trainable, reuse = reuse)
        conv = flow.nn.leaky_relu(conv, 0.2)

        for n in range(1, n_layers_D):
            nf_mult = min(2 ** n, 8)
            conv = conv2d_layer("conv_%d" % n, conv, ndf * nf_mult, kernel_size = kernel_size, strides = 2, padding = padding, trainable = trainable, reuse = reuse)
            conv = instance_norm(conv, "ins_%d" % n, trainable = trainable, reuse = reuse)
            conv = flow.nn.leaky_relu(conv, 0.2)

        nf_mult = min(2 ** n_layers_D, 8)
        conv = conv2d_layer("conv_%d" % n_layers_D, conv, ndf * nf_mult, kernel_size = kernel_size, strides = 1, padding = padding, trainable = trainable, reuse = reuse)
        conv = instance_norm(conv, "ins_%d" % n_layers_D, trainable = trainable, reuse = reuse)
        conv = flow.nn.leaky_relu(conv, 0.2)

        out = conv2d_layer("last_conv", conv, 1, kernel_size = kernel_size, strides = 1, padding = padding, trainable = trainable, reuse = reuse)

    return out

def L1Loss(input):
    return flow.math.reduce_mean(flow.math.abs(input))

def GANLoss(input, target_is_real, target_real_label = 1.0, target_fake_label = 0.0):
    if target_is_real:
        target = flow.constant_like(input, target_real_label)
    else:
        target = flow.constant_like(input, target_fake_label)
    return flow.math.reduce_mean(flow.math.square(input - target))
