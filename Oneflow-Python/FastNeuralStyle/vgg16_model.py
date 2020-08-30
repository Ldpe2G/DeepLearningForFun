import oneflow as flow
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util

def _batch_norm(inputs, name, trainable=True, training=True):
    params_shape = [inputs.shape[1]]
    # Float32 required to avoid precision-loss when using fp16 input/output
    params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype

    if not flow.current_global_function_desc().IsTrainable() or not trainable:
        training = False

    with flow.scope.namespace(name):
        beta = flow.get_variable(
            name="beta",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.zeros_initializer(),
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )

        gamma = flow.get_variable(
            name="gamma",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.ones_initializer(),
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )

        moving_mean = flow.get_variable(
            name="moving_mean",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.zeros_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
        )

        moving_variance = flow.get_variable(
            name="moving_variance",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.ones_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
        )
    builder = (
        flow.user_op_builder(id_util.UniqueStr(name))
        .Op("normalization")
        .Input("x", [inputs])
        .Input("moving_mean", [moving_mean])
        .Input("moving_variance", [moving_variance])
        .Input("gamma", [gamma])
        .Input("beta", [beta])
        .Output("y")
        .Attr("axis", 1)
        .Attr("epsilon", 1.001e-5)
        .Attr("training", training)
        .Attr("momentum", 0.997)
    )
    if trainable and training:
        builder = builder.Output("mean").Output("inv_variance")
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


def conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=True,
    weight_initializer=flow.variance_scaling_initializer(2, 'fan_out', 'random_normal', data_format="NCHW"),
    bias_initializer=flow.zeros_initializer(),
    bn=True,
    trainable=True, 
    training=True
):   
    weight_shape = (filters, input.shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "_weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        trainable=trainable
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate
    )
    if use_bias:
        bias = flow.get_variable(
            name + "_bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            trainable=trainable
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            if bn:
                output = _batch_norm(output, name + "_bn", trainable, training)
                output = flow.nn.relu(output)
            else:
                output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output

def _conv_block(in_blob, index, filters, conv_times, trainable, training):
    conv_block = in_blob
    for i in range(conv_times):
        conv_block = conv2d_layer(
            name="conv{}".format(index),
            input=conv_block,
            filters=filters,
            kernel_size=3,
            strides=1,
            bn=True,
            trainable=trainable, 
            training=training
        )
        index += 1
    return conv_block

def vgg16bn_content_layer(images, trainable=True, training=True):
    conv1 = _conv_block(images, 0, 64, 2, trainable, training)
    pool1 = flow.nn.max_pool2d(conv1, 2, 2, "VALID", "NCHW")
    conv2 = _conv_block(pool1, 2, 128, 2, trainable, training)
    return conv2

def vgg16bn_style_layer(images, trainable=True, training=True):
    conv1 = _conv_block(images, 0, 64, 2, trainable, training)
    pool1 = flow.nn.max_pool2d(conv1, 2, 2, "VALID", "NCHW")
    conv2 = _conv_block(pool1, 2, 128, 2, trainable, training)
    pool2 = flow.nn.max_pool2d(conv2, 2, 2, "VALID", "NCHW")
    conv3 = _conv_block(pool2, 4, 256, 3, trainable, training)
    pool3 = flow.nn.max_pool2d(conv3, 2, 2, "VALID", "NCHW")
    conv4 = _conv_block(pool3, 7, 512, 3, trainable, training)
    return conv1, conv2, conv3, conv4
