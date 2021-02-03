import numpy as np
import argparse
import cv2

import oneflow as flow
import oneflow.typing as tp
import style_model

def float_list(x):
    return list(map(float, x.split(',')))

def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def recover_image(im):
    im = np.squeeze(im)
    im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)

def get_predict_config(device_type="gpu", device_num=1):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_placement_scope(
        flow.scope.placement(device_type, "0:0-{}".format(device_num - 1))
    )
    return func_config

def main(args):
    input_image = load_image(args.input_image_path)
    height = input_image.shape[2]
    width = input_image.shape[3]

    flow.env.init()

    @flow.global_function("predict", get_predict_config())
    def PredictNet(
        image: tp.Numpy.Placeholder((1, 3, height, width), dtype = flow.float32)) -> tp.Numpy:
        style_out = style_model.styleNet(image, trainable = True)
        return style_out

    flow.load_variables(flow.checkpoint.get(args.model_load_dir))

    import datetime
    a = datetime.datetime.now()

    style_out = PredictNet(input_image)

    b = datetime.datetime.now()
    c = b - a

    print("time: %s ms, height: %d, width: %d" % (c.microseconds / 1000, height, width))

    cv2.imwrite(args.output_image_path, recover_image(style_out))

def get_parser(parser = None):
    parser = argparse.ArgumentParser("flags for neural style")
    parser.add_argument("--input_image_path", type = str, default = 'test_img/tiger.jpg', help = "image path")
    parser.add_argument("--output_image_path", type = str, default = 'test_img/tiger.jpg', help = "image path")
    parser.add_argument("--model_load_dir", type = str, default = "", help = "model save directory")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
