import numpy as np
import argparse
import cv2
import os

import oneflow as flow
import oneflow.typing as tp

import networks

def load_image2ndarray(image_path, resize):
    im = cv2.imread(image_path)
    height, width, channels = im.shape
    im = cv2.resize(im, (resize, resize), interpolation = cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = ((im.astype(np.float32) / 255.0) - 0.5) / 0.5
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32'), height, width

def ndarray2image(im):
    im = np.squeeze(im)
    im = (np.transpose(im, (1, 2, 0)) + 1) / 2.0 * 255.0
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)

def get_test_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config

def main(args):

    netG_name = "netG_A" if args.direction == "A2B" else "netG_B"

    @flow.global_function("predict", get_test_config())
    def TestGenerator(
        real: tp.Numpy.Placeholder((1, 3, args.network_input_size, args.network_input_size), dtype = flow.float32)) -> tp.Numpy:
        with flow.scope.placement("gpu", "0:0-0"):
            fake = networks.define_G(real, netG_name, ngf = 64, n_blocks = 9, trainable = False, reuse = True)
        return fake

    check_point = flow.train.CheckPoint()
    assert args.checkpoint_load_dir != ""
    check_point.load(args.checkpoint_load_dir)

    in_images = os.listdir(args.input_images)

    for i in in_images:
        input_image, org_height, org_width = load_image2ndarray("%s/%s" % (args.input_images, i), args.network_input_size)
        output = ndarray2image(TestGenerator(input_image))
        output = cv2.resize(output, (org_width, org_height), interpolation = cv2.INTER_CUBIC)
        output = np.concatenate((ndarray2image(input_image), output), axis = 1)
        cv2.imwrite("%s/%s" % (args.output_images, i), output)

def get_parser(parser = None):
    parser = argparse.ArgumentParser("flags for test CycleGan")

    parser.add_argument("--checkpoint_load_dir", type = str, default = "", help = "load previous saved checkpoint from.")
    parser.add_argument("--input_images", type = str, default = "", help = "")
    parser.add_argument("--output_images", type = str, default = "", help = "")
    parser.add_argument("--network_input_size", type = int, default = 256, help = "")
    parser.add_argument("--direction", type = str, default = "A2B", help = "'A2B' or 'B2A' .Transform image from domain A to domain B or reverse.")

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
