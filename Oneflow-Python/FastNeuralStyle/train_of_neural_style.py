import os
import numpy as np
import argparse
from datetime import datetime
import cv2
import random

import oneflow as flow
import oneflow.typing as tp
import vgg16_model
import style_model

CONSOLE_ARGUMENTS = None

def float_list(x):
    return list(map(float, x.split(',')))

def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.resize(im, (CONSOLE_ARGUMENTS.train_image_size, CONSOLE_ARGUMENTS.train_image_size))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def recover_image(im):
    im = np.squeeze(im)
    im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)

def get_train_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config

def get_predict_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config

def main(args):
    global CONSOLE_ARGUMENTS
    CONSOLE_ARGUMENTS = args

    @flow.global_function("train", get_train_config())
    def TrainNet(
        image: tp.Numpy.Placeholder((1, 3, CONSOLE_ARGUMENTS.train_image_size, CONSOLE_ARGUMENTS.train_image_size), dtype = flow.float32),
        mean: tp.Numpy.Placeholder((1, 3, 1, 1), dtype = flow.float32),
        std: tp.Numpy.Placeholder((1, 3, 1, 1), dtype = flow.float32),
        style_image_relu1_2: tp.Numpy.Placeholder((1, 64, CONSOLE_ARGUMENTS.train_image_size, CONSOLE_ARGUMENTS.train_image_size), dtype = flow.float32),
        style_image_relu2_2: tp.Numpy.Placeholder((1, 128, CONSOLE_ARGUMENTS.train_image_size // 2, CONSOLE_ARGUMENTS.train_image_size // 2), dtype = flow.float32),
        style_image_relu3_3: tp.Numpy.Placeholder((1, 256, CONSOLE_ARGUMENTS.train_image_size // 4, CONSOLE_ARGUMENTS.train_image_size // 4), dtype = flow.float32),
        style_image_relu4_3: tp.Numpy.Placeholder((1, 512, CONSOLE_ARGUMENTS.train_image_size // 8, CONSOLE_ARGUMENTS.train_image_size // 8), dtype = flow.float32), 
    ):
        with flow.scope.placement("gpu", "0:0-0"):
            style_out = style_model.styleNet(image, trainable = True)

            image_norm = (image - mean) / std
            org_content_relu2_2 = vgg16_model.vgg16bn_content_layer(image_norm, trainable = False, training = False)
            
            style_out_norm = (style_out - mean) / std
            style_out_relu1_2, style_out_relu2_2, style_out_relu3_3, style_out_relu4_3 = vgg16_model.vgg16bn_style_layer(style_out_norm, trainable = False, training = False)

            # compute mean square error loss
            content_loss = style_model.mse_loss(org_content_relu2_2 - style_out_relu2_2)
            style_loss = style_model.mse_loss(style_model.gram_matrix(style_out_relu1_2) - style_model.gram_matrix(style_image_relu1_2)) \
                        + style_model.mse_loss(style_model.gram_matrix(style_out_relu2_2) - style_model.gram_matrix(style_image_relu2_2)) \
                        + style_model.mse_loss(style_model.gram_matrix(style_out_relu3_3) - style_model.gram_matrix(style_image_relu3_3)) \
                        + style_model.mse_loss(style_model.gram_matrix(style_out_relu4_3) - style_model.gram_matrix(style_image_relu4_3))

            loss = content_loss * CONSOLE_ARGUMENTS.content_weight + style_loss * CONSOLE_ARGUMENTS.style_weight

            flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [CONSOLE_ARGUMENTS.learning_rate])).minimize(loss)

        return style_out, loss

    @flow.global_function("predict", get_predict_config())
    def getVgg16MiddleLayers(
        style_image: tp.Numpy.Placeholder((1, 3, CONSOLE_ARGUMENTS.train_image_size, CONSOLE_ARGUMENTS.train_image_size), dtype = flow.float32),
        mean: tp.Numpy.Placeholder((1, 3, 1, 1), dtype = flow.float32),
        std: tp.Numpy.Placeholder((1, 3, 1, 1), dtype = flow.float32)):
        with flow.scope.placement("gpu", "0:0-0"):
            style_image = (style_image - mean) / std
            style_out_relu1_2, style_out_relu2_2, style_out_relu3_3, style_out_relu4_3 = vgg16_model.vgg16bn_style_layer(style_image, trainable = False, training = False)
        return style_out_relu1_2, style_out_relu2_2, style_out_relu3_3, style_out_relu4_3

    check_point = flow.train.CheckPoint()
    check_point.load(CONSOLE_ARGUMENTS.model_load_dir)

    mean_nd = np.array(float_list(CONSOLE_ARGUMENTS.rgb_mean)).reshape((1, 3, 1, 1)).astype(np.float32)
    std_nd = np.array(float_list(CONSOLE_ARGUMENTS.rgb_std)).reshape((1, 3, 1, 1)).astype(np.float32)

    # prepare style image vgg16 middle layer outputs
    style_image = load_image(CONSOLE_ARGUMENTS.style_image_path)
    style_image_recover = recover_image(style_image)

    style_image_relu1_2, style_image_relu2_2, style_image_relu3_3, style_image_relu4_3 = \
        getVgg16MiddleLayers(style_image, mean_nd, std_nd).get()

    style_image_relu1_2 = style_image_relu1_2.numpy()
    style_image_relu2_2 = style_image_relu2_2.numpy()
    style_image_relu3_3 = style_image_relu3_3.numpy()
    style_image_relu4_3 = style_image_relu4_3.numpy()

    train_images = os.listdir(CONSOLE_ARGUMENTS.dataset_path)
    random.shuffle(train_images)
    images_num = len(train_images)
    print("dataset size: %d" % images_num)

    for e in range(CONSOLE_ARGUMENTS.train_epoch):
        for i in range(images_num):
            image = load_image("%s/%s" % (CONSOLE_ARGUMENTS.dataset_path, train_images[i]))
            style_out, loss = TrainNet(image, mean_nd, std_nd, style_image_relu1_2, style_image_relu2_2, style_image_relu3_3, style_image_relu4_3).get()

            if i % 100 == 0:
                image_recover = recover_image(image)
                style_out_recover = recover_image(style_out.numpy())
                result = np.concatenate((style_image_recover, image_recover), axis=1)
                result = np.concatenate((result, style_out_recover), axis=1)
                cv2.imwrite(CONSOLE_ARGUMENTS.save_tmp_image_path, result)

                cur_loss = loss.numpy().mean()
                
                check_point.save("%s/lr_%f_cw_%f_sw_%f_epoch_%d_iter_%d_loss_%f" % \
                    (CONSOLE_ARGUMENTS.model_save_dir, CONSOLE_ARGUMENTS.learning_rate, CONSOLE_ARGUMENTS.content_weight, CONSOLE_ARGUMENTS.style_weight, e, i, cur_loss))

                print("epoch: %d, iter: %d, loss : %f" % (e, i, cur_loss))


def get_parser(parser = None):
    parser = argparse.ArgumentParser("flags for neural style")

    parser.add_argument("--dataset_path", type = str, default = './Coco/test2015', help = "dataset path")
    parser.add_argument("--style_image_path", type = str, default = 'test_img/tiger.jpg', help = "image path")
    parser.add_argument("--save_tmp_image_path", type = str, default = 'images/train_temp_image.jpg', help = "image path")

    # for data process
    parser.add_argument('--rgb_mean', type = str, default = "123.68, 116.779, 103.939",
                        help = 'a tuple of size 3 for the mean rgb')
    parser.add_argument('--rgb_std', type = str, default = "58.393, 57.12, 57.375",
                        help = 'a tuple of size 3 for the std rgb')

    # snapshot
    parser.add_argument("--model_load_dir", type = str,
                        default = "./output/snapshots/model_save-{}".format(
                            str(datetime.now().strftime("%Y%m%d%H%M%S"))),
                        help = "model save directory",
                        )
    parser.add_argument("--model_save_dir", type = str, default = "./checkpoints", help = "model save directory")

    # training hyper-parameters
    parser.add_argument("--train_epoch", type = int, default = 2)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--content_weight", type = float, default = 1)
    parser.add_argument("--style_weight", type = float, default = 50)
    parser.add_argument("--train_image_size", type = int, default = 224)

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
