import os
import numpy as np
import argparse
from datetime import datetime
import cv2
import random
import math

import oneflow as flow
import oneflow.typing as tp

import networks
import image_pool

def random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

def load_image2ndarray(image_path, 
                       resize_and_crop = True,
                       load_size = 286,
                       crop_size = 256):
    im = cv2.imread(image_path)

    if resize_and_crop:
        im = cv2.resize(im, (load_size, load_size), interpolation = cv2.INTER_CUBIC)
        im = random_crop(im, crop_size, crop_size)
    else:
        im = cv2.resize(im, (crop_size, crop_size), interpolation = cv2.INTER_CUBIC)
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = ((im.astype(np.float32) / 255.0) - 0.5) / 0.5
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def ndarray2image(im):
    im = np.squeeze(im)
    im = (np.transpose(im, (1, 2, 0)) + 1) / 2.0 * 255.0
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)

def get_train_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config

def main(args):

    @flow.global_function("train", get_train_config())
    def TrainDiscriminator(
        real_A: tp.Numpy.Placeholder((1, 3, args.crop_size, args.crop_size), dtype = flow.float32),
        fake_A: tp.Numpy.Placeholder((1, 3, args.crop_size, args.crop_size), dtype = flow.float32),
        real_B: tp.Numpy.Placeholder((1, 3, args.crop_size, args.crop_size), dtype = flow.float32),
        fake_B: tp.Numpy.Placeholder((1, 3, args.crop_size, args.crop_size), dtype = flow.float32)):
        with flow.scope.placement("gpu", "0:0-0"):
            # Calculate GAN loss for discriminator D_A
            # Real
            pred_real_B = networks.define_D(real_B, "netD_A", ndf = args.ndf, n_layers_D = 3, trainable = True, reuse = True)
            loss_D_A_real = networks.GANLoss(pred_real_B, True)
            # Fake
            pred_fake_B = networks.define_D(fake_B, "netD_A", ndf = args.ndf, n_layers_D = 3, trainable = True, reuse = True)
            loss_D_A_fake = networks.GANLoss(pred_fake_B, False)
            # Combined loss and calculate gradients
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # Calculate GAN loss for discriminator D_B
            # Real
            pred_real_A = networks.define_D(real_A, "netD_B", ndf = args.ndf, n_layers_D = 3, trainable = True, reuse = True)
            loss_D_B_real = networks.GANLoss(pred_real_A, True)
            # Fake
            pred_fake_A = networks.define_D(fake_A, "netD_B", ndf = args.ndf, n_layers_D = 3, trainable = True, reuse = True)
            loss_D_B_fake = networks.GANLoss(pred_fake_A, False)
            # Combined loss and calculate gradients
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D = loss_D_A + loss_D_B

            flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [args.learning_rate]), beta1 = 0.5).minimize(loss_D)

        return loss_D

    @flow.global_function("train", get_train_config())
    def TrainGenerator(
        real_A: tp.Numpy.Placeholder((1, 3, args.crop_size, args.crop_size), dtype = flow.float32),
        real_B: tp.Numpy.Placeholder((1, 3, args.crop_size, args.crop_size), dtype = flow.float32)):
        with flow.scope.placement("gpu", "0:0-0"):
            # G_A(A)
            fake_B = networks.define_G(real_A, "netG_A", ngf = args.ngf, n_blocks = 9, trainable = True, reuse = True)
            # G_B(G_A(A))
            rec_A = networks.define_G(fake_B, "netG_B", ngf = args.ngf, n_blocks = 9, trainable = True, reuse = True)
            # G_B(B)
            fake_A = networks.define_G(real_B, "netG_B", ngf = args.ngf, n_blocks = 9, trainable = True, reuse = True)
            # G_A(G_B(B))
            rec_B = networks.define_G(fake_A, "netG_A", ngf = args.ngf, n_blocks = 9, trainable = True, reuse = True)

            # Identity loss
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = networks.define_G(real_B, "netG_A", ngf = args.ngf, n_blocks = 9, trainable = True, reuse = True)
            loss_idt_A = networks.L1Loss(idt_A - real_B) * args.lambda_B * args.lambda_identity

            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = networks.define_G(real_A, "netG_B", ngf = args.ngf, n_blocks = 9, trainable = True, reuse = True)
            loss_idt_B = networks.L1Loss(idt_B - real_A) * args.lambda_A * args.lambda_identity

            # GAN loss D_A(G_A(A))
            netD_A_out = networks.define_D(fake_B, "netD_A", ndf = args.ndf, n_layers_D = 3, trainable = False, reuse = True)
            loss_G_A = networks.GANLoss(netD_A_out, True)

            # GAN loss D_B(G_B(B))
            netD_B_out = networks.define_D(fake_A, "netD_B", ndf = args.ndf, n_layers_D = 3, trainable = False, reuse = True)
            loss_G_B = networks.GANLoss(netD_B_out, True)

            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = networks.L1Loss(rec_A - real_A) * args.lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = networks.L1Loss(rec_B - real_B) * args.lambda_B
            # combined loss and calculate gradients
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

            flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [args.learning_rate]), beta1 = 0.5).minimize(loss_G)

        return fake_B, rec_A, fake_A, rec_B, loss_G

    check_point = flow.train.CheckPoint()
    if args.checkpoint_load_dir != "":
        check_point.load(args.checkpoint_load_dir)
    else:
        check_point.init()

    datasetA = os.listdir(args.datasetA_path)
    datasetB = os.listdir(args.datasetB_path)

    datasetA_num = len(datasetA)
    datasetB_num = len(datasetB)
    print("dataset A size: %d" % datasetA_num)
    print("dataset B size: %d" % datasetB_num)

    train_iters = min(datasetA_num, datasetB_num)

    fake_A_pool = image_pool.ImagePool(50)  # create image buffer to store previously generated images
    fake_B_pool = image_pool.ImagePool(50)  # create image buffer to store previously generated images

    begin_epoch = 0
    for e in range(args.train_epoch):
        random.shuffle(datasetA)
        random.shuffle(datasetB)
        for i in range(train_iters):
            real_A = load_image2ndarray("%s/%s" % (args.datasetA_path, datasetA[i]), args.resize_and_crop, args.load_size, args.crop_size)
            real_B = load_image2ndarray("%s/%s" % (args.datasetB_path, datasetB[i]), args.resize_and_crop, args.load_size, args.crop_size)

            fake_B, rec_A, fake_A, rec_B, loss_G = TrainGenerator(real_A, real_B).get()

            fake_B = fake_B.numpy()
            fake_A = fake_A.numpy()

            fake_BB = fake_B_pool.query(fake_B)
            fake_AA = fake_A_pool.query(fake_A)
            loss_D = TrainDiscriminator(real_A, fake_AA, real_B, fake_BB).get()

            if i % 20 == 0:
                imageA = ndarray2image(real_A)
                imageB = ndarray2image(real_B)
                
                image_fake_B = ndarray2image(fake_B)
                image_rec_A = ndarray2image(rec_A.numpy())

                image_fake_A = ndarray2image(fake_A)
                image_rec_B = ndarray2image(rec_B.numpy())


                result1 = np.concatenate((imageA, image_fake_B, image_rec_A), axis = 1)
                result2 = np.concatenate((imageB, image_fake_A, image_rec_B), axis = 1)
                result = np.concatenate((result1, result2), axis = 0)

                cv2.imwrite(args.save_tmp_image_path, result)

                cur_g_loss = loss_G.numpy().mean()
                cur_d_loss = loss_D.numpy().mean()
                print("epoch: %d, iter: %d, gloss : %f, dloss : %f" % (e + begin_epoch, i, cur_g_loss, cur_d_loss))

                if i % 200 == 0:
                    check_point.save("%s/epoch_%d_iter_%d_gloss_%f_dloss_%f" % \
                        (args.checkpoint_save_dir, e + begin_epoch, i, cur_g_loss, cur_d_loss))
                

def get_parser(parser = None):
    parser = argparse.ArgumentParser("flags for cycle gan")

    parser.add_argument("--datasetA_path", type = str, default = "", help = "dataset A path")
    parser.add_argument("--datasetB_path", type = str, default = "", help = "dataset B path")

    # image preprocess
    parser.add_argument("--crop_size", type = int, default = 286)
    parser.add_argument("--load_size", type = int, default = 256)
    parser.add_argument("--resize_and_crop", type = bool, default = True)

    # checkpoint
    parser.add_argument("--checkpoint_load_dir", type = str, default = "", help = "load previous saved checkpoint from")
    parser.add_argument("--checkpoint_save_dir", type = str, default = "./checkpoints", help = "save checkpoint to")
    parser.add_argument("--save_tmp_image_path", type = str, default = 'train_temp_image.jpg', help = "image path")

    # hyper-parameters
    parser.add_argument("--train_epoch", type = int, default = 300)
    parser.add_argument("--learning_rate", type = float, default = 0.0002)
    parser.add_argument('--lambda_A', type=float, default = 10.0, help = 'weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default = 10.0, help = 'weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default = 0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--ngf', type = int, default = 64, help = '# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type = int, default = 64, help = '# of discrim filters in the first conv layer')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
