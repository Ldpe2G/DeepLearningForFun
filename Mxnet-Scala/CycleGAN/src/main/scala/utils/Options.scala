package utils

import org.kohsuke.args4j.Option

/**
 * @author Depeng Liang
 */
object Options {

  class OptTrain {
    @Option(name = "--domain-a-path", usage = "the domain A images path")
    var domainAPath: String = null
    @Option(name = "--domain-b-path", usage = "the domain B images path")
    var domainBPath: String = null
    @Option(name = "--scalesize", usage = "scale images to this size")
    var scaleSize: Int = 143
    @Option(name = "--cropsize", usage = "crop images to this size")
    var cropSize: Int = 128
    @Option(name = "--ngf", usage = "of gen filters in first conv layer")
    var ngf: Int = 64
    @Option(name = "--ndf", usage = "of discrim filters in first conv layer")
    var ndf: Int = 64
    @Option(name = "--input-nc", usage = "of input image channels")
    var inputNC: Int = 3
     @Option(name = "--output-nc", usage = "of output image channels")
    var outputNC: Int = 3
    @Option(name = "--niter", usage = "of iter at starting learning rate")
    var niter: Int = 100
    @Option(name = "--niter-decay", usage = "of iter to linearly decay learning rate to zero")
    var niterDecay: Int = 100
    @Option(name = "--lr", usage = "the initial learning rate for adam")
    var lr: Float = 0.0002f
    @Option(name = "--beta1", usage = "momentum term of adam")
    var beta1: Float = 0.5f
    @Option(name = "--flip", usage = "if flip the images for data argumentation, 1 means flip 0 means not")
    var flip: Int = 1
    @Option(name = "--display-freq", usage = "display the current results every display_freq iterations")
    var displayFreq: Int = 20
    @Option(name = "--save-epoch-freq", usage = "save a model every save_epoch_freq epochs (does not overwrite previously saved models)")
    var saveEpochFreq: Int = 1
    @Option(name = "--save-latest-freq", usage = "save the latest model every latest_freq sgd iterations (overwrites the previous latest model)")
    var saveLatestFreq: Int = 1000
    @Option(name = "--print-freq", usage = "print the debug information every prinFreq iterations")
    var printFreq: Int = 50
    @Option(name = "--checkpoints-dir", usage = "models are saved here")
    var checkpointsDir: String = null
    @Option(name = "--norm", usage = " 'batch' or 'instance' normalization")
    var norm: String = "batch"
    @Option(name = "--lambda-A", usage = "weight for cycle loss (A -> B -> A)")
    var lambdaA: Float = 10f
    @Option(name = "--lambda-B", usage = "weight for cycle loss (B -> A -> B)")
    var lambdaB: Float = 10f
    @Option(name = "--pool-size", usage = "the size of image buffer that stores previously generated images")
    var poolSize: Int = 50
    @Option(name = "--gpu", usage = "which gpu card to use 0,1,2, default is -1, means using cpu")
    var gpu: Int = -1
    @Option(name = "--batchsize", usage = "training batch size, only support batchsize 1")
    var batchSize: Int = 1
    @Option(name = "--load-checkpoints-dir", usage = "path to load previous saved models to continue training")
    var loadCheckpointsDir: String = null
    @Option(name = "--load-checkpoints-epoch", usage = "the epoch num of the load models")
    var loadCheckpointsEpoch: Int = 0
    @Option(name = "--identity", usage = "use identity mapping. Setting opt.identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set opt.identity = 0.1")
    var identity: Float = 0.01f
  }

  class OptTest {
    @Option(name = "--gan-model-path", usage = "the pretrain g model")
    var modelPath: String = null
    @Option(name = "--input-image", usage = "the input image")
    var inputImage: String = null
    @Option(name = "--output-nc", usage = "of output image channels")
    var outputNC: Int = 3
    @Option(name = "--ngf", usage = "of gen filters in first conv layer")
    var ngf: Int = 64
    @Option(name = "--ndf", usage = "of discrim filters in first conv layer")
    var ndf: Int = 64
    @Option(name = "--output-path", usage = "the output result path")
    var outputPath: String = null
    @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
    var gpu: Int = -1
    @Option(name = "--norm", usage = " 'batch' or 'instance' normalization")
    var norm: String = "batch"
    @Option(name = "--which-direction", usage = "'AtoB'  or 'BtoA'")
    var whichDirection: String = "BtoA"
  }
}
