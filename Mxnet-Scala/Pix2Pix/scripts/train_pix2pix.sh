ROOT=$(cd "$(dirname $0)/.."; pwd)

# put your mxnet jar file in the $ROOT/lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-1.2.0-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/nu.pattern/opencv/jars/opencv-2.4.9-7.jar

# continue training with "--load-checkpoints-dir" and  "--load-checkpoints-epoch" command line parameters
LOAD_CHECKPOINTS_DIR=
LOAD_CHECKPOINTS_EPOCH=

DATA_PATH=/home/ldpe2g/MyOpenSource/pix2pix/pix2pix/datasets/cityscapes/train

# 'AtoB' or 'BtoA'
DIRECTION="BtoA"

# -1 for cpu
GPU=0
SCALE_SIZE=286
CROP_SIZE=256
NGF=64
NDF=64
INPUT_NC=3
OUTPUT_NC=3
NITER=400
LEARNING_RATE=0.0002
BETA1=0.5
FLIP=1
DISPLAY_FREQ=20
SAVE_EPOCH_FREQ=1
SAVE_LASTESR_FRWQ=1000
PRINT_FREQ=50
LAMBDA=100
# only support batch size 1
BATCH_SIZE=1

CHECKPOINT_DIR=$ROOT/models/checkpoints
if [ ! -d $CHECKPOINT_DIR ] ; then
  mkdir -p $CHECKPOINT_DIR
fi

java -Xmx4G -cp $CLASS_PATH \
        TrainPix2Pix \
        --gpu $GPU \
        --data-path $DATA_PATH \
        --checkpoints-dir $CHECKPOINT_DIR \
        --scalesize $SCALE_SIZE \
        --cropsize $CROP_SIZE \
        --ngf $NGF \
        --ndf  $NDF \
        --input-nc  $INPUT_NC \
        --output-nc $OUTPUT_NC \
        --niter $NITER \
        --lr  $LEARNING_RATE \
        --beta1  $BETA1 \
        --flip  $FLIP \
        --display-freq  $DISPLAY_FREQ \
        --save-epoch-freq  $SAVE_EPOCH_FREQ \
        --save-latest-freq  $SAVE_LASTESR_FRWQ \
        --print-freq  $PRINT_FREQ \
        --lambda $LAMBDA \
        --gpu $GPU  \
        --batchsize  $BATCH_SIZE \
        --which-direction $DIRECTION
