ROOT=$(cd "$(dirname $0)/.."; pwd)

# put your mxnet jar file in the $ROOT/lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-0.10.0-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/nu.pattern/opencv/jars/opencv-2.4.9-7.jar

PREAREIN_G_MODEL=/home/ldpe2g/MyOpenSource/Pix2Pix/models/checkpoints_edges2shoes_new_symbol_new_init/latest-netG-0009.params

INPUT_IMAGE=/home/ldpe2g/MyOpenSource/pix2pix/pix2pix/datasets/edges2shoes/val/2_AB.jpg

# -1 for cpu
GPU=0

# "AtoB" or "BtoA"
DIRECTION="AtoB"

OUTPUT_PATH=$ROOT/outputs

if [ ! -d $OUTPUT_PATH ] ; then
  mkdir -p $OUTPUT_PATH
fi

java -Xmx4G -cp $CLASS_PATH \
        TestPix2Pix \
        --gan-model-path $PREAREIN_G_MODEL \
        --input-image $INPUT_IMAGE \
        --output-path $OUTPUT_PATH \
        --gpu $GPU \
        --which-direction $DIRECTION
