ROOT=$(cd "$(dirname $0)/.."; pwd)

# put your mxnet jar file in the $ROOT/lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-1.0.1-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/nu.pattern/opencv/jars/opencv-2.4.9-7.jar


# pretrain models are under the $ROOT/datas/pretrain_models directory
PREAREIN_G_MODEL=$ROOT/models/pretrain_models/

INPUT_IMAGE=

# -1 for cpu
GPU=0

# "AtoB" or "BtoA"
DIRECTION="BtoA"

OUTPUT_PATH=$ROOT/outputs

if [ ! -d $OUTPUT_PATH ] ; then
  mkdir -p $OUTPUT_PATH
fi

java -Xmx4G -cp $CLASS_PATH \
        TestCycleGan \
        --gan-model-path $PREAREIN_G_MODEL \
        --input-image $INPUT_IMAGE \
        --output-path $OUTPUT_PATH \
        --gpu $GPU \
        --which-direction $DIRECTION
