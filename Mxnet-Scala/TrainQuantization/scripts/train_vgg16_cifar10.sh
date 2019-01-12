ROOT=$(cd "$(dirname $0)/.."; pwd)

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-1.5.0-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.6.2.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-simple/jars/slf4j-simple-1.6.2.jar

DATA_PATH="${ROOT}/data"
if [ ! -d "$DATA_PATH" ]; then
  mkdir -p "$DATA_PATH"
fi

cifar_data_path="${ROOT}/data/cifar"
if [ ! -d "$cifar_data_path" ]; then
  wget http://data.dmlc.ml/mxnet/data/cifar10.zip -P $DATA_PATH
  cd $DATA_PATH
  unzip -u cifar10.zip
fi

BATCH_SIZE=128
FINETUNE_MODEL_EPOCH=-1
FINETUNE_MODEL_PREFIX=$ROOT/models/
# FINETUNE_MODEL_EPOCH=44
# FINETUNE_MODEL_PREFIX=$ROOT/models/cifar10_vgg16_acc_0.8771034
SAVE_MODEL_PATH=$ROOT/models/cifar10_vgg16
GPU=0
LR=0.01
TRAIN_EPOCH=100000

java -Xmx4G -cp $CLASS_PATH \
	TrainVGG \
	--batch-size $BATCH_SIZE \
	--data-path $DATA_PATH/cifar \
	--finetune-model-epoch $FINETUNE_MODEL_EPOCH \
	--finetune-model-prefix $FINETUNE_MODEL_PREFIX \
	--save-model-path $SAVE_MODEL_PATH \
	--gpu $GPU \
	--lr $LR \
	--train-epoch $TRAIN_EPOCH
