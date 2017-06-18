ROOT=$(cd "$(dirname $0)/.."; pwd)

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-0.10.1-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar

# path to the mnist dataset
TRAIN_DATA_PATH=$ROOT/datas/mnist

# -1 for cpu
GPU=0

BATCH_SIZE=100

LR=0.001

TRAIN_EPOCH=15

java -Xmx1G -cp $CLASS_PATH \
	snns.SNNs_MLP_MNIST \
	--batch-size $BATCH_SIZE \
	--data-path $TRAIN_DATA_PATH \
	--gpu $GPU \
	--lr $LR \
	--train-epoch $TRAIN_EPOCH
