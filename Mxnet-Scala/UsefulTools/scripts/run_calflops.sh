ROOT=$(cd "$(dirname $0)/.."; pwd)

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-1.3.1-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar\

# for multiple input, for example data1,data2
# DATA_SHAPES = "data1,1,3,224,224 data2,1,3,224,224"
DATA_SHAPES="data,1,3,224,224"
LABEL_SHAPES="softmax_label,1"

# caffenet-symbol.json squeezenet_v1.0-symbol.json
SYMBOL=$ROOT/models/resnet-101-symbol.json

java -Xmx1G -cp $CLASS_PATH \
	tools.CalFlops \
	--symbol $SYMBOL \
	--ds $DATA_SHAPES  \
	--ls $LABEL_SHAPES
