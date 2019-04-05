ROOT=$(cd "$(dirname $0)/.."; pwd)

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-INTERNAL.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.11.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/org.scala-lang.modules/scala-parser-combinators_2.11/bundles/scala-parser-combinators_2.11-1.0.4.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.6.2.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-simple/jars/slf4j-simple-1.6.2.jar

# for multiple input, for example data1,data2
# DATA_SHAPES = "data1,1,3,224,224 data2,1,3,224,224"
DATA_SHAPES="data,1,3,384,384"
LABEL_SHAPES="softmax_label,1,21,384,384"
SYMBOL=$ROOT/models/fcn32s-symbol.json


# DATA_SHAPES="data,1,3,224,224"
# LABEL_SHAPES="softmax_label,1"
# SYMBOL=$ROOT/models/resnet-101-symbol.json


java -Xmx24G -cp $CLASS_PATH \
	tools.CalFlops \
	--symbol $SYMBOL \
	--ds $DATA_SHAPES  \
	--ls $LABEL_SHAPES
