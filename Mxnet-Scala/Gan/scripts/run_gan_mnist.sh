ROOT=$(cd "$(dirname $0)/.."; pwd)

data_path="${ROOT}/data"
if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

mnist_data_path="${ROOT}/data/mnist.zip"
if [ ! -f "$mnist_data_path" ]; then
  wget http://data.mxnet.io/mxnet/data/mnist.zip -P $data_path
  cd $data_path
  unzip -u mnist.zip
fi

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-INTERNAL.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/nu.pattern/opencv/jars/opencv-2.4.9-7.jar:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.6.2.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-simple/jars/slf4j-simple-1.6.2.jar

DATA_PATH="${data_path}"
GPU=0

java -Xmx4G -cp $CLASS_PATH \
	example.GanMnist $DATA_PATH $GPU
