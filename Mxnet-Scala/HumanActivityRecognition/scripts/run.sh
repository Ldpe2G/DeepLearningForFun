ROOT=$(cd "$(dirname $0)/.."; pwd)

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-INTERNAL.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/org.sameersingh.scalaplot/scalaplot/jars/scalaplot-0.0.4.jar:\
$HOME/.ivy2/cache/jfree/jfreechart/jars/jfreechart-1.0.13.jar:\
$HOME/.ivy2/cache/jfree/jcommon/jars/jcommon-1.0.16.jar:\
$HOME/.ivy2/cache/com.itextpdf/itextpdf/jars/itextpdf-5.1.2.jar:\
$HOME/.ivy2/cache/com.itextpdf.tool/xmlworker/jars/xmlworker-1.1.0.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.6.2.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-simple/jars/slf4j-simple-1.6.2.jar

if [ ! -d $ROOT/data/UCI_HAR_Dataset ] ; then
	cd $ROOT/data/
	tar -zxf UCI_HAR_Dataset.tar.gz
	cd -
fi

DATA_PATH=$ROOT/data/UCI_HAR_Dataset

# -1 for cpu
GPU=0

java -Xmx1G -cp $CLASS_PATH \
	HumanAR $DATA_PATH $GPU
