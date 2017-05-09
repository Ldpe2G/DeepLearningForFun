ROOT=$(cd "$(dirname $0)/.."; pwd)

# put your mxnet jar file in the $ROOT/lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-0.9.5-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/com.sksamuel.scrimage/scrimage-core_2.11/jars/scrimage-core_2.11-2.1.7.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.7.7.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-core/jars/imageio-core-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.common/common-lang/jars/common-lang-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.common/common-io/jars/common-io-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.common/common-image/jars/common-image-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-jpeg/jars/imageio-jpeg-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-metadata/jars/imageio-metadata-3.2.1.jar:\
$HOME/.ivy2/cache/com.drewnoakes/metadata-extractor/jars/metadata-extractor-2.8.1.jar:\
$HOME/.ivy2/cache/com.adobe.xmp/xmpcore/jars/xmpcore-5.1.2.jar:\
$HOME/.ivy2/cache/commons-io/commons-io/jars/commons-io-2.4.jar:\
$HOME/.ivy2/cache/ar.com.hjg/pngj/jars/pngj-2.1.0.jar:\
$HOME/.ivy2/cache/com.sksamuel.scrimage/scrimage-filters_2.11/jars/scrimage-filters_2.11-2.1.7.jar:\
$HOME/.ivy2/cache/com.sksamuel.scrimage/scrimage-io-extra_2.11/jars/scrimage-io-extra_2.11-2.1.7.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-bmp/jars/imageio-bmp-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-icns/jars/imageio-icns-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-iff/jars/imageio-iff-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-pcx/jars/imageio-pcx-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-pict/jars/imageio-pict-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-pdf/jars/imageio-pdf-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-pnm/jars/imageio-pnm-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-psd/jars/imageio-psd-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-sgi/jars/imageio-sgi-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-tiff/jars/imageio-tiff-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-tga/jars/imageio-tga-3.2.1.jar:\
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-thumbsdb/jars/imageio-thumbsdb-3.2.1.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/org.sameersingh.scalaplot/scalaplot/jars/scalaplot-0.0.4.jar:\
$HOME/.ivy2/cache/jfree/jfreechart/jars/jfreechart-1.0.13.jar:\
$HOME/.ivy2/cache/jfree/jcommon/jars/jcommon-1.0.16.jar:\
$HOME/.ivy2/cache/com.itextpdf/itextpdf/jars/itextpdf-5.1.2.jar:\
$HOME/.ivy2/cache/com.itextpdf.tool/xmlworker/jars/xmlworker-1.1.0.jar

# pretrain models are under the $ROOT/datas/pretrain_models directory
PREAREIN_MODEL=$ROOT/datas/pretrain_models/la_muse/resdual_0000-0030000.params

INPUT_IMAGE=$ROOT/datas/images/chicago.jpg

OUTPUT_PATH=$ROOT/datas/output


if [ ! -d $OUTPUT_PATH ] ; then
	mkdir -p $OUTPUT_PATH
fi

# -1 for cpu
GPU=0

java -Xmx1G -cp $CLASS_PATH \
	FastNeuralStyle \
	--model-path  $PREAREIN_MODEL \
	--input-image $INPUT_IMAGE \
	--output-path $OUTPUT_PATH \
	--gpu $GPU
