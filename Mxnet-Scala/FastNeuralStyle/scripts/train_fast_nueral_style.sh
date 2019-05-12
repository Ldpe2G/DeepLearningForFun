ROOT=$(cd "$(dirname $0)/.."; pwd)

# put your mxnet jar file in the $ROOT/lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-INTERNAL.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:\
$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/com.sksamuel.scrimage/scrimage-core_2.11/jars/scrimage-core_2.11-2.1.7.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-api/jars/slf4j-api-1.6.2.jar:\
$HOME/.ivy2/cache/org.slf4j/slf4j-simple/jars/slf4j-simple-1.6.2.jar:\
$HOME/.ivy2/cache/nu.pattern/opencv/jars/opencv-2.4.9-7.jar:\
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
$HOME/.ivy2/cache/com.twelvemonkeys.imageio/imageio-thumbsdb/jars/imageio-thumbsdb-3.2.1.jar

# path to the coco dataset,
# you can download by : http://msvocds.blob.core.windows.net/coco2015/test2015.zip
TRAIN_DATA_PATH=/home/ldpe2g/DataSets/Coco/test2015

VGG_MODEL=$ROOT/datas/vggmodel/vgg19.params

SAVE_MODEL_PATH=$ROOT/datas/models

if [ ! -d $SAVE_MODEL_PATH ] ; then
	mkdir -p $SAVE_MODEL_PATH
fi

STYLE_IMAGE=$ROOT/datas/images/starry_night.jpg

LEARNING_RATE=0.0001

# resume the training progress
# by adding the commamd line parameter: 
# --resume-model-path $RESUME_MODEL_PATH
RESUME_MODEL_PATH=

# -1 for cpu
GPU=0

java -Xmx29G  -cp $CLASS_PATH \
	Train \
	--data-path $TRAIN_DATA_PATH \
	--vgg-model-path $VGG_MODEL \
	--save-model-path $SAVE_MODEL_PATH \
	--style-image $STYLE_IMAGE \
	--lr $LEARNING_RATE \
	--gpu $GPU

