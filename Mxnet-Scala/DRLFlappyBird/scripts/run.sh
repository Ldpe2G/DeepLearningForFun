ROOT=$(cd "$(dirname $0)/.."; pwd)

# put yur mxnet jar file in the lib folder
MXNET_JAR_FILE=$ROOT/lib/mxnet-full_2.11-linux-x86_64-gpu-1.0.1-SNAPSHOT.jar

CLASS_PATH=$MXNET_JAR_FILE:$ROOT/target/scala-2.11/classes/:$HOME/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.11.8.jar:\
$HOME/.ivy2/cache/args4j/args4j/bundles/args4j-2.33.jar:\
$HOME/.ivy2/cache/nu.pattern/opencv/jars/opencv-2.4.9-7.jar

# -1 for cpu
GPU=0

SAVE_MODRL_PATH=$ROOT/models

RESOURCES_PATH=$ROOT/assets


#### training #####

java -Xmx4G -cp $CLASS_PATH \
	FlappyBirdDQN \
	--gpu $GPU \
	--save-model-path $SAVE_MODRL_PATH \
	--resources-path $RESOURCES_PATH


#### resume training ####

# RESUME_MODRL_PATH=$ROOT/models/pretrain-model/network-dqn_mx46000.params

# java -Xmx4G -cp $CLASS_PATH \
# 	FlappyBirdDQN \
# 	--gpu $GPU \
# 	--save-model-path $SAVE_MODRL_PATH \
# 	--resources-path $RESOURCES_PATH \
# 	--resume-model-path $RESUME_MODRL_PATH
