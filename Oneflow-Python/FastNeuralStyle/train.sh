MODEL_LOAD_DIR="./vgg16_of_best_model_val_top1_721"

# make sure $MODEL_SAVE_DIR is already exists before start the training process.
MODEL_SAVE_DIR="./checkpoints"

RGB_MEAN="123.68,116.779,103.939"
RGB_STD="58.393,57.12,57.375"

STYLE_IMAGE="./images/style-images/sketch.jpg"

# download from http://msvocds.blob.core.windows.net/coco2015/test2015.zip
TRAIN_DATASET_PATH="/home/ldpe2g/DataSets/Coco/test2015/"

IMAGE_SIZE=256

# The setting of content_weight and style_weight depends on the style you want to transfer.
CW=10000 # sketch
# CW=3000 # wave_crop
# CW=3000 # the_scream
# CW=3000 # rain-princess
# CW=3000  # starry_night
# CW=3000  # mosaic
SW=1e10

LR=0.001

python3 train_of_neural_style.py \
    --style_image_path $STYLE_IMAGE \
    --dataset_path $TRAIN_DATASET_PATH \
    --learning_rate $LR  \
    --rgb_mean $RGB_MEAN \
    --rgb_std $RGB_STD \
    --content_weight $CW \
    --style_weight $SW \
    --train_image_size $IMAGE_SIZE \
    --model_save_dir $MODEL_SAVE_DIR \
    --model_load_dir $MODEL_LOAD_DIR
