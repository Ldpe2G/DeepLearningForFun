CHECKPOINT_LOAD_DIR="./checkpoints/epoch_38_iter_400_gloss_5.349748_dloss_0.334908"

# make sure $CHECKPOINT_SAVE_DIR is already exists before start the training process.
CHECKPOINT_SAVE_DIR="./checkpoints"

# 
TRAIN_DATASET_A="./datasets/summer2winter_yosemite/trainA"
TRAIN_DATASET_B="./datasets/summer2winter_yosemite/trainB"

RESIZE_AND_CROP=True
CROP_SIZE=256
LOAD_SIZE=286

SAVE_TMP_IMAGE_PATH="./training_cyclegan.jpg"

EPOCH=300
LR=0.0002

LAMBDA_A=10.0 
LAMBDA_B=10.0
LAMBDA_IDENTITY=0.5

python3 train_of_cyclegan.py \
    --datasetA_path $TRAIN_DATASET_A \
    --datasetB_path $TRAIN_DATASET_B \
    --resize_and_crop $RESIZE_AND_CROP \
    --crop_size $CROP_SIZE \
    --load_size $LOAD_SIZE \
    --save_tmp_image_path $SAVE_TMP_IMAGE_PATH \
    --train_epoch $EPOCH \
    --learning_rate $LR \
    --lambda_A $LAMBDA_A \
    --lambda_B $LAMBDA_B \
    --lambda_identity $LAMBDA_IDENTITY \
    --checkpoint_save_dir $CHECKPOINT_SAVE_DIR \
    # --checkpoint_load_dir $CHECKPOINT_LOAD_DIR

