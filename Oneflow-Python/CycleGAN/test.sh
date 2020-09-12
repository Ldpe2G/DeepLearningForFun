CHECKPOINT_LOAD_DIR="./pretrain_models/horse2zebra_epoch_73_iter_200_gloss_3.497316_dloss_0.253239/"

NET_IN_SIZE=256

INPUT_IMAGES="./datasets/horse2zebra/testB/"
OUTPUT_IMAGES="./outputs/horse2zebra/outB2A/"

# DIRECTION="A2B"
DIRECTION="B2A"

python3 test_of_cyclegan.py \
    --input_images $INPUT_IMAGES \
    --network_input_size $NET_IN_SIZE \
    --output_images $OUTPUT_IMAGES \
    --direction $DIRECTION \
    --checkpoint_load_dir $CHECKPOINT_LOAD_DIR
