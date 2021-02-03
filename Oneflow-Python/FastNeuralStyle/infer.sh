MODEL_LOAD_DIR="pretrain_models/sketch_lr_0.001000_cw_10000.000000_sw_10000000000.000000_epoch_0_iter_4400_loss_3008.877197/"

INPUT_IMAGE="./images/content-images/oneflow.png"
OUTPUT_IMAGE="./images/style_out_oneflow.jpg"

python3 infer_of_neural_style.py \
    --input_image_path $INPUT_IMAGE \
    --output_image_path $OUTPUT_IMAGE \
    --model_load_dir $MODEL_LOAD_DIR
