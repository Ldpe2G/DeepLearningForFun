PRETRAIN_MODEL_PATH="pretrain_models"
CHECKPOINTS_PATH="checkpoints"


if [ -d "$CHECKPOINTS_PATH" ]; then
  rm -rf $CHECKPOINTS_PATH
fi

python3 FlappyBirdDQN.py \
    --checkpoints_path $CHECKPOINTS_PATH \
    --pretrain_models $PRETRAIN_MODEL_PATH
    
