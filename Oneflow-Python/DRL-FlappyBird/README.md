# Playing Flappy Bird Using Deep Reinforcement Learning

Implementation of DQN to play the Flappy Bird game with [Oneflow](https://github.com/Oneflow-inc/oneflow/) framework .

This work is based on two repos: [li-haoran/DRL-FlappyBird](https://github.com/li-haoran/DRL-FlappyBird) and [songrotek/DRL-FlappyBird](https://github.com/songrotek/DRL-FlappyBird.git)

## screenshots
<img src="play.gif"/>

## Environment
| Spec                        |                                                             |
|-----------------------------|-------------------------------------------------------------|
| Operating System            | Ubuntu 20                                             |
| GPU                         | Nvidia GTX 1070                                          |
| CUDA Version                | 10.2                                                        |
| Driver Version              | 440.100                                                      |
| Oneflow Version 	          | branch: master, commit_id: c8695c0c3768fd3d154c22fa1c997907e4b46666 |

## Requirements

* python3
    - pygame
    - numpy
    - opencv
* [Oneflow](https://github.com/Oneflow-inc/oneflow)

## Pretrain Model

[DQN_FlappyBird.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/rl/DQN_FlappyBird.zip)

## Play with pretrain model

```bash
bash ruh_flappy_bird.sh
```

## Train from scratch
 
Comments `--pretrain_models $PRETRAIN_MODEL_PATH` showing below in `run_flappy_bird.sh` then run the script

```bash
python3 FlappyBirdDQN.py \
    --checkpoints_path $CHECKPOINTS_PATH \
    #--pretrain_models $PRETRAIN_MODEL_PATH
```

It will take about at least 40000 time steps before the model can learn to play the game, be patients :).



