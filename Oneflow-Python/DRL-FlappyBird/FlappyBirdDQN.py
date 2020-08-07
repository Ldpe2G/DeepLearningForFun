import cv2
import numpy as np
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQNOneFlow import OfBrainDQN

# preprocess raw image to get 80 * 80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))

def playFlappyBird(args):
    # Step 1: init BrainDQN
    brain = OfBrainDQN(args)
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1,0])  # do nothing

    # observation0.shape = ((288, 512, 3), reward0: Float, terminal: Boolean
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    # observation0.shape = (80, 80)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while 1!= 0:
        action = brain.getAction()
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward, terminal)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoints_path', type=str, default='')
    parser.add_argument('--pretrain_models', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    playFlappyBird(args)
