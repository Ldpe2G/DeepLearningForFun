import numpy as np
import os 
import random
from collections import deque
import oneflow as flow
import oneflow.typing as tp
from threading import Thread, Lock
import threading

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0. # 0.001 # final value of epsilon
INITIAL_EPSILON = 0. # 0.005 # starting value of epsilon
MAX_REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100

ACTIONS_NUM = 2 # two actions
DEVICE_TAG = "gpu"
DEVICE_NUM = 1

def dataPrep(data):
    # at the begining data.shape = (64, 64, 4) or (64, 64, 1)
    if data.shape[2] > 1:
        mean = np.array([128, 128, 128, 128])
        reshaped_mean = mean.reshape(1, 1, 4)
    else:
        mean=np.array([128])
        reshaped_mean = mean.reshape(1, 1, 1)
    data = data - reshaped_mean
    # convert hwc -> chw 
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    data = np.expand_dims(data, axis = 0)
    # before return data.shape = (1, 4, 64, 64) or (1, 1, 64, 64)
    return data

# get QNet parameters
def getQNetParams(var_name_prefix: str = "QNet",
                  is_train: bool = True):
    weight_init = flow.variance_scaling_initializer(scale = 1.0, mode = "fan_in", distribution = "truncated_normal", data_format = "NCHW")
    bias_init = flow.constant_initializer(value = 0.)

    conv_prefix = "_conv1"
    conv1_weight = flow.get_variable(
        var_name_prefix + conv_prefix + "_weight",
        shape = (32, 4, 3, 3),
        dtype = flow.float32,
        initializer = weight_init,
        trainable = is_train        
    )
    conv1_bias = flow.get_variable(
        var_name_prefix + conv_prefix + "_bias",
        shape = (32,),
        dtype = flow.float32,
        initializer = bias_init,
        trainable = is_train
    )

    conv_prefix = "_conv2"
    conv2_weight = flow.get_variable(
        var_name_prefix + conv_prefix + "_weight",
        shape = (32, 32, 3, 3),
        dtype = flow.float32,
        initializer = weight_init,
        trainable = is_train        
    )
    conv2_bias = flow.get_variable(
        var_name_prefix + conv_prefix + "_bias",
        shape = (32,),
        dtype = flow.float32,
        initializer = bias_init,
        trainable = is_train
    )

    fc_prefix = "_fc1"
    fc1_weight = flow.get_variable(
        var_name_prefix + fc_prefix + "_weight",
        shape = (512, 32 * 16 * 16),
        dtype = flow.float32,
        initializer = weight_init,
        trainable = is_train        
    )
    fc1_bias = flow.get_variable(
        var_name_prefix + fc_prefix + "_bias",
        shape = (512,),
        dtype = flow.float32,
        initializer = bias_init,
        trainable = is_train
    )

    fc_prefix = "_fc2"
    fc2_weight = flow.get_variable(
        var_name_prefix + fc_prefix + "_weight",
        shape = (ACTIONS_NUM, 512),
        dtype = flow.float32,
        initializer = weight_init,
        trainable = is_train        
    )
    fc2_bias = flow.get_variable(
        var_name_prefix + fc_prefix + "_bias",
        shape = (ACTIONS_NUM,),
        dtype = flow.float32,
        initializer = bias_init,
        trainable = is_train
    )

    return conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias


def createOfQNet(input_image: tp.Numpy.Placeholder((BATCH_SIZE, 4, 64, 64), dtype = flow.float32),
                 var_name_prefix: str = "QNet",
                 is_train: bool = True) -> tp.Numpy:
    
    conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias = \
        getQNetParams(var_name_prefix = var_name_prefix, is_train = is_train)

    conv1 = flow.nn.compat_conv2d(
        input_image,
        conv1_weight,
        strides = [1, 1],
        padding = "same",
        data_format = "NCHW"
    )
    conv1 = flow.nn.bias_add(conv1, conv1_bias, "NCHW")
    conv1 = flow.layers.batch_normalization(inputs = conv1, axis = 1, name = "conv1_bn")
    conv1 = flow.nn.relu(conv1)

    pool1 = flow.nn.max_pool2d(conv1, 2, 2, "VALID", "NCHW", name = "pool1")

    conv2 = flow.nn.compat_conv2d(
        pool1,
        conv2_weight,
        strides = [1, 1],
        padding = "same",
        data_format = "NCHW"
    )
    conv2 = flow.nn.bias_add(conv2, conv2_bias, "NCHW")
    conv2 = flow.layers.batch_normalization(inputs = conv2, axis = 1, name = "conv2_bn")
    conv2 = flow.nn.relu(conv2)
    
    pool2 = flow.nn.max_pool2d(conv2, 2, 2, "VALID", "NCHW", name = "pool2")

    # conv3.shape = (32, 32, 16, 16), after reshape become (32, 32 * 16 * 16)
    pool2_flatten = flow.reshape(pool2, (BATCH_SIZE, -1))
    fc1 = flow.matmul(a = pool2_flatten, b = fc1_weight, transpose_b = True)
    fc1 = flow.nn.bias_add(fc1, fc1_bias)
    fc1 = flow.layers.batch_normalization(inputs = fc1, axis = 1, name = "fc1_bn")
    fc1 = flow.nn.relu(fc1)

    fc2 = flow.matmul(a = fc1, b = fc2_weight, transpose_b = True)
    fc2 = flow.nn.bias_add(fc2, fc2_bias)

    return fc2

def get_train_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config

def get_predict_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config

@flow.global_function("train", get_train_config())
def trainQNet(input_image: tp.Numpy.Placeholder((BATCH_SIZE, 4, 64, 64), dtype = flow.float32),
              y_input: tp.Numpy.Placeholder((BATCH_SIZE,), dtype = flow.float32),
              action_input: tp.Numpy.Placeholder((BATCH_SIZE, 2), dtype = flow.float32)):
    with flow.scope.placement(DEVICE_TAG, "0:0-%d" % (DEVICE_NUM - 1)):
        out = createOfQNet(input_image, var_name_prefix = "QNet", is_train = True)
        Q_Action = flow.math.reduce_sum(out * action_input, axis = 1)
        cost = flow.math.reduce_mean(flow.math.square(y_input - Q_Action))
        learning_rate = 0.0002
        beta1 = 0.9
        flow.optimizer.Adam(flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]), beta1 = beta1).minimize(cost)

@flow.global_function("predict", get_predict_config())
def predictQNet(input_image: tp.Numpy.Placeholder((BATCH_SIZE, 4, 64, 64), dtype = flow.float32)) -> tp.Numpy:
    with flow.scope.placement(DEVICE_TAG, "0:0-%d" % (DEVICE_NUM - 1)):
        out = createOfQNet(input_image, var_name_prefix = "QNetT", is_train = False)
        return out

# copy QNet parameters to QNetT
@flow.global_function("predict", get_predict_config())
def copyQNetToQnetT():
    with flow.scope.placement(DEVICE_TAG, "0:0-%d" % (DEVICE_NUM - 1)):
        t_conv1_weight, t_conv1_bias, t_conv2_weight, t_conv2_bias, t_fc1_weight, t_fc1_bias, t_fc2_weight, t_fc2_bias = \
            getQNetParams(var_name_prefix = "QNet", is_train = True)
        p_conv1_weight, p_conv1_bias, p_conv2_weight, p_conv2_bias, p_fc1_weight, p_fc1_bias, p_fc2_weight, p_fc2_bias = \
            getQNetParams(var_name_prefix = "QNetT", is_train = False)

        flow.assign(p_conv1_weight, t_conv1_weight)
        flow.assign(p_conv1_bias, t_conv1_bias)
        flow.assign(p_conv2_weight, t_conv2_weight)
        flow.assign(p_conv2_bias, t_conv2_bias)
        flow.assign(p_fc1_weight, t_fc1_weight)
        flow.assign(p_fc1_bias, t_fc1_bias)
        flow.assign(p_fc2_weight, t_fc2_weight)
        flow.assign(p_fc2_bias, t_fc2_bias)

class OfBrainDQN:
    def __init__(self, args):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON

        self.trainQNet = trainQNet
        self.predictQNet = predictQNet
        self.copyQNetToQnetT = copyQNetToQnetT

        self.check_point_dir = args.checkpoints_path
        self.pretrain_models = args.pretrain_models

        self.check_point = flow.train.CheckPoint()
        if self.pretrain_models != '':
            self.check_point.load(self.pretrain_models)
        else:
            self.check_point.init()

        self.time_step_mutex = Lock()
        self.predict_QNet_mutex = Lock()
        self.replay_memory_mutex = Lock()
        self.train_thread = Thread(target = self.trainQNetwork)
        self.thread_started = False
        
    def trainQNetwork(self):

        while True:
            self.replay_memory_mutex.acquire()
            # Step 1: obtain random minibatch from replay memory
            minibatch = random.sample(self.replayMemory, BATCH_SIZE)
            self.replay_memory_mutex.release()

            # state_batch.shape = (BATCH_SIZE, 4, 80, 80)
            state_batch = np.squeeze([data[0] for data in minibatch])
            action_batch = np.squeeze([data[1] for data in minibatch])
            reward_batch = np.squeeze([data[2] for data in minibatch])
            next_state_batch = np.squeeze([data[3] for data in minibatch])

            # Step 2: calculate y_batch
            self.predict_QNet_mutex.acquire()
            Qvalue_batch = self.predictQNet(next_state_batch)
            self.predict_QNet_mutex.release()

            terminal = np.squeeze([data[4] for data in minibatch])
            y_batch = reward_batch.astype(np.float32)
            terminal_false = terminal == False
            if (terminal_false).shape[0] > 0:
                y_batch[terminal_false] += (GAMMA * np.max(Qvalue_batch, axis=1))[terminal_false]

            # do forward, backward and update parameters
            self.trainQNet(state_batch, y_batch, action_batch)

            self.time_step_mutex.acquire()
            localTimeStep = self.timeStep
            self.time_step_mutex.release()

            # save network every 100 iterations
            if localTimeStep % 100 == 0:
                if not os.path.exists(self.check_point_dir):
                    os.mkdir(self.check_point_dir)
                save_path = '%s/network-dqn_of_%d' % (self.check_point_dir, localTimeStep)
                if not os.path.exists(save_path):
                    self.check_point.save(save_path)

            if localTimeStep % UPDATE_TIME == 0:
                self.predict_QNet_mutex.acquire()
                self.copyQNetToQnetT()
                self.predict_QNet_mutex.release()
            

    def setInitState(self, observation):
        # temp.shape = (1, 4, 80, 80)
        temp = dataPrep(np.stack((observation, observation, observation, observation), axis = 2))
        self.currentState = temp
    
    def setPerception(self, nextObservation, action, reward, terminal):
        # discard the first channel of currentState and append nextObervation
        # newState.shape = (1, 4, 80, 80)
        newState = np.append(self.currentState[:, 1:, :, :], dataPrep(nextObservation), axis = 1)
        self.replay_memory_mutex.acquire()
        self.replayMemory.append(
            (self.currentState.astype(np.float32), action.astype(np.float32), reward, newState.astype(np.float32), terminal))
        self.replay_memory_mutex.release()

        if len(self.replayMemory) > MAX_REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE and not self.thread_started:
            # Train the network
            self.train_thread.start()
            self.thread_started = True

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.timeStep % UPDATE_TIME == 0:
            print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.time_step_mutex.acquire()
        self.timeStep += 1
        self.time_step_mutex.release()

    def getAction(self):
        
        input_images = np.repeat(self.currentState, BATCH_SIZE, axis = 0).astype(np.float32)

        self.predict_QNet_mutex.acquire()
        Qvalue = np.squeeze(self.predictQNet(input_images))
        self.predict_QNet_mutex.release()

        Qvalue = Qvalue[0]
        action = np.zeros(ACTIONS_NUM)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(ACTIONS_NUM)
                action[action_index] = 1
            else:
                action_index = np.argmax(Qvalue)
                action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action