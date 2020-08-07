import numpy as np
import os 
import random
from collections import deque
import oneflow as flow
import oneflow.typing as tp

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
    # at the begining data.shape = (80, 80, 4) or (80, 80, 1)
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
    # before return data.shape = (1, 4, 80, 80) or (1, 1, 80, 80)
    return data

# get QNet parameters
def getQNetParams(var_name_prefix: str = "QNet",
                  is_train: bool = True):
    weight_init = flow.variance_scaling_initializer(scale = 1.0, mode = "fan_in", distribution = "truncated_normal", data_format = "NCHW")
    bias_init = flow.constant_initializer(value = 0.)

    conv_prefix = "_conv1"
    conv1_weight = flow.get_variable(
        var_name_prefix + conv_prefix + "_weight",
        shape = (32, 4, 8, 8),
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
        shape = (64, 32, 4, 4),
        dtype = flow.float32,
        initializer = weight_init,
        trainable = is_train        
    )
    conv2_bias = flow.get_variable(
        var_name_prefix + conv_prefix + "_bias",
        shape = (64,),
        dtype = flow.float32,
        initializer = bias_init,
        trainable = is_train
    )

    conv_prefix = "_conv3"
    conv3_weight = flow.get_variable(
        var_name_prefix + conv_prefix + "_weight",
        shape = (64, 64, 4, 4),
        dtype = flow.float32,
        initializer = weight_init,
        trainable = is_train        
    )
    conv3_bias = flow.get_variable(
        var_name_prefix + conv_prefix + "_bias",
        shape = (64,),
        dtype = flow.float32,
        initializer = bias_init,
        trainable = is_train
    )

    fc_prefix = "_fc1"
    fc1_weight = flow.get_variable(
        var_name_prefix + fc_prefix + "_weight",
        shape = (512, 64 * 5 * 5),
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

    return conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, conv3_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias


def createOfQNet(input_image: tp.Numpy.Placeholder((BATCH_SIZE, 4, 80, 80), dtype = flow.float32),
                 var_name_prefix: str = "QNet",
                 is_train: bool = True) -> tp.Numpy:
    
    conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, conv3_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias = \
        getQNetParams(var_name_prefix = var_name_prefix, is_train = is_train)

    conv1 = flow.nn.compat_conv2d(
        input_image,
        conv1_weight,
        strides = [4, 4],
        padding = "same",
        data_format = "NCHW"
    )
    conv1 = flow.nn.bias_add(conv1, conv1_bias, "NCHW")
    conv1 = flow.nn.relu(conv1)

    pool1 = flow.nn.max_pool2d(conv1, 2, 2, "VALID", "NCHW")

    conv2 = flow.nn.compat_conv2d(
        pool1,
        conv2_weight,
        strides = [2, 2],
        padding = "same",
        data_format = "NCHW"
    )
    conv2 = flow.nn.bias_add(conv2, conv2_bias, "NCHW")
    conv2 = flow.nn.relu(conv2)
    
    conv3 = flow.nn.compat_conv2d(
        conv2,
        conv3_weight,
        strides = [1, 1],
        padding = "same",
        data_format = "NCHW"
    )
    conv3 = flow.nn.bias_add(conv3, conv3_bias, "NCHW")
    conv3 = flow.nn.relu(conv3)

    # conv3.shape = (32, 64, 5, 5), after reshape become (32, 64 * 5 * 5)
    conv3_flatten = flow.reshape(conv3, (BATCH_SIZE, -1))
    fc1 = flow.matmul(a = conv3_flatten, b = fc1_weight, transpose_b = True)
    fc1 = flow.nn.bias_add(fc1, fc1_bias)
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
def trainQNet(input_image: tp.Numpy.Placeholder((BATCH_SIZE, 4, 80, 80), dtype = flow.float32),
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
def predictQNet(input_image: tp.Numpy.Placeholder((BATCH_SIZE, 4, 80, 80), dtype = flow.float32)) -> tp.Numpy:
    with flow.scope.placement(DEVICE_TAG, "0:0-%d" % (DEVICE_NUM - 1)):
        out = createOfQNet(input_image, var_name_prefix = "QNetT", is_train = False)
        return out

# copy QNet parameters to QNetT
@flow.global_function("predict", get_predict_config())
def copyQNetToQnetT():
    with flow.scope.placement(DEVICE_TAG, "0:0-%d" % (DEVICE_NUM - 1)):
        t_conv1_weight, t_conv1_bias, t_conv2_weight, t_conv2_bias, t_conv3_weight, t_conv3_bias, t_fc1_weight, t_fc1_bias, t_fc2_weight, t_fc2_bias = \
            getQNetParams(var_name_prefix = "QNet", is_train = True)
        p_conv1_weight, p_conv1_bias, p_conv2_weight, p_conv2_bias, p_conv3_weight, p_conv3_bias, p_fc1_weight, p_fc1_bias, p_fc2_weight, p_fc2_bias = \
            getQNetParams(var_name_prefix = "QNetT", is_train = False)

        flow.assign(p_conv1_weight, t_conv1_weight)
        flow.assign(p_conv1_bias, t_conv1_bias)
        flow.assign(p_conv2_weight, t_conv2_weight)
        flow.assign(p_conv2_bias, t_conv2_bias)
        flow.assign(p_conv3_weight, t_conv3_weight)
        flow.assign(p_conv3_bias, t_conv3_bias)
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
        
    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        # state_batch.shape = (BATCH_SIZE, 4, 80, 80)
        state_batch = np.squeeze([data[0] for data in minibatch])
        action_batch = np.squeeze([data[1] for data in minibatch])
        reward_batch = np.squeeze([data[2] for data in minibatch])
        next_state_batch = np.squeeze([data[3] for data in minibatch])

        next_input_images = np.array(next_state_batch).astype(np.float32)
        # Step 2: calculate y_batch
        Qvalue_batch = self.predictQNet(next_input_images)
        y_batch = np.zeros((BATCH_SIZE,))
        terminal = np.squeeze([data[4] for data in minibatch])
        y_batch[:] = reward_batch
        if (terminal == False).shape[0] > 0:
            y_batch[terminal == False] += (GAMMA * np.max(Qvalue_batch, axis=1))[terminal == False]

        input_images = np.array(state_batch).astype(np.float32)
        action_input = np.array(action_batch).astype(np.float32)
        y_input = np.array(y_batch).astype(np.float32)
        # do forward, backward and update parameters
        self.trainQNet(input_images, y_input, action_input)

        # save network every 100 iterations
        if self.timeStep % 100 == 0:
            if not os.path.exists(self.check_point_dir):
                os.mkdir(self.check_point_dir)
            self.check_point.save('%s/network-dqn_of_%d' % (self.check_point_dir, self.timeStep))

        if self.timeStep % UPDATE_TIME == 0:
            self.copyQNetToQnetT()

    def setInitState(self, observation):
        # temp.shape = (1, 4, 80, 80)
        temp = dataPrep(np.stack((observation, observation, observation, observation), axis = 2))
        self.currentState = temp
    
    def setPerception(self, nextObservation, action, reward, terminal):
        # discard the first channel of currentState and append nextObervation
        # newState.shape = (1, 4, 80, 80)
        newState = np.append(self.currentState[:, 1:, :, :], dataPrep(nextObservation), axis = 1)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > MAX_REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

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
        self.timeStep += 1

    def getAction(self):
        
        input_images = np.repeat(self.currentState, BATCH_SIZE, axis = 0).astype(np.float32)
        Qvalue = np.squeeze(self.predictQNet(input_images))
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