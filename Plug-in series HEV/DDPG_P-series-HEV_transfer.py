# -*- coding: utf-8 -*-
"""
DDPG_Prius
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from series_new import series_HEV
import scipy.io as scio
import matplotlib.pyplot as plt
from Priority_Replay import Memory
import tensorflow.contrib.slim as slim

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 500
LR_A = 0.0009    # learning rate for actor
LR_C = 0.0009    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 50000
BATCH_SIZE = 64

RENDER = False

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
#        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.memory = Memory(capacity = MEMORY_CAPACITY)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], 'ISWeights')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.td_error_up = abs(q_target - q) * self.ISWeights 
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error * self.ISWeights, var_list=self.ce_params)

        a_loss = tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
                
        self.sess.run(tf.global_variables_initializer())  

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

#        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
#        bt = self.memory[indices, :]
        tree_index, bt, ISWeights = self.memory.sample(BATCH_SIZE)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.ISWeights: ISWeights})
       
        abs_td_error = self.sess.run(self.td_error_up, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.ISWeights: ISWeights}) 
        self.memory.batch_update(tree_index, abs_td_error)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
#        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
#        self.memory[index, :] = transition
        self.memory.store(transition)
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, 100, activation=tf.nn.relu, name = 'l2', trainable=trainable)
            net3 = tf.layers.dense(net2, 50, activation=tf.nn.relu, name = 'l3', trainable=trainable)
            a = tf.layers.dense(net3, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 200
            n_l2 = 100
            n_l3 = 50
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            w2 = tf.get_variable('w2', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            w3 = tf.get_variable('w3', [n_l2, n_l3], trainable=trainable)
            b3 = tf.get_variable('b3', [1, n_l3], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.nn.relu(tf.matmul(net1, w2) + b2)
            net3 = tf.nn.relu(tf.matmul(net2, w3) + b3)
            return tf.layers.dense(net3, 1, trainable=trainable)  # Q(s,a)

    def load_partial_weights(self):
        for tv in tf.trainable_variables():
            print (tv.name)
        variables_to_restore = slim.get_variables_to_restore(include = ['Actor/eval/l1', 'Actor/eval/l2', 'Actor/eval/l3',
                                                                        'Critic/eval/w1_s', 'Critic/eval/w1_a', 'Critic/eval/b1', 'Critic/eval/w2', 'Critic/eval/b2', 'Critic/eval/w3', 'Critic/eval/b3'])
        self.saver = tf.train.Saver(variables_to_restore)
        self.saver.restore(self.sess, os.path.join('Checkpoints/Prius', 'save_net.ckpt-500'))
   
    def savemodel(self):   
        self.saver = tf.train.Saver(max_to_keep = MAX_EPISODES)                    
        self.saver.save(self.sess, 'Checkpoints/Series_transfer/save_net.ckpt', global_step = step_episode)

            
s_dim = 3
a_dim = 1
a_bound = 1
DDPG = DDPG(a_dim, s_dim, a_bound)
series_HEV = series_HEV()
# initialize the neural network parameters with the transferred knowledge from Prius
DDPG.load_partial_weights()
# control exploration
var = 0.0
total_step = 0
step_episode = 0
mean_reward_all = 0
CNY_cost_list = []
cost_100Km_list = []
mean_reward_list = []
SOC_final_list = []
Reward_list_all = []
cost_list = []
#data_path = 'Data_Standard Driving Cycles/Standard_Chinacity.mat'
#data = scio.loadmat(data_path)
#car_spd_one = data['speed_vector']
#car_spd_one = np.tile(car_spd_one, (1, 3))
#total_milage = np.sum(car_spd_one) / 1000

for i in range(MAX_EPISODES): 
    path = "Data_Standard Driving Cycles/SHEV_training_data"
    path_list = os.listdir(path)
    random_data = np.random.randint(0,len(path_list))
    base_data = path_list[random_data]
    print(base_data)
    data = scio.loadmat(path + '/' + base_data)
    car_spd_one = data['speed_vector']
    if (np.sum(car_spd_one) / 1000) < 6:
        car_spd_one = np.tile(car_spd_one, (1, 3))
    total_milage = np.sum(car_spd_one) / 1000
    
    SOC = 0.3
    SOC_origin = SOC
    ep_reward = 0
    ep_reward_all = 0
    step_episode += 1
    SOC_data = []
    P_axle_list = []
    P_eng_list = []
    W_eng_list = []
    T_eng_list = []
    Eng_pwr_opt_list = []
    P_gen_list = []
    T_gen_list = []
    W_gen_list = []
    W_mot_list = []
    T_mot_list = []
    P_mot_list = []
    P_batt_list = []
    Reward_list = []
    Reward_list_1 = []
    T_axle_list = []
    Mot_eta_list = []
    Gen_eta_list = []
    price_fuel_list = []
    price_elec_list = []
    a_list = []
    car_spd = car_spd_one[:, 0]
    car_a = car_spd_one[:, 0] - 0
    s = np.zeros(s_dim)
    s[0] = car_spd / 23.5
    s[1] = (car_a - (-3.24)) / (1.81 - (-3.24))
#    s[0] = car_spd / 16.67
#    s[1] = (car_a - (-1.042)) / (0.914- (-1.042))
    s[2] = SOC
    for j in range(car_spd_one.shape[1] - 1):
        action = DDPG.choose_action(s)
        a_list.append(action)
        a = np.clip(np.random.laplace(action, var), 0, 1)
        Eng_pwr_opt = (a[0]) * 85000
        if Eng_pwr_opt < 20093:
            Eng_pwr_opt = 0
        
        out, cost, SOC_new = series_HEV.run(car_spd, car_a, Eng_pwr_opt, SOC)
        P_axle_list.append(float(out['P_axle']))
        T_axle_list.append(float(out['T_axle'])) 
        P_eng_list.append(float(out['P_eng']))
        W_eng_list.append(float(out['W_eng']))
        T_eng_list.append(float(out['T_eng'])) 
        Eng_pwr_opt_list.append(float(out['P']))
        W_mot_list.append(float(out['W_mot']))
        T_mot_list.append(float(out['T_mot']))        
        P_mot_list.append(float(out['P_mot']))  
        W_gen_list.append(float(out['W_gen']))
        T_gen_list.append(float(out['T_gen']))        
        P_gen_list.append(float(out['P_gen']))
        P_batt_list.append(float(out['P_batt']))
        price_fuel_list.append(float(out['price_fuel']))
        price_elec_list.append(float(out['price_elec']))
        Mot_eta_list.append(float(out['eff_m']))
        Gen_eta_list.append(float(out['eff_g']))
        SOC_new = float(SOC_new)
        SOC_data.append(SOC_new)
        cost = 10 * float(cost)
        r = cost
        ep_reward += r
        Reward_list.append(r)
        cost_list.append(r)

        r += (30 * ((0.5 - SOC_new) ** 4))            
                    
        # Obtained from the wheel speed sensor            
        car_spd = car_spd_one[:, j + 1]
        car_a = car_spd_one[:, j + 1] - car_spd_one[:, j]
        s_ = np.zeros(s_dim)
        s_[0] = car_spd / 23.5  
        s_[1] = (car_a - (-3.24)) / (1.81 - (-3.24))
#        s_[0] = car_spd / 16.67
#        s_[1] = (car_a - (-1.042)) / (0.914- (-1.042))
        s_[2] = SOC_new
        DDPG.store_transition(s, a, r, s_)
        
        if total_step > MEMORY_CAPACITY:
            var *= 0.999995
            DDPG.learn()
        
        s = s_
        ep_reward_all += r
        Reward_list_1.append(r)
        total_step += 1
        SOC = SOC_new

        if j == (car_spd_one.shape[1] - 2):
            SOC_final_list.append(SOC)
            Reward_list_all.append(ep_reward_all)
            mean_reward = ep_reward_all / car_spd_one.shape[1]
            mean_reward_list.append(mean_reward)
            CNY_cost = (ep_reward / 10)
            CNY_cost_list.append(CNY_cost)
            cost_100Km_list.append(CNY_cost * (10 / total_milage))
            print('Episode:', i, 'CNY_cost: %.3f' % (CNY_cost), 'Terminal SOC: %.3f' % SOC, 'mean_reward: %.3f' % mean_reward, ' Explore: %.2f' % var)          
   
    DDPG.savemodel()
      
x = np.arange(0, len(SOC_data), 1)
y = SOC_data
plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('SOC')