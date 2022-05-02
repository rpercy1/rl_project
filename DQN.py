#Using the MDP example that we saw in class, the code below implements the Q-Learning algorithm

#shape: [ [actions in s0], [actions in s1], [actions in s2]]
allowed_actions = [[0,1,2],[0,1],[2]]

transition_probabilities = [
    #when in s0 [probs of [going to s0,s1,s2 given a0],[going to s0,s1,s2 given a1],[going to s0,s1,s2 given a2]]:
    [[1.0,0.0,0.0],[0.3,0.7,0.0],[0.6,0.4,0.0]], 
    #when in s1 [probs of [going to s0,s1,s2 given a0],[going to s0,s1,s2 given a1],[going to s0,s1,s2 given a2]]:
    [[0.5,0.5,0.0],[0.0,0.0,1.0],None], 
    #when in s2 [probs of [going to s0,s1,s2 given a0],[going to s0,s1,s2 given a1],[going to s0,s1,s2 given a2]]:   
    [None,None,[0.0,0.9,0.1]]
]


rewards = [
    #when in s0 [reward of [going to s0,s1,s2 given a0],[going to s0,s1,s2 given a1],[going to s0,s1,s2 given a2]]:
    [[0,0,0],[0,0,0],[20,0,0]], 
    #when in s1 [reward of [going to s0,s1,s2 given a0],[going to s0,s1,s2 given a1],[going to s0,s1,s2 given a2]]:
    [[-30,0,0],[0,0,10],None], 
    #when in s2 [reward of [going to s0,s1,s2 given a0],[going to s0,s1,s2 given a1],[going to s0,s1,s2 given a2]]:   
    [None,None,[0,0,0]]
]


import numpy as np
import tensorflow as tf
#model
#Specify architecture

inputs = tf.keras.layers.Input(shape=(3,), name='input') #We have 3 states
#We know that we only need 9 numbers so a single linear layer is enough, no hidden layers required
#We do not need a bias, because the inputs will never be [0,0,0]
output = tf.keras.layers.Dense(units=3, activation = "linear", name= 'output', use_bias = False)(inputs) #we have at most three actions, therefore 3 units

#Create model 
model = tf.keras.Model(inputs = inputs, outputs = output)

#Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)


#loop
#########################################

#define helper functions    

#function to choose actions
def egreedy(s,model,epsilon):
    if np.random.random() < epsilon:
        #explore
        a = np.random.choice(allowed_actions[s])
    else:
        #exploit
        s_onehot = np.array([tf.one_hot(s, 3)])
        Q_values = model.predict(s_onehot)
        #cancel out actions that are not allowed by adding negative infinity to those Q-values
        mask = np.array([-np.Inf]*3)
        for ind in allowed_actions[s]:
            mask[ind] = 0
        Q_values = Q_values + mask
        a = np.argmax(Q_values[0])
    return a



#function to take action and receive s' and r
#Note:
#-here we do not know the transition probabilities and rewards until we take the action
#-in Value-Iteration we do know the transition probabilities in advance (before the start of the algorithm) and we can therefore use them in the sum across all transitions (i.e., weighted average across all possible next s from (s,a) with the weight being the transition probability)
def take_action(s,a):
    #we do not know the probabilities until we take the action here and get feedback from the environment
    probs = transition_probabilities[s][a] 
    sp = np.random.choice([0,1,2],p=probs) #sp is short for s prime or next state
    #we do not know the rewards until we take the action here and get feedback from the environment
    r = rewards[s][a][sp] #reward
    return sp, r

s = 0 #initial state 

from collections import deque
replay_memory = deque(maxlen=5000)
#A deque is a special list (a linked list (each element points to the next one and to the pevious one).
#It has a convenient maxlen argument:
#Once a deque has reached its maxlen, when new items are added, a corresponding number of items are discarded from the opposite end.
#This means that we don't need to bother deleting old items from the list, everything is handled by the deque.


gamma = 0.95 #discount factor
batch_size = 20
nbr_states = 3
nbr_actions = 3
loss_fn_1 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
# loss_fn_2 = tf.keras.metrics.mean_squared_error
r_list = [] #for plotting

nbr_update_steps = 350000
for i in range(nbr_update_steps):
        
    #Select action
    starting_value = 0.3
    epsilon = max(starting_value - i/nbr_update_steps, 0.1) #linearly decrease epsilon
    a = egreedy(s,model,epsilon)
    
    #Take action
    sp, r = take_action(s,a)
    r_list.append(r) #for plotting
    
    #Store transition in D (and delete oldest transition if D is full)
    replay_memory.append((s,a,r,sp))
    
    #Train
    if len(replay_memory) > 100: #only train when we have at least 100 records to have some diversity in the dataset
        #sample minibatch from D
        indices = np.random.randint(len(replay_memory),size = batch_size)
        batch = np.array([replay_memory[index] for index in indices])
        #unpack minibatch
        state_batch = tf.one_hot(batch[:,0],nbr_states)
        action_batch = tf.one_hot(batch[:,1],nbr_actions)
        reward_batch = batch[:,2]
        next_state_batch = tf.one_hot(batch[:,3],nbr_states)
        
        #Compute loss
        ###Compute Q_target
        next_Q_values = model.predict(next_state_batch)
        #mask Q values of actions that are not allowed
        mask =  np.full((next_Q_values.shape[0],next_Q_values.shape[1]),-np.Inf)
        for ii,iii in enumerate([allowed_actions[ss] for ss in batch[:,3]]):
            for iiii in iii:
                mask[ii,iiii] = 0
        next_Q_values = next_Q_values + mask
        
        max_next_Q_values = np.max(next_Q_values,axis=1) #axis = 1 is by row
        Q_targets = reward_batch + gamma*max_next_Q_values
        ###Compute Q, loss, gradients, and updated weights
        with tf.GradientTape() as tape:
            #We only want the Q value of the action that was actually taken
            #However, the model returns Q values for all actions
            #Therefore we need to zero out the ones that we do not want by multiplication with the action_batch matrix.
            #We then sum by row to discard all the zeros, keeping only the Q-value of the experienced action.
            all_Q_values = model(state_batch)        
            Q_values = tf.reduce_sum(all_Q_values * action_batch, axis = 1, keepdims = True)
            loss = loss_fn_1(Q_targets,Q_values)
            #loss = tf.reduce_mean(loss_fn_2(Q_targets,Q_values))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    
    #set state
    s = sp
    
    if (i % 1000) == 0:
        print(i)



#Let's see which Q-values the DQN predicts for all state-action pairs
Qmatrix = []
states = np.array([[1,0,0],[0,1,0],[0,0,1]])
for state in states:
    Qmatrix.append(model.predict(state[np.newaxis]).tolist()[0])
Qmatrix = np.array(Qmatrix)
Qmatrix[1,2] = -np.Inf
Qmatrix[2,0] = -np.Inf
Qmatrix[2,1] = -np.Inf
Qmatrix


# The optimal policy is:
np.argmax(Qmatrix, axis=1)  #axis=1 means per row across columns
    # take a2 when in s0
    # take a1 when in s1
    # take a2 when in s2 (this is also the only allowed action)

#Plots

import matplotlib.pyplot as plt

#moving average last 1000 steps
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

plt.plot(moving_average(r_list,1000))
plt.ylabel('Moving average of rewards')
plt.xlabel('Iterations')
plt.title('Epsilon lin sched start 0.3, gamma = 0')
plt.show()


plt.plot(np.cumsum(r_list))
plt.ylabel('Cumulative sum of rewards')
plt.xlabel('Iterations')
plt.title('Epsilon lin sched start 0.3, gamma = 0')
plt.show()


#model.layers[1].get_weights()

