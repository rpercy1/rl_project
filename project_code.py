import os, pathlib
import numpy as np
import pandas as pd
import tensorflow as tf


#read in the data
try:
    all_random = pd.read_csv(r"C:/Users/caleb/Downloads/GroupAssignmentRecommender/data/all_random/all.csv")
except FileNotFoundError:
    all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\project\random\all.csv")

# my_path = str(pathlib.Path('__file__').parent.absolute().parent.absolute())
# #all_random = pd.read_csv(os.path.join(my_path, 'Project_Data', 'Random', 'all.csv'), engine='pyarrow', index_col=0)
# all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\random\all.csv")


all_random.tail()

################# Generate Unique Customer Groups ##########################
all_random.info() # 4 user features, use this to group users
len(all_random.user_feature_0.unique()) #4 options
len(all_random.user_feature_1.unique()) #6 options
len(all_random.user_feature_2.unique()) #10 options
len(all_random.user_feature_3.unique()) #10 options

#possible_combos = 4*6*10*10 #2400

# test concatenation
#all_random.iloc[0,6] + " " + all_random.iloc[0,7] + " " + all_random.iloc[0,8] + " " + all_random.iloc[0,9]

# create new column
all_random['user'] = all_random['user_feature_0'] + " " + all_random['user_feature_1'] + " " + all_random['user_feature_2'] + " " + all_random['user_feature_3']
len(all_random['user'].unique()) # 404 unique customer groups

#Discretize
codes, uniques = pd.factorize(all_random['user']) 
x = all_random['user'].astype('category')
len(x.cat.codes.unique())
all_random['user'] = x.cat.codes
all_random['user'].head()


all_random.info()


all_random['user-item_affinity_77'].unique()
all_random['user-item_affinity_45'].unique()


all_random = all_random[[c for c in all_random if c not in ['click']] + ['click']].head()


def get_action():
    '''
    Returns a random row of the dataset in state, reward form
    '''
    row = int(np.floor(np.random.uniform(low = 0, high = 1374327+1)))
    sp = all_random.iloc[row, :]
    # update sp user affinities if clicked
    if all_random.click[row] == 1:
        affinity = 'user-item_affinity_' + str(all_random.item_id[row])
        sp[affinity] = all_random[affinity][row] + 1
    r = all_random.click[row]
    return sp, r





# questions
# update what a unique user will see?
#   affinities are 
# get state of user before showing something and then show state after and then get next state
# able to compute what next state is
# show something, get a reward, update affinity 
# change the affinity binary if they click
# use the affintity score
# inputs are states? what dim
# computing loss how do we do it without the matrix format
# item is an action, position is part of the state
# don't need user groups, just update user affinity
# how fast converges, just get the loss



all_random.position.unique()
len(all_random.item_id.unique())

use_cols = [col for col in all_random.columns if 'user_feature' not in col]
x = all_random.loc[:, use_cols].copy()
x.drop(columns=['item_id','propensity_score'], inplace=True)
x.timestamp = pd.to_datetime(x.timestamp).apply(lambda x: x.value)
x_num = x.drop(columns=['user'])
x_usr = x.user
y = all_random.item_id

allowed_actions = []
for _ in range(80):
    allowed_actions.append([0,1,2])
    
inputs_usr = tf.keras.layers.Input(shape=(1,),name = 'in_user') 
embedding_usr = tf.keras.layers.Embedding(input_dim=404, output_dim=323, input_length=1,name = 'embedding_cat')(inputs_usr)
embedding_flat_usr = tf.keras.layers.Flatten(name='flatten_cat')(embedding_usr)

inputs_num = tf.keras.layers.Input(shape=(83,),name = 'in_num') 

inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat_usr, inputs_num])

hidden1 = tf.keras.layers.Dense(25, activation="relu")(inputs_concat)
hidden2 = tf.keras.layers.Dense(25, activation="relu")(hidden1)
hidden3 = tf.keras.layers.Dense(25, activation="relu")(hidden2)
q_values = tf.keras.layers.Dense(80, activation="softmax")(hidden3)

model = tf.keras.Model(inputs=[inputs_usr, inputs_num], outputs=[q_values])





def construct_q_network(state_dim, action_dim):
    """Construct the q-network with q-values per action as output"""
    inputs = tf.keras.layers.Input(shape=(state_dim,))  # input dimension
    hidden1 = tf.keras.layers.Dense(
        25, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
    )(inputs)
    hidden2 = tf.keras.layers.Dense(
        25, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
    )(hidden1)
    hidden3 = tf.keras.layers.Dense(
        25, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
    )(hidden2)
    q_values = tf.keras.layers.Dense(
        action_dim, kernel_initializer=tf.keras.initializers.Zeros(), activation="linear"
    )(hidden3)

    return tf.keras.Model(inputs=inputs, outputs=[q_values])

q_network = construct_q_network(4, 80)
q_network.summary()


q_values = q_network.predict(x)

# nbr_update_steps = 101
# for i in range(nbr_update_steps):
    # read in row
       
    #Select action = item picture chosen

    #Take action = reward if any 
        # update the user affinity if necessary
        # sp = this same row with user affinity updated
    
    # Train
    #Compute loss
    #   make a q-value prediciton using the sp we found above
    
    ###Compute Q_target
    
    # max_next_Q_values = np.max(next_Q_values,axis=1) #axis = 1 is by row
    # Q_targets = reward + gamma*max_next_Q_values
    ###Compute Q, loss, gradients, and updated weights
    # with tf.GradientTape() as tape:
    #     #track gradients in this section, start a context
    #     # make a prediction, look at all trainable variables and comput gradients
    #     #We only want the Q value of the action that was actually taken
    #     #However, the model returns Q values for all actions
    #     #Therefore we need to zero out the ones that we do not want by multiplication with the action_batch matrix.
    #     #We then sum by row to discard all the zeros, keeping only the Q-value of the experienced action.
    #     all_Q_values = model(state_batch)  #same as model.predict but tracks gradients      
    #     Q_values = tf.reduce_sum(all_Q_values * action_batch, axis = 1, keepdims = True) #20x3 all q values, cancel out not-taken actions, sum across rows
    #     loss = loss_fn_1(Q_targets,Q_values)
    #     #loss = tf.reduce_mean(loss_fn_2(Q_targets,Q_values))
    # gradients = tape.gradient(loss, model.trainable_variables) #tracks impact of tiny change on output (loss)
    # optimizer.apply_gradients(zip(gradients,model.trainable_variables)) #updates weights
    
    # #set state
    # s = sp
    
    # if (i % 1000) == 0:
    #     print(i)


