import os, pathlib
import numpy as np
import pandas as pd
import tensorflow as tf


#read in the data for Random
try:
    all_random = pd.read_csv(r"C:/Users/caleb/Downloads/GroupAssignmentRecommender/data/all_random/all.csv")
except FileNotFoundError:
    try:
        all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\project\thompson\all.csv", nrows = 300000)
    except FileNotFoundError:
        my_path = str(pathlib.Path('__file__').parent.absolute().parent.absolute())
        all_random = pd.read_csv(os.path.join(my_path, 'Project_Data', 'Random', 'all.csv'), engine='pyarrow', index_col=0)

all_random.tail()

# # read in data for Thompson
# try:
#     all_random = pd.read_csv(r"C:/Users/caleb/Downloads/GroupAssignmentRecommender/data/all_random/all.csv")
# except FileNotFoundError:
#     try:
#         all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\project\thompson\all.csv")
#     except FileNotFoundError:
#         my_path = str(pathlib.Path('__file__').parent.absolute().parent.absolute())
#         all_random = pd.read_csv(os.path.join(my_path, 'Project_Data', 'Random', 'all.csv'), engine='pyarrow', index_col=0)

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
len(all_random['user'].unique()) # 314 unique customer groups


#Discretize
codes, uniques = pd.factorize(all_random['user']) 
x = all_random['user'].astype('category')
len(x.cat.codes.unique())
all_random['user'] = x.cat.codes


# all_random['user-item_affinity_77'].unique()

# all_random['user'].head()
# all_random['user-item_affinity_77'].unique()
# all_random['user-item_affinity_45'].unique()

#reorder columns for take action function
all_random = all_random[[c for c in all_random if c not in ['click']] + ['click']]

row = int(np.floor(np.random.uniform(low = 0, high = 299999)))
def get_action(df, row):
    '''
    Updates user affinity if item was clicked on, records reward
    '''
    sp = df.loc[[row], :]
    # update sp user affinities if clicked
    if df.click[row] == 1:
        affinity = 'user-item_affinity_' + str(df.item_id[row])
        sp[affinity] = df[affinity][row] + 1
    r = df.click[row]
    return sp, r

get_action(all_random, 871998)
all_random.shape

all_random.columns

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
all_random.timestamp = pd.to_datetime(all_random.timestamp).apply(lambda x: x.value)

all_random = pd.concat([all_random,pd.get_dummies(all_random.position,prefix="postion"),
                        pd.get_dummies(all_random.item_id,prefix="y_item")],axis=1)
all_random.drop(columns=['position','propensity_score',
                         'user_feature_0', 'user_feature_1','user_feature_2',
                         'user_feature_3'], inplace=True)

#all_random_new.head()
from sklearn.preprocessing import StandardScaler
from itertools import chain

scaler = StandardScaler()
x_cols = [col for col in all_random.columns if 'y_item' not in col]
scaler.fit(all_random[x_cols].drop(columns=['user','item_id']))

def prepare_data(df, row):
    """
    Prepare the data for training
    """
    x_cols = [col for col in df.columns if 'y_item' not in col]
    y_cols = [col for col in df.columns if 'y_item' in col]
    x = df.loc[[row], x_cols].copy()
    x = x.drop(columns=['item_id'])
    x_num = scaler.transform(x.drop(columns=['user']))
    x_usr = x.user
    y = df.loc[[row], y_cols]

    return x_num, x_usr, y

#x_num, x_usr, y = prepare_data(all_random, 871998)



# create allowed actions
# for each product, we are allowed to put it in position 1, 2, or 3
# the allowed actions do not change based on the state, so the list 
# is the same independent of the state
allowed_actions = []
for _ in range(80):
    allowed_actions.append([0,1,2])
    

############################# DQN ###################################
def construct_q_network():
    """Construct the q-network with q-values per action as output"""
    
    inputs_usr = tf.keras.layers.Input(shape=(1,),name = 'in_user') 
    embedding_usr = tf.keras.layers.Embedding(input_dim=314, output_dim=250, input_length=1,name = 'embedding_cat')(inputs_usr)
    embedding_flat_usr = tf.keras.layers.Flatten(name='flatten_cat')(embedding_usr)


    inputs_num = tf.keras.layers.Input(shape=(86,),name = 'in_num') 

    inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat_usr, inputs_num])

    BNorm1 = tf.keras.layers.BatchNormalization()(inputs_concat)
    hidden1 = tf.keras.layers.Dense(250, activation="elu", kernel_initializer = 'he_uniform')(BNorm1)
    BNorm2 = tf.keras.layers.BatchNormalization()(hidden1)
    hidden2 = tf.keras.layers.Dense(200, activation="elu",  kernel_initializer = 'he_uniform')(BNorm2)
    BNorm3 = tf.keras.layers.BatchNormalization()(hidden2)
    hidden3 = tf.keras.layers.Dense(150, activation="elu",  kernel_initializer = 'he_uniform')(BNorm3)
    BNorm4 = tf.keras.layers.BatchNormalization()(hidden3)
    q_values = tf.keras.layers.Dense(80, activation="linear")(BNorm4)

    return tf.keras.Model(inputs=[inputs_usr, inputs_num], outputs=[q_values])

q_network = construct_q_network()  

#Fixed q-value targets
target_network = tf.keras.models.clone_model(q_network)
target_network.set_weights(q_network.get_weights())

loss_fn_1 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, clipnorm=.5)


state_q1 = []
state_q2 = []
state_q3 = []
state_q4 = []
state_q5 = []
track_x1_num, track_x1_user, track_y1 = prepare_data(all_random, 240)
track_x2_num, track_x2_user, track_y2 = prepare_data(all_random, 296)
track_x3_num, track_x3_user, track_y3 = prepare_data(all_random, 50000)
track_x4_num, track_x4_user, track_y4 = prepare_data(all_random, 500000)
track_x5_num, track_x5_user, track_y5 = prepare_data(all_random, 1000000)
#nbr_update_steps = 100000

#####
# state_q1 = []
# state_q2 = []
q_values_chosen_state = []
track_x1_num, track_x1_user, track_y1 = prepare_data(all_random, 240)
track_x2_num, track_x2_user, track_y2 = prepare_data(all_random, 296)

tf.keras.backend.clear_session()
gamma = .5
counter = 0
nbr_update_steps = 30000

for i in range(nbr_update_steps):
    
    counter += 1
    # randomly select a row from the data
    row_nbr = int(np.floor(np.random.uniform(low = 0, high = 299999+1)))

    # prepare the data  
    x_num, x_user, y = prepare_data(all_random, row_nbr)
    x_user = tf.convert_to_tensor(x_user)

    #get action, update sp if clicked 
    sp,r = get_action(all_random, row_nbr)
    x_num_sp, x_user_sp, y_sp = prepare_data(sp, row_nbr)

    x_num.shape

    #print(sp)
    
    # Train
    #Compute loss
    next_Q_values = target_network.predict([x_user_sp, x_num_sp])

    # action_batch = tf.one_hot(batch[:,1],nbr_actions)
    ###Compute Q_target

    max_next_Q_values = np.max(next_Q_values,axis=1) #axis = 1 is by row
    Q_targets = r + gamma*max_next_Q_values

    ###Compute Q, loss, gradients, and updated weights
    with tf.GradientTape() as tape:
    #     #track gradients in this section, start a context
    #     # make a prediction, look at all trainable variables and comput gradients
    #     #We only want the Q value of the action that was actually taken
    #     #However, the model returns Q values for all actions
    #     #Therefore we need to zero out the ones that we do not want by multiplication with the action_batch matrix.
    #     #We then sum by row to discard all the zeros, keeping only the Q-value of the experienced action.
        all_Q_values = q_network([x_user, x_num])  #same as model.predict but tracks gradients      
        Q_values = tf.reduce_sum(all_Q_values * y, axis = 1, keepdims = True) 
        loss = loss_fn_1(Q_targets,Q_values)
    #     #loss = tf.reduce_mean(loss_fn_2(Q_targets,Q_values))
    gradients = tape.gradient(loss, q_network.trainable_variables) #tracks impact of tiny change on output (loss)
    optimizer.apply_gradients(zip(gradients,q_network.trainable_variables)) #updates weights
    
    # state_q1.append(q_network.predict([track_x1_user, track_x1_num])[0][60])
    # state_q2.append(q_network.predict([track_x2_user, track_x2_num])[0][60])
    # state_q3.append(q_network.predict([track_x3_user, track_x3_num])[0][60])
    # state_q4.append(q_network.predict([track_x4_user, track_x4_num])[0][60])
    # state_q5.append(q_network.predict([track_x5_user, track_x5_num])[0][60])
    

    if counter % 100 == 0:
        # update target network weights
        target_network.set_weights(q_network.get_weights())
        
        # append q-values for plotting
        q_values_chosen_state.append(q_network.predict([track_x1_user, track_x1_num])[0])
        # state_q1.append(q_network.predict([track_x1_user, track_x1_num])[0][60])
        # state_q2.append(q_network.predict([track_x2_user, track_x2_num])[0][60])
        print(gradients)
        print(counter)

# MAKE A RECOMMENDATION
row_nbr = int(np.floor(np.random.uniform(low = 0, high = 1374327+1))) 
x_num, x_user, y = prepare_data(all_random, row_nbr)
q_network.predict([x_user, x_num])
np.argmax(q_network.predict([x_user, x_num]))



# item_track_1 = all_random.loc[240, 'item_id']
# item_track_2 = all_random.loc[296, 'item_id']

lst1 = [item[20] for item in q_values_chosen_state]
lst2 = [item[40] for item in q_values_chosen_state]
lst3 = [item[60] for item in q_values_chosen_state]
lst4 = [item[79] for item in q_values_chosen_state]
lst5 = [item[10] for item in q_values_chosen_state]
lst6 = [item[30] for item in q_values_chosen_state]
lst7 = [item[50] for item in q_values_chosen_state]
lst8 = [item[70] for item in q_values_chosen_state]
lst9 = [item[5] for item in q_values_chosen_state]
lst10 = [item[1] for item in q_values_chosen_state]

# store list of q_values
import pickle
with open('Q_240.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(state_q1, filehandle)
with open('Q_296.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(state_q2, filehandle)
with open('Q_50000.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(state_q3, filehandle)
with open('Q_500000.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(state_q4, filehandle)
with open('Q_1000000.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(state_q5, filehandle)
    
# with open('Q_240.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     state_q1 = pickle.load(filehandle)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(state_q1, label = 'State 240')
ax.plot(state_q2, label = 'State 296')
ax.plot(state_q3, label = 'State 50,000')
ax.plot(state_q4, label = 'State 500,000')
ax.plot(state_q5, label = 'State 1,000,000')
ax.set_xlabel('Number of Updates', fontweight='bold')
ax.set_ylabel('Q-Value', fontweight='bold')
ax.set_title('Q-Value of Item Tracking', fontweight='bold')
ax.set_yticklabels([0,0.15,0.3,0.45,0.6,0.75,0.9,1])
ax.legend(loc = 'upper left')
plt.savefig('q_values.png', bbox_inches='tight')
fig.show()

######
# plt.plot(state_q1, label = 'q1')
# plt.plot(state_q2, label = 'q2')
plt.plot(lst1, label='q20')
plt.plot(lst2, label='q40')
plt.plot(lst3, label='q60')
plt.plot(lst4, label='q79')
plt.legend(loc = 'upper left')
plt.show()


