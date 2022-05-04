import os, pathlib
import numpy as np
import pandas as pd
import tensorflow as tf


#read in the data for Random
try:
    all_random = pd.read_csv(r"C:/Users/caleb/Downloads/GroupAssignmentRecommender/data/all_random/all.csv")
except FileNotFoundError:
    try:
        all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\project\random\all.csv")
    except FileNotFoundError:
        my_path = str(pathlib.Path('__file__').parent.absolute().parent.absolute())
        all_random = pd.read_csv(os.path.join(my_path, 'Project_Data', 'Random', 'all.csv'), engine='pyarrow', index_col=0)

all_random.tail()

# read in data for Thompson
try:
    all_random = pd.read_csv(r"C:/Users/caleb/Downloads/GroupAssignmentRecommender/data/all_random/all.csv")
except FileNotFoundError:
    try:
        all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\project\random\all.csv")
    except FileNotFoundError:
        my_path = str(pathlib.Path('__file__').parent.absolute().parent.absolute())
        all_random = pd.read_csv(os.path.join(my_path, 'Project_Data', 'Random', 'all.csv'), engine='pyarrow', index_col=0)

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


# all_random['user-item_affinity_77'].unique()

# all_random['user'].head()
# all_random['user-item_affinity_77'].unique()
# all_random['user-item_affinity_45'].unique()

#reorder columns for take action function
all_random = all_random[[c for c in all_random if c not in ['click']] + ['click']]

row = int(np.floor(np.random.uniform(low = 0, high = 1374327+1)))
def get_action(row):
    '''
    Updates user affinity if item was clicked on, records reward
    '''
    sp = all_random.loc[[row], :]
    # update sp user affinities if clicked
    if all_random.click[row] == 1:
        affinity = 'user-item_affinity_' + str(all_random.item_id[row])
        sp[affinity] = all_random[affinity][row] + 1
    r = all_random.click[row]
    return sp, r

get_action(871998)
all_random.shape


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
all_random.drop(columns=['position','item_id','propensity_score',
                         'user_feature_0', 'user_feature_1','user_feature_2',
                         'user_feature_3'], inplace=True)

def prepare_data(df, row):
    """
    Prepare the data for training
    """
    x_cols = [col for col in df.columns if 'y_item' not in col]
    y_cols = [col for col in df.columns if 'y_item' in col]
    x = df.loc[[row], x_cols].copy()
    x_num = x.drop(columns=['user'])
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
    embedding_usr = tf.keras.layers.Embedding(input_dim=404, output_dim=323, input_length=1,name = 'embedding_cat')(inputs_usr)
    embedding_flat_usr = tf.keras.layers.Flatten(name='flatten_cat')(embedding_usr)


    inputs_num = tf.keras.layers.Input(shape=(86,),name = 'in_num') 

    inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat_usr, inputs_num])

    hidden1 = tf.keras.layers.Dense(25, activation="relu")(inputs_concat)
    hidden2 = tf.keras.layers.Dense(25, activation="relu")(hidden1)
    hidden3 = tf.keras.layers.Dense(25, activation="relu")(hidden2)
    q_values = tf.keras.layers.Dense(80, activation="linear")(hidden3)

    return tf.keras.Model(inputs=[inputs_usr, inputs_num], outputs=[q_values])

q_network = construct_q_network()  

loss_fn_1 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)


gamma = .95 #smaller


nbr_update_steps = 100
for i in range(nbr_update_steps):
    
    # randomly select a row from the data
    row_nbr = int(np.floor(np.random.uniform(low = 0, high = 1374327+1)))

    # prepare the data  
    x_num, x_user, y = prepare_data(all_random, row_nbr)
    x_user = tf.convert_to_tensor(x_user)

    #get action, update sp if clicked 
    sp,r = get_action(row_nbr)
    x_num_sp, x_user_sp, y_sp = prepare_data(sp, row_nbr)

    x_num.shape

    #print(sp)
    
    # Train
    #Compute loss
    next_Q_values = q_network.predict([x_user_sp, x_num_sp])

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
    

x1, x2, y = prepare_data(all_random, 844363)

q_network.predict([x2, x1]) 