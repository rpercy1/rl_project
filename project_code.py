import numpy as np
import pandas as pd


all_random = pd.read_csv(r"C:/Users/percy/OneDrive - University of Tennessee/MSBA/BZAN 583 Reinforcement/random/all.csv")
#caleb's path
all_random = pd.read_csv(r"C:/Users/caleb/Downloads/GroupAssignmentRecommender.zip/data/all_random/all.csv")

all_random.head()

################# Generate Unique Customer Groups ##########################
all_random.info() # 4 user features, use this to group users
len(all_random.user_feature_0.unique()) #4 options
len(all_random.user_feature_1.unique()) #6 options
len(all_random.user_feature_2.unique()) #10 options
len(all_random.user_feature_3.unique()) #10 options

possible_combos = 4*6*10*10 #2400

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

all_random['user-item_affinity_77'].unique()
all_random['user-item_affinity_45'].unique()






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
