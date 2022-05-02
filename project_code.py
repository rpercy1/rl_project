import os, pathlib
import numpy as np
import pandas as pd
import tensorflow as tf

my_path = str(pathlib.Path('__file__').parent.absolute().parent.absolute())
all_random = pd.read_csv(os.path.join(my_path, 'Project_Data', 'Random', 'all.csv'), engine='pyarrow', index_col=0)
# all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\random\all.csv")
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