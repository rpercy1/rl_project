import numpy as np
import pandas as pd

all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\random\all.csv")
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



