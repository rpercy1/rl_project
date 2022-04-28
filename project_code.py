import numpy as np
import pandas as pd

all_random = pd.read_csv(r"C:\Users\percy\OneDrive - University of Tennessee\MSBA\BZAN 583 Reinforcement\random\all.csv")
all_random.head()

all_random.info() # 4 user features, use this to group users
all_random.user_feature_0.unique() #4 options
all_random.user_feature_1.unique() #6 options
len(all_random.user_feature_2.unique()) #10 options
len(all_random.user_feature_3.unique()) # 10 options

