import pandas as pd
import numpy as np

def process(root):
    data = pd.read_csv(root)
    data['pixels'] = data['pixels'].apply(lambda x:  np.array(x.split(), dtype = "float32"))
    data = data.sample(frac = 1)
    
    subsample_len = int(0.8 * len(data))
    train_df = data[:subsample_len]
    eval_df = data[subsample_len:]

    return train_df, eval_df