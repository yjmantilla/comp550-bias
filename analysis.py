import json
import os
import traceback
import time
from pprint import pprint
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from openai import OpenAI # this is for the synthetic data
import yaml
import glob
import matplotlib.pyplot as plt
def load_json(f):
    with open(f,'r') as f:
        return json.load(f)

data_path = 'data/'
results=glob.glob(data_path+'*_samples.json')
# Load the data
results = [load_json(f) for f in results]
results = [pd.DataFrame.from_dict(r,orient='index') for r in results]
df = pd.concat(results)

def sanitize_model_name(x):
    return x.lower().replace('lmstudio-community/','').replace('instruct-gguf','').replace('instruct-','').replace('lmstudio-','')

df['model'] = df['model'].apply(sanitize_model_name)

score_cols = [x for x in df.columns if 'bias' in x]
for c,df_this in df.groupby(['task']):
    print(c)
    print(df_this)

    # make a boxplot for each score column
    for score_col in score_cols:
        df_this.boxplot(column=score_col,by=['model','language'])
        plt.title(score_col)
        fig=plt.gcf()
        # vertical x-axis labels
        plt.xticks(rotation=30)
        fname=f'task-{c[0]}_{score_col}.png'
        fig_path = os.path.join(data_path,fname)
        fig.suptitle(fname)
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close('all')
