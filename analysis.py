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
    
import yaml
def read_yaml(f):
    with open(f, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

cfg=read_yaml('cfg.yml')


data_path = 'data/'
results=glob.glob(data_path+'*_samples.json')
# Load the data
results = [load_json(f) for f in results]
results = [pd.DataFrame.from_dict(r,orient='index') for r in results]
df = pd.concat(results)

def sanitize_model_name(x):
    return x.lower().replace('lmstudio-community/','').replace('instruct-gguf','').replace('instruct-','').replace('lmstudio-','')

df['model'] = df['model'].apply(sanitize_model_name)


def get_closes_identity(answer,words,identities):
    """
    Get the score of the answer based on the words and identities
    words: dictionary with words as keys and valences as values
    identities: dictionary with identities as keys and valences as values
    answer: string with the answer
    words_idx: dictionary with words as keys and indexes as values
    identities_idx: dictionary with identities as keys and indexes as values
    valences_idx: dictionary with valences as keys and indexes as values
    """
    answer = answer.lower()
    distances={}
    for w in [x.lower() for x in words]:
        try:
            distances[w] ={}
            for id in [ y.lower() for y in identities]:
                try:
                    distances[w][id] = answer[answer.index(w):].index(id)
                except:
                    distances[w][id] = np.inf # if not found, set to infinity
        except:
            print(f'{w} not found in content {answer}')

    # For each word, get the identity that is closest to it
    closest_identities = {w: min(distances[w], key=distances[w].get) for w in distances.keys()}

    return closest_identities


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


# prototype, count for each word how many times it is assigned to each identity
PATH_CFG='data/task-genderBias_language-english_model-lmstudio~community!Phi~3.1~mini~4k~instruct~GGUF_cfg.json'

cfg=load_json(PATH_CFG)
words=cfg['words']
identities=cfg['identities']
answer=df.iloc[0]['answer']

get_closes_identity(answer,words,identities)

df_english=df[df['language']=='english']
word_counts={}
for w in words:
    word_counts[w]=[]
    for _,instance in df_english.iterrows():
        #print(instance['answer'])
        assignment = get_closes_identity(instance['answer'],words,identities)[w]
        #print(w,assignment)
        word_counts[w].append(assignment)
    word_counts[w]=np.unique(word_counts[w],return_counts=True)

