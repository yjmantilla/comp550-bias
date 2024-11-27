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

def get_word_counts(words,identities,df):

    word_counts={}
    for w in words:
        word_counts[w]=[]
        for _,instance in df.iterrows():
            #print(instance['answer'])
            assignment = get_closes_identity(instance['answer'],words,identities)[w]
            #print(w,assignment)
            word_counts[w].append(assignment)
        ws,cns=np.unique(word_counts[w],return_counts=True)
        word_counts[w]= {k:v for k,v in zip(ws,cns)}
    
    return word_counts
sanitize_model2 = lambda x: x.replace('/','!').replace('-','~').replace('_','&')
retrieve_model2 = lambda x: x.replace('!','/').replace('~','-').replace('&','_')

import glob
score_cols = [x for x in df.columns if 'bias' in x]
counts_dict = {}
grouping_by=['task','language','model']
index=0
full_cfg=cfg=read_yaml('cfg.yml')
for c,df_this in df.groupby(grouping_by):
    index+=1
    print(df_this)
    task = c[0]
    language = c[1]
    model = c[2]
    df_this['model2'] = df_this['model'].apply(sanitize_model2)
    df_this['path']=df_this.apply(lambda x: f'data/task-{x["task"]}_language-{x["language"]}_model-*{x["model2"]}*_cfg.json',axis=1)
    df_this['cfg_path'] = df_this.apply(lambda x: glob.glob(x['path'])[0] if len(glob.glob(x['path']))==1 else None,axis=1)
    print(any(df_this['cfg_path'].isna().tolist()))
    assert df_this['cfg_path'].unique().shape[0] == 1


    cfg=load_json(df_this['cfg_path'].unique()[0])
    words=cfg['words']
    identities=cfg['identities']
    counts = get_word_counts(words,identities,df_this)
    
    # invert identities dict
    val2idx = {v: k for k, v in sorted(cfg['identities'].items(), key=lambda item: item[0])}
    #sort val2idx

    # ratio count positive valence/(sum count valences)
    val2idx = list(val2idx.items())
    num = val2idx[-1][-1] # name of julia in general
    den = val2idx[0][-1] # name of ben in general

    counts
    word_dict={}
    word_dict.update({k:v for k,v in zip(grouping_by,c)})

    # find correspondence between english and the other language
    lang2lang = {}
    ref_lange = full_cfg['tasks'][c[0]]['languages'].index('english')
    for w,dataw in full_cfg['tasks'][c[0]]['words'].items():
        this_lang_index=full_cfg['tasks'][c[0]]['languages'].index(language)
        lang2lang[dataw[this_lang_index+1]]=dataw[ref_lange+1]
    for w,c_dict in counts.items():
        word_dict[lang2lang[w]]=c_dict.get(num,0)/(c_dict.get(num,0)+c_dict.get(den,0))
    
    counts_dict[index]=word_dict




df_words=pd.DataFrame.from_dict(counts_dict,orient='index')
df_words.to_csv('data/df_words.csv',index=False)

