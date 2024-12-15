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
from utils import get_closest_identity
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




def get_word_counts(df):
    word_counts = []
    wset = []
    for _,instance in df.iterrows():
        #print(instance['answer'])
        assignment = get_closest_identity(instance['answer'],instance['words'],instance['identities'])
        #print(w,assignment)
        word_counts.append(assignment)
        wset+=list(assignment.keys())
        wset = list(set(wset))
    per_word_counts = {w: {x: y for x, y in zip(*np.unique([wc.get(w, None) for wc in word_counts], return_counts=True))} for w in wset}
    return per_word_counts
sanitize_model2 = lambda x: x.replace('/','!').replace('-','~').replace('_','&')
retrieve_model2 = lambda x: x.replace('!','/').replace('~','-').replace('&','_')

import glob
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
    print('anyna',any(df_this['cfg_path'].isna().tolist()))
    assert df_this['cfg_path'].unique().shape[0] == 1


    cfg=load_json(df_this['cfg_path'].unique()[0])
    words=cfg['words']
    identities=cfg['identities']
    description=full_cfg['tasks'][c[0]]['task_label']

    identities_2 = identities

    counts = get_word_counts(df_this)
    
    # invert identities dict
    val2idx = {v: k for k, v in sorted(identities_2.items(), key=lambda item: item[1]['valence'])}
    #sort val2idx

    # ratio count positive valence/(sum count valences)
    val2idx = list(val2idx.items())
    num = val2idx[-1][-1] # name of julia in general
    den = val2idx[0][-1] # name of ben in general


    word_dict={}
    word_dict['description']=description
    word_dict['formula']=f'{num}/({num}+{den})'
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
df_words.sort_values(by=['model','language','task'],inplace=True)
df_words.to_csv('data/df_words.csv',index=False)

