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

FULL_CFG=read_yaml('cfg.yml')


data_path = 'data/'
results=glob.glob(data_path+'*_samples.json')
# Load the data
results = [load_json(f) for f in results]
results = [pd.DataFrame.from_dict(r,orient='index') for r in results]
df = pd.concat(results)

cfgs = glob.glob(data_path+'*_cfg.json')
cfgs = [load_json(f) for f in cfgs]
df_cfg = pd.DataFrame(cfgs)

# drop phi for now
df = df[~df['model'].str.contains('phi')]

def sanitize_model_name(x):
    return x.lower().replace('lmstudio-community/','').replace('instruct-gguf','').replace('instruct-','').replace('lmstudio-','')

df['model'] = df['model'].apply(sanitize_model_name)

# add configuration to dataframe
df_cfg['model'] = df_cfg['model'].apply(sanitize_model_name)

dict_cfgs=[]
for _,row in df.iterrows():
    # look for its cfg
    matching_criteria = ['task','language','model','bias','domain','baseline']

    df_temp = df_cfg.copy()
    for c in matching_criteria:
        df_temp = df_temp[df_temp[c]==row[c]]
    assert df_temp.shape[0]==1
    df_temp = df_temp.iloc[0].to_dict()
    dict_cfgs.append(df_temp)
df['dict_cfg']=dict_cfgs


row = df.iloc[0]
row['dict_cfg']['identities']
row['identities']

def get_assignment(row, general_cfg):
    identity_map = {' '.join([v['label']]+v['variants'][row['language']]):v['group'] for k,v in row['dict_cfg']['identities'].items()}
    word_map={v[row['language']]:v['group'] for k,v in row['dict_cfg']['words'].items()}
    all_identities = [v['group'] for k,v in row['dict_cfg']['identities'].items()]
    assert len(identity_map) == len(all_identities)
    assignment = get_closest_identity(row['answer'],row['words'],row['identities'])

    # Make multilingual word map
    general_cfg['tasks'][row['task']]['words']
    multi_word_map = {}

    for k,v in general_cfg['tasks'][row['task']]['words'].items():

        full_word =[]
        for _,v2 in v.items():
            full_word.append(v2)
        full_word = ' '.join(full_word)
        # assume group is at the start before the words in each languages, this will help to compute the scores
        multi_word_map[v[row['language']]]=full_word

    if row['baseline']=='0nope':
        group_map ={}
        for k,v in row['dict_cfg']['identities'].items():
            for id in row['identities']:
                if id in v['variants'][row['language']]:
                    group_def = v['group']
                    group_map[id]=group_def
    else:
        names=[]
        for k,v in row['dict_cfg']['identities'].items():
            names.append(v['variants'][row['language']])
        # if baseline, names should be the same
        name_def = names[0]
        for n in names:
            assert n==name_def
        # if 2 ids, there are two ways to make the baseline (either one or the other poses as the "other" identity, right we are doing this from the the order of the identities)
        group_map={k:v for k,v in zip(row['identities'],all_identities)}
        

    assignment_grouped={w:group_map[id] for w,id in assignment.items()}
    multi_lingual_assignment_grouped = {multi_word_map[w]:group_map[id] for w,id in assignment.items()}
    return multi_lingual_assignment_grouped

df['assignment']=df.apply(get_assignment,axis=1,args=(FULL_CFG,))

# note in sexual orientation i forgot to assign the words with s and g, will have to correct that hardcoding

def correct_assignment(row):
    correct_map = {'f':'s','m':'g'}
    new_assignment = {}
    for w,a in row['assignment'].items():
        if a in ['s','g']:
            new_w = w.split(' ')
            new_w[0] = correct_map[new_w[0]]
            new_assignment[' '.join(new_w)]=a
        else:
            new_assignment[w]=a
    return new_assignment

df['assignment']=df.apply(correct_assignment,axis=1)

def score(row):
    points = []
    for w,a in row['assignment'].items():
        if w[0]==a:
            points.append(1)
        else:
            points.append(-1)
    

df.iloc[243]
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

