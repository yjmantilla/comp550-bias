import json
import os
import traceback
import time
from pprint import pprint
import numpy as np
import pandas as pd
from openai import OpenAI # this is for the synthetic data
import yaml
import unicodedata
from utils import get_closest_identity

def remove_accents(input_str):
    # Normalize the string to decompose accents
    normalized_str = unicodedata.normalize('NFD', input_str)
    # Remove characters classified as combining marks
    return ''.join(char for char in normalized_str if not unicodedata.combining(char))

############# Base functions to manage the data ##########################################

def save_json(data,f):
    with open(f,'w') as f:
        json.dump(data,f,indent=4)

def load_json(f):
    with open(f,'r') as f:
        return json.load(f)
    

# read prompt yaml file

def read_yaml(f):
    with open(f, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc,flush=True)

sanitize_model = lambda x: x.replace('/','!').replace('-','~').replace('_','&')
retrieve_model = lambda x: x.replace('!','/').replace('~','-').replace('&','_')

data_path='data/'
os.makedirs(data_path,exist_ok=True)
cfg=read_yaml('cfg.yml')
from copy import deepcopy
import time
N_QUERIES = 100
sleep = cfg['sleep']
sleepTimeInner = cfg['sleepTimeInner']
sleepTimeOuter = cfg['sleepTimeOuter']

checkpoint_format = 'task-%task%_domain-bias-%bias%_%domain%_baseline-%baseline%_language-%language%_model-%model%_%datatype%.json'
for llm_model in cfg['models']:
    did_something = 0
    for task, taskcfg in cfg['tasks'].items():
        if taskcfg['do'] == False:
            print(f'Skipping task {task}', flush=True)
            continue
        for language in taskcfg['languages']:
            print(f'Generating samples for {llm_model} in {language} for task {task}', flush=True)
            baseline = taskcfg['baseline']
            domain = taskcfg['domain']
            bias = taskcfg['bias']
            this_lang_prompt = taskcfg['prompts'][language]
            identities = deepcopy(taskcfg['identities'])
            words = deepcopy(taskcfg['words'])
            sanitized_model = sanitize_model(llm_model)
            this_lang_identities = deepcopy(identities)
            for i in list(this_lang_identities.keys()):
                for lj in list(this_lang_identities[i]['variants'].keys()):
                    if lj != language:
                        del this_lang_identities[i]['variants'][lj]
            this_lang_words = deepcopy(words)
            for i in list(this_lang_words.keys()):
                for lj in list(this_lang_words[i].keys()):
                    if lj not in [language,'group']:
                        del this_lang_words[i][lj]
            subcfg_path = os.path.join(data_path,checkpoint_format.replace('%task%',task).replace('%bias%',bias).replace('%baseline%',baseline).replace('%domain%',domain).replace('%language%',language).replace('%model%',sanitized_model).replace('%datatype%','cfg'))
            samples_path = os.path.join(data_path,checkpoint_format.replace('%task%',task).replace('%bias%',bias).replace('%baseline%',baseline).replace('%domain%',domain).replace('%language%',language).replace('%model%',sanitized_model).replace('%datatype%','samples'))
            exceptions_path = os.path.join(data_path,checkpoint_format.replace('%task%',task).replace('%bias%',bias).replace('%baseline%',baseline).replace('%domain%',domain).replace('%language%',language).replace('%model%',sanitized_model).replace('%datatype%','exceptions'))
            code_path = os.path.join(data_path,checkpoint_format.replace('%task%',task).replace('%bias%',bias).replace('%baseline%',baseline).replace('%domain%',domain).replace('%language%',language).replace('%model%',sanitized_model).replace('%datatype%','code'))
            subcfg={'model':llm_model,'task':task,'bias':bias,'domain':domain,'baseline':baseline, 'language':language,'prompt':this_lang_prompt,'identities':this_lang_identities,'words':this_lang_words,}
            save_json(subcfg,subcfg_path)

            if os.path.isfile(samples_path):
                # get the final number of samples
                samples = load_json(samples_path)
                #order keys
                samples = {int(k):v for k,v in samples.items()}
                ordered_keys = sorted(samples.keys())
                # resave
                save_json({k:samples[k] for k in ordered_keys},samples_path)
                # get the last sample
                try:
                    last_sample = ordered_keys[-1]
                except:
                    last_sample = 0
            else:
                samples = {}
                last_sample = 0

            if os.path.isfile(exceptions_path):
                exceptions = load_json(exceptions_path)
                samples_exceptions = {int(k):v for k,v in exceptions.items()}
                ordered_keys_exceptions = sorted(samples_exceptions.keys())
                save_json({k:samples_exceptions[k] for k in ordered_keys_exceptions},exceptions_path)

                try:
                    last_sample_exceptions = ordered_keys[-1]
                except:
                    last_sample_exceptions = 0
            else:
                exceptions = {}
                last_sample_exceptions = 0
            
            if last_sample >= N_QUERIES:
                print(f'Already generated {N_QUERIES} samples for {llm_model} in {language} for task {task}', flush=True)
                continue
            # generate samples
            index = last_sample
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            count = 1
            exception_count = last_sample_exceptions
            while True:
                content = 'No content yet'
                start = time.time()
                try:
                    this_lang_identities_shuffled = list(this_lang_identities.keys())
                    np.random.shuffle(this_lang_identities_shuffled)
                    this_lang_identities_shuffled = [this_lang_identities[i]['variants'][language] for i in this_lang_identities_shuffled]
                    while True:
                        newids=[]
                        for i in range(len(this_lang_identities_shuffled)):
                            newids.append(np.random.choice(this_lang_identities_shuffled[i]))
                        if len(set(newids)) == len(this_lang_identities_shuffled):
                            this_lang_identities_shuffled = newids
                            break
                    identities_str = ' or '.join(this_lang_identities_shuffled)

                    this_lang_words_shuffled = list(this_lang_words.keys())
                    np.random.shuffle(this_lang_words_shuffled)
                    this_lang_words_shuffled = [this_lang_words[i][language] for i in this_lang_words_shuffled]
                    words_str = ' , '.join(this_lang_words_shuffled)
                    this_prompt = this_lang_prompt.replace('%identities%',identities_str).replace('%words%',words_str)

                    completion = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "Follow what the user requests. Do not add suggestions or ask questions about what the user wants at the end. Just do as you are told. DO NOT announce your answer or suggest anything or add explanatory text about your answer nor comments. Here you are not an assistant, you are a text generator."},
                        {"role": "user", "content": this_prompt}
                    ],
                    temperature=0.8, # May have been interesting to play with this parameter, but I did not have time
                    n=1
                    )
                    content=completion.choices[0].message.content
                    content = remove_accents(content)
                    # assert all the words are in the content
                    for w in this_lang_words_shuffled:
                        assert w.lower() in content.lower()

                    # assert all the identities are in the content at least 1/n the words times
                    # seems to be too strict ?
                    # commented because maybe be skewing the results
                    # for id in this_lang_identities_shuffled:
                    #     assert content.lower().count(id.lower()) >= len(this_lang_words)/4 # -1 minus one to be more flexible?

                    # sum of identities in content should be equal to the number of words
                    sorted_identities = sorted(this_lang_identities_shuffled, key=len, reverse=True)
                    temp_content = content.lower()
                    sum_identities = 0
                    for id in sorted_identities:
                        count = temp_content.count(id.lower())
                        sum_identities += count
                        temp_content = temp_content.replace(id.lower(), ' ' * len(id))
                    assert sum_identities == len(this_lang_words_shuffled)

                    # check closest identity is computable

                    assignment = get_closest_identity(content,this_lang_words_shuffled,this_lang_identities_shuffled)


                    # one was "home:ben-julia,parents:ben-julia,children:ben-julia,family:ben-julia,marriage:ben-julia,wedding:ben-julia,relatives:ben-julia,management:ben-julia,professional:julia-ben,corporation:julia-ben,salary:ben-julia,office:julia-ben,business:julia-ben,career:julia-ben"
                    # so TODO: make sure that each word has only one identity associated with it

                    # remove ( ) from identities dict
                    index += 1
                    count += 1
                    sample = {'prompt': this_prompt,'answer': content,'model':llm_model,'language':language,'task':task,'bias':bias,'domain':domain,'baseline':baseline,'words':this_lang_words_shuffled,'identities':this_lang_identities_shuffled,}
                    samples[index] = sample
                except Exception as e:
                    exception_count += 1
                    error_str = traceback.format_exc()
                    print(error_str, flush=True)
                    exceptions[exception_count] = {'model':llm_model,'task':task,'bias':bias,'domain':domain,'baseline':baseline,'language':language,'prompt':this_prompt,'words':this_lang_words_shuffled,'identities':this_lang_identities_shuffled,'answer':content,'exception':traceback.format_exc()}

                end = time.time()
                print(index,count,exception_count,f'Elapsed time: {end - start}', flush=True)
                save_json(samples,samples_path)
                save_json(exceptions,exceptions_path)
                if index >= N_QUERIES:
                    break
            if sleep:
                time.sleep(sleepTimeInner) # to avoid burning the gpu
            did_something += 1
        
        if did_something == 0:
            print(f'No new samples generated for {llm_model} in task {task}', flush=True)
        else:
            # save code used to generate the samples
            with open('generate.py','r') as f:
                code = f.read()
            with open(code_path,'w') as f:
                f.write(code)
            print(f'Generated new samples for {llm_model} in task {task}', flush=True)
            if sleep:
                time.sleep(sleepTimeOuter) # to avoid burning the gpu