#!/usr/bin/env python
# coding: utf-8

# In[ ]:


output_filepath = input('Enter filepath to output documents: ')
input_filepath = input('Enter filepath of audio file to be checked: ')
audio_fps = float(input('Enter framerate of audio file: '))


# In[22]:


import os
import whisper
import re
import ffmpeg
import torch.cuda
import datetime
import pandas as pd
from pathlib import Path
from whisper.utils import write_srt


# In[23]:


def whisper_transcription(input_filepath):
    x = str(input_filepath)
    if x[0]=="'" or x[0]=='"':
        x = x[1:-1]
    model = whisper.load_model("medium.en")
    audio = whisper.load_audio(str(x))
    result = model.transcribe(audio)
    return(result)


def seconds_to_TC (seconds, fps):
    '''function gets the seconds where dx occurs from whisper dict returns formatted timecodes'''
    framerate = int(round(fps))
    frames = seconds * framerate
    if framerate == 50 or framerate ==60:
        framerate=framerate/2
    frames = round(seconds * framerate)
    hours = frames // (60*framerate*60)
    frames -= hours * (60*framerate*60)
    minutes = frames // (60*framerate)
    frames -= minutes * (60*framerate)
    second = frames // framerate
    frames -= second * framerate
    if framerate%25==0:
        hours+=10
    else: 
        hours+=1
    return ( "%02d:%02d:%02d:%02d" % ( hours, minutes, second, frames), framerate)


def profanity_check(dialogue_text):
    """this function loops through profanity 
    list and each word in segment text, returning flag in 
    cases where profanity is present"""
    profanity_list = ['assholes',
                         'aholes',
                         'arses',
                         'bastards',
                         'bitchings',
                         'bullshits',
                         'cock',
                         'cocks',
                         'cunts',
                         'dicks',
                         'fubars',
                         'fucks',
                         'fuckers',
                         'milfs',
                         'mother fuckers',
                         'pimps',
                         'prostitutes',
                         'shits',
                         'sluts',
                         'sobs',
                         'whores',
                         'balls',
                         'bums',
                         'bungholes',
                         'butts',
                         'buttocks',
                         'cocks',
                         'dicks',
                         'peckers',
                         'cows',
                         'craps',
                         'crotches',
                         'fags',
                         'faggots',
                         'hos',
                         'hoes',
                         'hookers',
                         'hooters',
                         'midgets',
                         'mongs',
                         'tard',
                         'nads',
                         'nards',
                         'nuts',
                         'orgasms',
                         'pigs',
                         'pussies',
                         'pricks',
                         'retards',
                         'savages',
                         'tards',
                         'sucks',
                         'turds',
                         '*s',
                         '***s',
                         '****s',
                         'f***s',
                         's***s',
                         'b*****s',
                        'Asshole',
                        'ahole',
                        'Arse',
                        'Bastard',
                        'Bitch',
                        'Bitched',
                        'Bitching', 
                        'Bitches',
                        'Bitchy',
                        'Bollocks',
                        'Bugger',
                        'BS',
                        'Bullshit',
                        'Cunt',
                        'Damn',
                        'Dammit',
                        'Dick',
                        'effed',
                        'fing',
                        'effin',
                        'effing',
                        'FWord',
                        'Feck',
                        'Freaking',
                        'Freakin',
                        'Fricking',
                        'Frickin',
                        'Frigging',
                        'Friggin',
                        'FUBAR',
                        'Fuck',
                        'Fucker',
                        'Fucked',
                        'Fuckin',
                        'Fucking',
                        'God Damn',
                        'Goddamn',
                        'Goddam',
                        'Goddammit',
                        'Goddamnit',
                        'God dammit',
                        'MILF',
                        'Mother Fuck',
                        'Mother Fucker',
                        'Pimp',
                        'Prostitute',
                        'Shit',
                        'Shat',
                        'Shitted',
                        'Slut',
                        'SOB',
                        'Tits',
                        'Whore',
                        'Ass',
                        'Ball',
                        'Bejesus',
                        'Bitch',
                        'Blow',
                        'Bottom',
                        'Bum',
                        'Bunghole',
                        'Butt',
                        'Buttock',
                        'Christ',
                        'Cock',
                        'Dick',
                        'Pecker',
                        'Junk',
                        'Coconuts',
                        'Cow',
                        'Crap',
                        'Crotch',
                        'Fag',
                        'Faggot',
                        'Ho',
                        'Hoe',
                        'Holy',
                        'Hooker',
                        'Hooter',
                        'Humping',
                        'Jesus',
                        'Jesus Christ',
                        'Mary',
                        'Midget',
                        'Mong',
                        'Mother Mary',
                        'Nad',
                        'Nard',
                        'Nut',
                        'Orgasm',
                        'Pig',
                        'Pussy',
                        'Prick',
                        'Retard',
                        'Savage',
                        'Semen',
                        'Suck',
                        'Turd',
                        '*',
                        '***',
                        '****',
                        'f***',
                        'S***',
                        'b*****'
                        ]
    term_list = [str(x.lower()) for x in profanity_list]
    ptup = tuple(term_list)
    text = [x.lower() for x in dialogue_text]
    char_remove = ['.',',','?','!']
    for char in char_remove:
        text = [x.replace(char,'') for x in text]
#     text = [x.replace('.','')for x in text]
#     text = [x.replace(',','')for x in text]
#     text = [x.replace('?','')for x in text]
#     text = [x.replace('!','')for x in text]
    dx_tup = tuple(text)
    terms = list(set(ptup).intersection(set(dx_tup)))
    return terms
                    

def profanity_segments(result_segments, framerate):
    """
    function iterates through all whisper
    segments formatting timecode,dialogue and profanity flag.
    """
    segment_list = []
    dialogue_list = []
    timecode_list = []
    prof_list = []
    for idx, segment in enumerate(result_segments):
        segment_list.append(idx + 1)
        seconds=segment.get('start')
        start_ = seconds_to_TC(seconds, framerate)
        seconds=segment.get('end')
        end_ = seconds_to_TC(seconds, framerate)
        start_ = str(start_)
        end_ = str(end_)
        timecode_list.append(f"{start_}--->{end_}")
        text = segment.get('text').strip().lower()
        dialogue_list.append(f"{text}")
        prof_text = segment.get('text').split(" ")
        profanity_flag = profanity_check(prof_text)
        prof_list.append(profanity_flag)
    return segment_list, timecode_list, dialogue_list, prof_list


def process_dialogue(input_filepath, output_filepath, framerate):
    result = whisper_transcription(input_filepath)
    script_df = pd.DataFrame()
    script_df['segment #'], script_df['timecode'], script_df['dialogue'], script_df['profanity_flag'] = profanity_segments(result['segments'], framerate)
    script_df['flags'] = script_df.profanity_flag.apply(lambda x: len(x))
    audio_basename = Path(input_filepath).stem
    output_folder = output_filepath / Path(audio_basename + '_output')
    os.mkdir(output_folder) 
    os.chdir(output_folder)
    with open(output_folder / Path(audio_basename + "_subs.txt"), "w", encoding="utf-8") as srt:
        write_srt(result["segments"], file=srt)
    torch.cuda.empty_cache()  
    torch.backends.cuda.cufft_plan_cache.clear()
    pos_prof_df = script_df[script_df['flags']>=1]
    pos_prof_df.to_csv(output_folder / Path("_profanity.csv"), encoding='utf-8', index=False)
    count_df = script_df['profanity_flag'].value_counts()
    count_df.to_csv(output_folder / Path("_count.csv"), encoding='utf-8', index=False)
    script_df.to_csv(output_folder / Path("_full.csv"), encoding='utf-8', index=False)


# In[30]:


process_dialogue(input_filepath, output_filepath, audio_fps)


# In[ ]:




