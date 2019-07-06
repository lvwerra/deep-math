#!/usr/bin/env python
# coding: utf-8

# ## Concatenate all files
# 
# ```bash
# $ cd path/to/train-easy/
# $ find -name '*.txt' -exec cat {} \; > ../../../interim/train-easy_all.txt
# ```

# ## Load libraries

# In[69]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[70]:


import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import pickle
import json
import matplotlib.pyplot as plt
from lstm import LSTM_S2S
from metrics import exact_match_metric
from callbacks import NValidationSetsCallback, GradientLogger
from generator import DataGenerator

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())


# ## Load settings

# In[71]:


settings_path = Path('../../settings/settings.json')


# In[73]:


with open(str(settings_path), 'r') as file:
    settings_dict = json.load(file)


# In[74]:


settings_dict


# ## Load data
# 
# Start with batching a single file before tackling the whole dataset.

# In[4]:


raw_path = Path(settings_dict['data_path'])
get_ipython().system('ls {raw_path}')


# In[5]:


interpolate_path = raw_path/'interpolate'
get_ipython().system('ls {interpolate_path} | head -5')


# In[6]:


extrapolate_path = raw_path/'extrapolate'
get_ipython().system('ls {extrapolate_path} | head -5')


# In[7]:


train_easy_path = raw_path/'train-easy/'
get_ipython().system('ls {train_easy_path} | head -5')


# In[8]:


def concatenate_texts(path, pattern):
    file_paths = list(path.glob('{}*.txt'.format(pattern)))
    
    input_texts = []
    target_texts = []

    for file_path in file_paths:
        with open(str(file_path), 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]

        input_texts.extend(lines[0::2])
        target_texts.extend(['\t' + target_text + '\n' for target_text in lines[1::2]])
        
    return input_texts, target_texts


# ### Data settings

# In[9]:


math_module = settings_dict["math_module"]
train_level = settings_dict["train_level"]


# In[28]:


datasets = {
    'train':(raw_path, 'train-' + train_level + '/' + math_module),
    'interpolate':(interpolate_path, math_module),
    'extrapolate':(extrapolate_path, math_module)
           }


# In[29]:


get_ipython().run_cell_magic('time', '', "\ninput_texts = {}\ntarget_texts = {}\n\nfor k, v in datasets.items():\n    input_texts[k], target_texts[k] = concatenate_texts(v[0], v[1])\n    print('Length of set {} is {}'.format(k, len(input_texts[k])))")


# **Sample:**

# In[30]:


print('INPUT:', input_texts['train'][42])
print('OUTPUT:', target_texts['train'][42].strip())


# Concatenate texts to get text metrics (max length, number of unique tokens, etc.):

# In[31]:


all_input_texts = sum(input_texts.values(), [])
all_target_texts = sum(target_texts.values(), [])


# In[32]:


input_characters = set(''.join(all_input_texts))
target_characters = set(''.join(all_target_texts))


# In[33]:


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in all_input_texts])
max_decoder_seq_length = max([len(txt) for txt in all_target_texts])

print('Number of samples:', len(all_input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# ### Delete all texts to realease memory

# In[34]:


del all_input_texts
del all_target_texts


# ## Create train test splits

# In[35]:


input_texts_train, input_texts_valid, target_texts_train, target_texts_valid = train_test_split(input_texts['train'], target_texts['train'], test_size=0.2, random_state=42)


# In[36]:


print('Number of training samples:', len(input_texts_train))


# In[37]:


print('Number of validation samples:', len(input_texts_valid))


# ## Process text

# ### Vectorise the text
# Before training, we need to map strings to a numerical representation. Create two lookup tables: one mapping question characters to numbers, and another for answer characters to number.

# In[38]:


# Creating a mapping from unique characters to indices
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# In[39]:


target_token_index


# ## Create keras data generator

# In[40]:


# Parameters
params = {'batch_size': settings_dict["batch_size"],
          'max_encoder_seq_length': max_encoder_seq_length,
          'max_decoder_seq_length': max_decoder_seq_length,
          'num_encoder_tokens': num_encoder_tokens,
          'num_decoder_tokens': num_decoder_tokens,
          'input_token_index': input_token_index,
          'target_token_index': target_token_index,
          'num_thinking_steps': settings_dict["thinking_steps"]
         }


# In[41]:


training_generator = DataGenerator(input_texts=input_texts_train, target_texts=target_texts_train, **params)
validation_generator = DataGenerator(input_texts=input_texts_valid, target_texts=target_texts_valid, **params)
interpolate_generator = DataGenerator(input_texts=input_texts['interpolate'], target_texts=target_texts['interpolate'], **params)
extrapolate_generator = DataGenerator(input_texts=input_texts['extrapolate'], target_texts=target_texts['extrapolate'], **params)


# ## Train model

# In[42]:


valid_dict = {
    'validation':validation_generator,
    'interpolation': interpolate_generator,
    'extrapolation': extrapolate_generator
}


# In[43]:


history = NValidationSetsCallback(valid_dict)
gradient = GradientLogger(live_metrics=['loss', 'exact_match_metric'], live_gaps=10)


# In[44]:


epochs = settings_dict['epochs']  # Number of epochs to train for.
latent_dim = settings_dict['latent_dim']  # Latent dimensionality of the encoding space.


# In[45]:


lstm = LSTM_S2S(num_encoder_tokens, num_decoder_tokens, latent_dim)


# In[46]:


model = lstm.get_model()


# In[47]:


adam = Adam(lr=6e-4, beta_1=0.9, beta_2=0.995, epsilon=1e-9, decay=0.0, amsgrad=False, clipnorm=0.1)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[exact_match_metric])
print('start training...')
train_hist = model.fit_generator(training_generator,
                                 epochs=epochs,
                                 #use_multiprocessing=True, workers=8,
                                 callbacks=[history, gradient],
                                 verbose=0,
                                )


# In[1]:


plt.plot(train_hist.history['loss'],color='C0', label='train')
plt.plot(train_hist.history['validation_loss'], color='C0', label='valid', linestyle='--')
plt.plot(train_hist.history['extrapolation_loss'], color='C1', label='extra',)
plt.plot(train_hist.history['interpolation_loss'], color='C2', label='inter')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.ylim([0,1])
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(settings_dict['save_path'] + 'losses.png', dpi=300)


# In[ ]:


plt.plot(train_hist.history['exact_match_metric'],color='C0', label='train')
plt.plot(train_hist.history['validation_exact_match_metric'], color='C0', label='valid', linestyle='--')
plt.plot(train_hist.history['extrapolation_exact_match_metric'], color='C1', label='extra',)
plt.plot(train_hist.history['interpolation_exact_match_metric'], color='C2', label='inter')

plt.xlabel('epochs')
plt.ylabel('exact match metric')
plt.legend(loc='best')
plt.ylim([0,1])
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(settings_dict['save_path'] + 'metrics.png', dpi=300)


# In[50]:


with open(settings_dict['save_path']+'experiments_output.pkl','wb') as file:
    pickle.dump(train_hist.history, file)


# In[53]:


model.save(settings_dict['save_path']+'model.h5')


# In[78]:


with open(settings_dict['save_path']+'settings.json','w') as file:
    json.dump(settings_dict, file)

