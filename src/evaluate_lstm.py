import numpy as np
from pathlib import Path
import pickle
import json
from lstm import SimpleLSTM
from metrics import exact_match_metric
from tensorflow.keras.optimizers import Adam
from generators import DataGeneratorSeq


class EvaluateSimpleLSTM:
    
    def __init__(self, path, data_gen_pars):
        
        with open(str(path/'settings.json'), 'r') as file:
            self.settings_dict = json.load(file)

        self.token_index = data_gen_pars['token_index']
        self.num_tokens = len(self.token_index)
        
        adam = Adam(lr=6e-4, beta_1=0.9, beta_2=0.995, epsilon=1e-9, decay=0.0, amsgrad=False, clipnorm=0.1)
        print('params', self.num_tokens, self.settings_dict['latent_dim'])
        self.lstm = SimpleLSTM(self.num_tokens, self.settings_dict['latent_dim'])
        _ = self.lstm.get_model()
        self.lstm.model.load_weights(str(path/'model.h5'))
        self.lstm.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[exact_match_metric])
        
    def evaluate_model(self, input_texts, output_texts, teacher_forcing=True, batch_size=128, n_samples=1000):
        max_seq_length  = max([len(txt_in)+len(txt_out) for txt_in, txt_out in zip(input_texts,output_texts)])
        
        params = {'batch_size': batch_size,
                  'max_seq_length': max_seq_length,
                  'num_tokens': self.num_tokens,
                  'token_index': self.token_index,
                  'num_thinking_steps': self.settings_dict["thinking_steps"]
                 }
        
        self.data_generator = DataGeneratorSeq(input_texts=input_texts,
                                               target_texts=output_texts,
                                               **params)
        
        if not teacher_forcing:
            outputs_true, outputs_preds = self.predict_without_teacher(n_samples, max_seq_length)
            exact_match = len([0 for out_true, out_preds in zip(outputs_true, outputs_preds) if out_true.strip()==out_preds.strip()])/len(outputs_true)
        
        else:
            result = self.lstm.model.evaluate_generator(self.data_generator, verbose=0)
            exact_match = result[1]
            
        return exact_match
    
    def predict_on_string(self, text, max_output_length=100):
        
        max_seq_length = len(text) + max_output_length

        
        params = {'batch_size': 1,
                  'max_seq_length': max_seq_length,
                  'num_tokens': self.num_tokens,
                  'token_index': self.token_index,
                  'num_thinking_steps': self.settings_dict["thinking_steps"]
                 }
        
        
        self.data_generator = DataGeneratorSeq(input_texts=[text],
                                               target_texts=['0'*max_output_length],
                                               **params)
        
        outputs_true, outputs_preds = self.predict_without_teacher(1, max_seq_length)
        
        return outputs_preds[0].strip()

    def predict_without_teacher(self, n_samples, max_seq_length, random=True):
        
        encoded_texts = [] 
        outputs_true = []
        if random:
            samples = np.random.choice(self.data_generator.indexes, n_samples, replace=False)
        else:
            samples = list(range(n_samples))
        for i in samples:
            sample = self.data_generator._DataGeneratorSeq__data_generation([i])         
            input_len = len(self.data_generator.input_texts[i])
            outputs_true.append(self.data_generator.target_texts[i])
            x = sample[0][0][:input_len+self.settings_dict["thinking_steps"]+1]
            encoded_texts.append(np.expand_dims(x, axis=0))
            
        outputs_preds = self.lstm.decode_sample(encoded_texts, self.token_index, max_seq_length)
        return outputs_true, outputs_preds

    def __get_stoi_from_data(self):
        """
        This function reloads all the data that was used to train and evalute
        model to construct the string to integer map (stoi).
        """
        
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
        
        raw_path = Path(self.settings_dict['data_path'])
        interpolate_path = raw_path/'interpolate'
        extrapolate_path = raw_path/'extrapolate'
        train_easy_path = raw_path/'train-easy/'
        math_module = self.settings_dict["math_module"]
        train_level = self.settings_dict["train_level"]
        datasets = {
            'train':(raw_path, 'train-' + train_level + '/' + math_module),
            'interpolate':(interpolate_path, math_module),
            'extrapolate':(extrapolate_path, math_module)
                   }

        input_texts = {}
        target_texts = {}

        for k, v in datasets.items():
            input_texts[k], target_texts[k] = concatenate_texts(v[0], v[1])
        
        all_input_texts = sum(input_texts.values(), [])
        all_target_texts = sum(target_texts.values(), [])

        input_characters = set(''.join(all_input_texts))
        target_characters = set(''.join(all_target_texts))

        tokens = sorted(list(input_characters | target_characters))
        token_index = dict([(char, i) for i, char in enumerate(tokens)])
        
        return token_index
        
        
        

