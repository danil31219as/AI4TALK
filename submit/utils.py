from torch.nn.utils.rnn import pad_sequence
import json
from transformers import Wav2Vec2Model, Wav2Vec2Config
import torch
# import torchaudio
import pandas as pd
from tqdm import tqdm
import numpy as np
import librosa


class Tokenizer:
    def __init__(self, vocab_path):
        self.vocab = json.load(open(vocab_path))
        self.token_id = sorted(list(self.vocab.items()), key=lambda x: -len(x[0]))
        self.id_token = {id_: token for token, id_ in self.vocab.items()}
    
    def __call__(self, sentence):
        for token, id_ in self.token_id:
            sentence = sentence.replace(token, f' {id_} ')
        return torch.LongTensor(list(map(int, sentence.split())))
    
    def decode(self, sequence):
        result = ''
        for id_ in sequence:
            if self.id_token[id_] != '<pad>':
                result += self.id_token[id_]
        return result
    
    def batch_decode(self, batch):
        result = []
        for sequence in batch:
            result.append(self.decode(sequence))
        return result
    
    def pad(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.vocab['<pad>'])
    
    @property
    def vocab_size(self):
        return len(self.vocab)
class TalkDatasetTest(torch.utils.data.Dataset):
    def __init__(self, df, feature_extractor, augmentations=None, stage='train'):
        super().__init__()
        self.df = df
        self.feature_extractor = feature_extractor
        self.sr = 16000
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.df)
    
    def cut(self):
        audios = [0]*len(self.df)
        files = set(self.df['source'])
        sources = {i:[] for i in files}
        for row in self.df.iloc:
            sources[row['source']].append([row['id'],round(self.sr* row['start']), round(self.sr*row['end'])])
        for key in sources.keys():
            audio = librosa.load(row['path'], sr=self.sr)[0]
            for v in sources[key]:
                cut_audio = audio[v[1]:v[2]]
                audios[v[0]] = cut_audio
                # audios.append(cut_audio)
        self.df['audio'] = audios
        
    def __getitem__(self, idx):
        # path = self.df.iloc[idx]['audio_path']
        
        # waveform, sample_rate = torchaudio.load(path)
        waveform = torch.from_numpy(self.df['audio'][idx])
        if self.augmentations:
            waveform = self.augmentations(waveform)
        sample = {}
        sample['input_values'] = self.feature_extractor(waveform)
        return sample



class WaveTextCollatorTest:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor  
        # self.tokenizer = tokenizer
        
    def __call__(self, batch):
        # input_values = [{'input_values': feature['input_values']} for feature in batch]
        input_values = [feature['input_values'] for feature in batch]
        input_values, attention_mask = self.feature_extractor.pad(input_values)
        
        return input_values, attention_mask




class FeatureExtractor:
    def __call__(self, x):
        return (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    
    def pad(self, input_values):
        attention_mask = [torch.ones(x.size(0)) for x in input_values]
        input_values = pad_sequence(input_values, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # return {'input_values':input_values, 'attention_mask':attention_mask}
        return input_values, attention_mask




class Wav2VecCTC(torch.nn.Module):
    def __init__(self, vocab_size, dropout=0.0):
        super().__init__()
        self.config = Wav2Vec2Config.from_pretrained('config.json')
        self.encoder = Wav2Vec2Model(self.config)
        self.encoder.config.mask_time_length = 1
        self.dropout = torch.nn.Dropout(dropout)
        output_hidden_size = self.encoder.config.hidden_size
        self.lm_head = torch.nn.Linear(output_hidden_size, vocab_size)
        
    def forward(self, input_values, attention_mask):
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)
        return logits

def test(model, loader, device, tokenizer):
    model.eval()
    preds = []
    with torch.no_grad():
        for input_values, attention_mask in tqdm(loader):
            output = model(input_values, attention_mask)
            pred_ids = torch.argmax(output.detach().cpu(), dim=-1).numpy()
            pred_str = tokenizer.batch_decode(pred_ids)
            preds.append(pred_str)
    
    return preds
