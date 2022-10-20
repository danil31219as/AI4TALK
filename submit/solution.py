import pandas as pd
import os
import warnings
warnings.simplefilter("ignore")
# import torchaudio
from torch.utils.data import DataLoader
# import torch.nn as nn
import torch
import json
import sys
from utils import TalkDatasetTest, WaveTextCollatorTest, Wav2VecCTC, Tokenizer, FeatureExtractor, test


args = sys.argv[1:]
WEIGHTS_PATH = args[0]
TOKENIZER_PATH = args[1]
DF_PATH = args[2]
DEVICE ='cuda'
BATCH_SIZE=1
os.makedirs('./output', exist_ok=True)

df = pd.read_csv(DF_PATH).rename(columns={'new_path': 'audio_path'})

df['path'] = [os.path.join('../input/', i) for i in df['source']]

tokenizer = Tokenizer(os.path.join(TOKENIZER_PATH, 'vocab.json'))

feature_extractor = FeatureExtractor()
test_dataset = TalkDatasetTest(df, feature_extractor)
collator = WaveTextCollatorTest(feature_extractor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=True)
model = Wav2VecCTC(tokenizer.vocab_size)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device(DEVICE))['model'])

preds = test(model=model, 
            loader = test_loader, 
            tokenizer = tokenizer,
             device=DEVICE)
df['transcription'] = preds
df["transcription"].str.replace(' ','')
df["transcription"] = df["transcription"].apply(lambda v: v if v else "SIL")
df.drop(columns=['audio'])
df.to_csv("../output/asr-solution.csv", index=False)