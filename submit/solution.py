import pandas as pd
import os
# import torchaudio
from torch.utils.data import DataLoader
import torch.nn as nn
import json
import sys
from utils import TalkDatasetTest, WaveTextCollatorTest, Wav2VecCTC, Tokenizer, FeatureExtractor, test

args = sys.argv[1:]
WEIGHTS_PATH = args[0]
TOKENIZER_PATH = args[1]
DF_PATH = args[2]
DEVICE ='cuda:0'
BATCH_SIZE=1
os.makedirs('./output', exist_ok=True)

df = pd.read_csv(DF_PATH).rename(columns={'new_path': 'audio_path'})
df = df.drop_duplicates(subset=['start', 'end']).reset_index(drop=True)
df['path'] = [os.path.join('../input/', i) for i in df['source']]

tokenizer = Tokenizer(os.path.join(TOKENIZER_PATH, 'vocab.json'))

feature_extractor = FeatureExtractor()
test_dataset = TalkDatasetTest(df, feature_extractor)
collator = WaveTextCollatorTest(feature_extractor, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=True)
model = Wav2VecCTC(tokenizer.vocab_size)
model.load('pretrained.pt')
preds = test(model, test_loader, tokenizer, device=DEVICE)

df["transcription"].str.replace(' ','')
df["transcription"] = df["transcription"].apply(lambda v: v if v else "SIL")
df.to_csv("./output/asr-solution.csv", index=False)


