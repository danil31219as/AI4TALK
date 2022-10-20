import pandas as pd
df = pd.read_csv('asr.csv')
idxs =[]
for i in range(len(df)):
    if df.iloc[i]['source'] == '5825.mp3' or df.iloc[i]['source'] == '5850.mp3':
        pass
    else:
        idxs.append(i)
    # print(df.iloc[i]['source'])
    # break

df.drop(index=idxs, inplace=True)
df.reset_index(inplace=True)
df.to_csv('asr_test.csv')