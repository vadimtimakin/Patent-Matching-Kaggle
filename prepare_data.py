import numpy as np
import pandas as pd
import os
import re
from sklearn import model_selection


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["score"], bins=num_bins, labels=False
    )

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedGroupKFold(n_splits=num_splits, shuffle=True, random_state=0xFACED)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['bins'].values, groups=data['anchor'].values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop("bins", axis=1)
    # return dataframe with folds 
    return data


def get_cpc_texts():
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in os.listdir('/home/toefl/K/ptp/dataset/context/CPCSchemeXML202105'):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(f'/home/toefl/K/ptp/dataset/context/CPCTitleList202202/cpc-section-{cpc}_20220201.txt') as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results


cpc_texts = get_cpc_texts()

# read training data
df = pd.read_csv("/home/toefl/K/ptp/dataset/train.csv")
df['context_text'] = df['context'].map(cpc_texts)

# create folds
df = create_folds(df, num_splits=5)

print(df.kfold.value_counts())
print(df.head())

df.to_csv("train_folds.csv", index=False)