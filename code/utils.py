import pandas as pd
import json

def load_files(dataset):
    senteval = ['subj_number', 'top_constituents', 'tree_depth']
    blimp = ['adjunct_island', 'principle_A_c_command', 'passive_1', 'transitive']
    if dataset in senteval:
        data = pd.read_csv(f'{dataset}.txt', sep='\t', header=None)
        TRAIN = data[data[0] == 'tr']
        TEST = data[data[0] == 'te']
        X_train = TRAIN[2]
        X_test = TEST[2]
        y_train = TRAIN[1].values
        y_test = TEST[1].values
        return X_train, y_train, X_test, y_test
    elif dataset == 'person':
        data = pd.read_csv('person.tsv', sep='\t')
        TRAIN = data[data['subset']=='tr']
        TEST = data[data['subset']=='te']
        X_train = TRAIN['text']
        X_test = TEST['text']
        y_train = TRAIN['label'].values
        y_test = TEST['label'].values
        return X_train, y_train, X_test, y_test
    elif dataset == 'conn':
        TRAIN = pd.read_csv('Conn_train.tsv', sep='\t')
        TEST = pd.read_csv('Conn_test.tsv', sep='\t')
        X_train = TRAIN[['sentence_1', 'sentence_2']].values.tolist()
        X_test = TEST[['sentence_1', 'sentence_2']].values.tolist()
        y_train = TRAIN['marker'].values
        y_test = TEST['marker'].values
        return X_train, y_train, X_test, y_test
    elif dataset == 'DC' or dataset == 'SP':
        TRAIN = pd.read_csv(f'{dataset}_train.csv')
        TEST = pd.read_csv(f'{dataset}_test.csv')
        X_train = TRAIN['sentence'].apply(eval)
        X_test = TEST['sentence'].apply(eval)
        y_train = TRAIN['label'].values
        y_test = TEST['label'].values
        return X_train, y_train, X_test, y_test
    elif dataset == 'PDTB':
        TRAIN = pd.read_csv('PDTB_train.csv')
        TEST = pd.read_csv('PDTB_test.csv')
        X_train = TRAIN[['sentence_1', 'sentence_2']].values.tolist()
        X_test = TEST[['sentence_1', 'sentence_2']].values.tolist()
        y_train = TRAIN['label'].values
        y_test = TEST['label'].values
        return X_train, y_train, X_test, y_test
    elif dataset in blimp:
        with open(f'{dataset}.jsonl') as file:
            tasks = list(file)
        text = []
        for i in tasks:
            string = json.loads(i)
            text.append([string['sentence_bad'], string['sentence_good']])
        dataframe = pd.DataFrame(text, columns=['sentence_bad', 'sentence_good'])
        return dataframe
    
