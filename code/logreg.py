import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pickle


class LogRegClassification(object):
    def __init__(self, dir_path, output_file, random=False):
        self.tasks = ["conn", "PDTB", "DC", "SP", "subj_number",
                      "top_constituents", "tree_depth", "person"]
        self.dir_path = dir_path
        self.output_file = output_file
        self.random = random
        self.enc = LabelEncoder()
        self.logreg = LogisticRegression

    def load_files(self, dataset):
        senteval = ['subj_number', 'top_constituents', 'tree_depth']
        if dataset in senteval:
            data = pd.read_csv(f'{dataset}.txt', sep='\t', header=None)
            TRAIN = data[data[0] == 'tr']
            TEST = data[data[0] == 'te']
            X_train = TRAIN[2]
            X_test = TEST[2]
            y_train = TRAIN[1].values
            y_test = TEST[1].values
        elif dataset == 'person':
            data = pd.read_csv('person.tsv', sep='\t')
            TRAIN = data[data['subset'] == 'tr']
            TEST = data[data['subset'] == 'te']
            X_train = TRAIN['text']
            X_test = TEST['text']
            y_train = TRAIN['label'].values
            y_test = TEST['label'].values
        elif dataset == 'conn':
            TRAIN = pd.read_csv('Conn_train.tsv', sep='\t')
            TEST = pd.read_csv('Conn_test.tsv', sep='\t')
            X_train = TRAIN[['sentence_1', 'sentence_2']].values.tolist()
            X_test = TEST[['sentence_1', 'sentence_2']].values.tolist()
            y_train = TRAIN['marker'].values
            y_test = TEST['marker'].values
        elif dataset == 'DC' or dataset == 'SP':
            TRAIN = pd.read_csv(f'{dataset}_train.csv')
            TEST = pd.read_csv(f'{dataset}_test.csv')
            X_train = TRAIN['sentence'].apply(eval)
            X_test = TEST['sentence'].apply(eval)
            y_train = TRAIN['label'].values
            y_test = TEST['label'].values
        else:
            TRAIN = pd.read_csv('PDTB_train.csv')
            TEST = pd.read_csv('PDTB_test.csv')
            X_train = TRAIN[['sentence_1', 'sentence_2']].values.tolist()
            X_test = TEST[['sentence_1', 'sentence_2']].values.tolist()
            y_train = TRAIN['label'].values
            y_test = TEST['label'].values
        return X_train, y_train, X_test, y_test

    def load_data(self, path):
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        return np.asarray(data)

    def classify(self, X_train, X_test, y_train, y_test):
        """
        Trains a logistic regression and predicts labels
        :return: metrics of logistic regression perfomance
        """
        logreg = self.logreg()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        return [y_pred, accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='micro'),
                recall_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'), logreg]

    def write_to_files(self, pred, scores, task):
        """
        Saves results to a file
        :param pred:
        :return:
        """
        pred = '\n'.join([','.join(list(map(str, i))) for i in pred])
        with open(f'pred_{task}.txt', 'a') as f:
            f.write(pred)
        sc = pd.DataFrame(scores, columns=['task', 'epoche', 'layer', 'accuracy', 'precision', 'recall', 'f1-score'])
        if os.path.isfile(self.output_file):
            sc.to_csv(self.output_file, mode='a', header=False)
        else:
            sc.to_csv(self.output_file, mode="w", header=True)
        return sc

    def probe(self, checkpoints, y_train, y_test, task):
        predictions = []
        all_scores = []
        for train, test in tqdm(checkpoints):
            scores = []
            pred = []
            epoche = int(train.split('.')[0].split('_')[-1])
            TRAIN = self.load_data(train)
            TEST = self.load_data(test)
            y_train = self.enc.fit_transform(y_train)
            y_test = self.enc.transform(y_test)
            layer_num = TRAIN.shape[2]
            for layer in range(layer_num):
                X_train = TRAIN[:, :, layer]
                X_test = TEST[:, :, layer]
                sc = self.classify(X_train, X_test, y_train, y_test)
                pred.append(sc[0])
                score = [task, epoche, layer, ] + sc[1:5]
                all_scores.append(score)
                scores.append(score)
            predictions.append(pred)
            sc = self.write_to_files(predictions, scores, task)
        return predictions, all_scores

    def filter_(self, task):
        checkpoints = []
        for file in os.listdir(self.dir_path):
            if file.startswith(f"BERT_checkpoints_{task}_TRAIN"):
                train_path = os.path.join(self.dir_path, file)
                test_path = os.path.join(self.dir_path, file.replace("TRAIN", "TEST"))
                checkpoints.append([train_path, test_path])
        return checkpoints

    def run_logreg(self):
        for task in self.tasks:
            print(f"Calculating {task} task...")
            _, y_train, _, y_test = self.load_files(task)
            checkpoints = self.filter_(task)
            if self.random:
                np.random.shuffle(y_train)
                np.random.shuffle(y_test)
            self.probe(checkpoints, y_train, y_test, task)

