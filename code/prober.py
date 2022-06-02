import os
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from tqdm import tqdm
import pickle
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import json
import re


class Embeddings(object):
    def __init__(self, device, tokenizer_path,
                 output_path, delay=0):
        self.device = device
        self.output_path = output_path
       # self.checkpoints = self.get_checkpoints(dir_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.delay = delay

    def get_checkpoints(self, dir_path):
        checkpoints = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if re.match("checkpoint.*", file)]
        return checkpoints

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

    def load_model(self, checkpoint_path):
        """
        Loads a transformer model
        :return: a model and a tokenizer
        """
        model = AutoModel.from_pretrained(checkpoint_path, output_hidden_states=True)
        model = model.to(self.device)
        return model

    def get_emb(self, sent, model):
        """
        Encodes a sentence and returns an embedding
        :param sent: a sentence, str
        :param model: a transformer model
        :return:
        """
        with torch.no_grad():
            enc = self.tokenizer(sent, padding=True, truncation=True,
                                 max_length=512, return_tensors='pt')
            enc = enc.to(self.device)
            output = model(**enc, return_dict=True)
        states = output.hidden_states
        mean_pool = np.zeros((model.config.hidden_size,
                              model.config.num_hidden_layers + 1))
        for num, emb in enumerate(states):
            mean_pool[:, num] = torch.mean(emb, 1).squeeze(0).cpu().numpy()
        return mean_pool

    def calculate_embeddings(self, model, data):
        """
        Calculates embeddings for all sentences
        :param model: a path to a checkpoint
        :param data: a corpus of texts
        :return: a matrix of embeddings
        """
        embeddings = np.zeros((len(data),
                               model.config.hidden_size,
                               model.config.num_hidden_layers + 1))
        for i, sentence in enumerate(data):
            embeddings[i] = self.get_emb(sentence, model)
        return embeddings

    def save_embeddings(self, test_name, embeddings, checkpoint):
        """
        Saves embeddings to a pickle file
        :param embeddings: a matrix of embeddings
        :param checkpoint: a checkpoint
        :return: a pickle file
        """
        with open(os.path.join(self.output_path, f'BERT_checkpoints_{test_name}_{int(checkpoint)+self.delay}.pickle'), 'wb') as f:
            pickle.dump(embeddings, f, protocol=4)

    def calculate(self, task, X_train, X_test):
        for checkpoint in tqdm(self.checkpoints):
            num_checkpoint = checkpoint.split('-')[-1]
            model = self.load_model(checkpoint)
            embs = self.calculate_embeddings(model, X_train)
            self.save_embeddings(f"{task}_TRAIN", embs, num_checkpoint)
            embs = self.calculate_embeddings(model, X_test)
            self.save_embeddings(f"{task}_TEST", embs, num_checkpoint)


class DiscourseEmbeddings(Embeddings):
    def __init__(self, device, tokenizer_path,
                 output_path, delay=0):
        super().__init__(device, tokenizer_path,
                         output_path, delay)

    def encode(self, sentences):
        batches = []
        for text in sentences:
            tokenized_text = self.tokenizer.batch_encode_plus(text,
                                                  max_length=512,
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=True)
            batches.append(tokenized_text)
        return batches

    def mean_pooling(self, model_output, attention_mask, emb_number, size, num_layers):
        final_tokens = np.zeros((model_output[0].shape[0], size, num_layers))
        tokens = model_output[0].cpu().detach().numpy()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(tokens.shape).float().cpu().detach().numpy()
        normalized = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        for num, i in enumerate(model_output):
            tokens = i.cpu().detach().numpy() # batch_size x seq_len x emb_size
            final_tokens[:,:,num] = np.sum(tokens * input_mask_expanded, axis=1)/normalized
        return final_tokens

    def get_discourse_emb(self, batch, model, emb_number):
        """
        Encodes a sentence and returns an embedding
        :param batch: a batch
        :param model: a transformer model
        :return:
        """
        input_ids = torch.tensor(batch['input_ids'], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(batch['attention_mask'], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              return_dict=True
            )
        emb = output.hidden_states
        mean_pool = self.mean_pooling(emb, attention_mask,
                                      emb_number, model.config.hidden_size,
                                      model.config.num_hidden_layers + 1)
        embedding = np.concatenate(mean_pool, axis=0)
        return embedding

    def calculate_discourse_embeddings(self, model, data):
        """
        Calculates embeddings for all sentences
        :param model: a path to a checkpoint
        :param data: a corpus of texts
        :return: a matrix of embeddings
        """
        batches = self.encode(data)
        emb_number = len(batches[0]['input_ids'])
        embeddings = np.zeros((len(data),
                               model.config.hidden_size * emb_number,
                               model.config.num_hidden_layers + 1))
        for i, batch in enumerate(batches):
            embeddings[i] = self.get_discourse_emb(batch, model, emb_number)
        return embeddings

    def calculate_discourse(self, task, X_train, X_test):
        for checkpoint in tqdm(self.checkpoints):
            num_checkpoint = checkpoint.split('-')[-1]
            model = self.load_model(checkpoint)
            embs = self.calculate_discourse_embeddings(model, X_train)
            self.save_embeddings(f"{task}_TRAIN", embs, num_checkpoint)
            embs = self.calculate_discourse_embeddings(model, X_test)
            self.save_embeddings(f"{task}_TEST", embs, num_checkpoint)


class BLiMPEmbeddings(Embeddings):
    def __init__(self, device, tokenizer_path,
                 output_path, delay=0):
        super().__init__(device, tokenizer_path,
                         output_path, delay)
        self.preprocess = WordPunctTokenizer()

    def load_blimp_file(self, dataset):
        with open(f'{dataset}.jsonl') as file:
            tasks = list(file)
        text = []
        for i in tasks:
            string = json.loads(i)
            text.append([string['sentence_bad'], string['sentence_good']])
        dataframe = pd.DataFrame(text, columns=['sentence_bad', 'sentence_good'])
        return dataframe

    def mask_sentences(self, data):
        masked_sentences = []
        sentences = [a for pair in data.values for a in pair]
        for sentence in sentences:
            masks = []
            s = self.preprocess.tokenize(sentence)
            for i, word in enumerate(s):
                masked = self.preprocess.tokenize(sentence)
                masked[i] = '[MASK]'
                masks.append([' '.join(masked), word])
            masked_sentences.append(masks)
        return masked_sentences

    def calculate_probs(self, masked_sentences, unmasker):
        probs = []
        for s in masked_sentences:
            prob = 0
            for sent, word in s:
                for a in unmasker(sent):
                    if a['token_str'] == word.lower():
                        prob += a['score']
            probs.append(prob / len(sent))
        return probs

    def calculate_accuracy(self, probs):
        accuracy = 0
        for i in range(0, len(probs), 2):
            if probs[i] < probs[i + 1]:
                accuracy += 1
        accuracy = accuracy * 2 / len(probs)
        return accuracy

    def save_probs(self, test_name, probs, checkpoint):
        with open(os.path.join(self.output_path, f'probes_{test_name}_{int(checkpoint) + self.delay}.txt'), 'w', encoding='utf-8') as file:
            for i in range(0, len(probs), 2):
                prob_str = str(probs[i]) + '\t' + str(probs[i + 1])
                file.write(prob_str)

    def save_metrics(self, metrics, task):
        metrics_csv = pd.DataFrame(metrics, columns=["Checkpoint", "Accuracy"])
        metrics_csv.to_csv(os.path.join(self.output_path, f"{task}_metrics.csv"))

    def get_probabilities(self, task, dataframe):
        masked_sentences = self.mask_sentences(dataframe)
        metrics = []
        for checkpoint in tqdm(self.checkpoints):
            num_checkpoint = checkpoint.split('-')[-1]
            unmasker = pipeline('fill-mask', tokenizer=self.tokenizer,
                                model=checkpoint, device=0)
            probs = self.calculate_probs(masked_sentences, unmasker)
            self.save_probs(task, probs, num_checkpoint)
            accuracy = self.calculate_accuracy(probs)
            metrics.append([num_checkpoint, accuracy])
        self.save_metrics(metrics, task)
        return metrics


class Prober(DiscourseEmbeddings, BLiMPEmbeddings):
    def __init__(self, dir_path, tokenizer_path, output_path,
                 delay=0, device="cuda:0"):
        self.discourse = ["conn", "PDTB", "DC", "SP", ]
        self.morphosyntax = ["subj_number", "top_constituents",
                                "tree_depth", "person", ]
        self.blimp = ["adjunct_island", "principle_A_c_command",
                        "passive_1", "transitive"]
        self.checkpoints = self.get_checkpoints(dir_path)
        super().__init__(device=device, delay=delay, 
                         output_path=output_path, 
                         tokenizer_path=tokenizer_path)

    def run_probe(self):
        for task in self.morphosyntax:
            print(f"Calculating {task} task...")
            X_train, y_train, X_test, y_test = self.load_files(task)
            self.calculate(task, X_train, X_test)
        for task in self.discourse:
            print(f"Calculating {task} task...")
            X_train, y_train, X_test, y_test = self.load_files(task)
            self.calculate_discourse(task, X_train, X_test)
        for task in self.blimp:
            print(f"Calculating {task} task...")
            dataframe = self.load_blimp_file(task)
            self.get_probabilities(task, dataframe)
