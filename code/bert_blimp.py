from transformers import BertTokenizer, BertModel, pipeline
import torch
from tqdm import tqdm
import pickle
import numpy as np
from string import punctuation
from nltk.tokenize import WordPunctTokenizer

device = 'cuda:0'
preprocess = WordPunctTokenizer()
tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")


class Embeddings(object):
    def __init__(self, path, dataframe, checkpoints, emb_name, delay=0):
        self.path = path
        self.dataframe = dataframe
        self.checkpoints = checkpoints
        self.emb_name = emb_name
        self.delay = delay
    
    def mask_sentences(self):
        masked_sentences = []
        sentences = [a for pair in self.dataframe.values for a in pair]
        for sentence in sentences:
            masks = []
            s = preprocess.tokenize(sentence)
            for i, word in enumerate(s):
                masked = preprocess.tokenize(sentence)
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
            probs.append(prob/len(sent))
        return probs

    def calculate_accuracy(self, probs):
        accuracy = 0
        for i in range(0, len(probs), 2):
            if probs[i] < probs[i + 1]:
                  accuracy += 1
        accuracy = accuracy*2/len(probs)
        return accuracy
    
    def save_probs(self, probs, checkpoint):
        with open(f'probes_{self.emb_name}_{checkpoint+self.delay}.txt', 'w', encoding='utf-8') as file:
            for i in range(0, len(probs), 2):
                prob_str = str(probs[i]) + '\t' + str(probs[i+1])
                file.write(prob_str)

    def calculate(self):
        masked_sentences = self.mask_sentences()
        metrics = []
        for checkpoint in tqdm(range(self.checkpoints)):
            path = self.path[checkpoint]
            unmasker = pipeline('fill-mask', tokenizer=tokenizer, model=path, device=0)
            print('Model is loaded')
            probs = self.calculate_probs(masked_sentences, unmasker)
            print('Probabilities are calculated')
            self.save_probs(probs, checkpoint)
            accuracy = self.calculate_accuracy(probs)
            metrics.append(accuracy)
        return metrics