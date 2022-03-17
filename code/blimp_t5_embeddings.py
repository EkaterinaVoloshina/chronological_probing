from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from tqdm import tqdm
import pickle
import numpy as np
from string import punctuation
from nltk.tokenize import WordPunctTokenizer

device = 'cuda:0'
preprocess = WordPunctTokenizer()
tokenizer = T5Tokenizer.from_pretrained("t5-small")


class Embeddings(object):
    def __init__(self, path, dataframe, checkpoints, emb_name, delay=0):
        self.path = path
        self.dataframe = dataframe
        self.checkpoints = checkpoints
        self.emb_name = emb_name
        self.delay = delay
        
    def load_model(self, checkpoint):
        m = torch.load(checkpoint)
        model = T5ForConditionalGeneration(T5Config.from_pretrained('t5-small'))
        model.load_state_dict(m['model_state_dict'])
        model.to(device)
        return model
        
    
    def mask_sentences(self):
        masked_sentences = []
        sentences = [a for pair in self.dataframe.values for a in pair]
        for sentence in sentences:
            masks = []
            s = preprocess.tokenize(sentence)
            for i, word in enumerate(s):
                masked = preprocess.tokenize(sentence)
                masked[i] = '<extra_id_0>'
                masks.append([' '.join(masked), word])
            masked_sentences.append(masks)
        return masked_sentences
    
    def filter(self, output, scores, end_token='<extra_id_1>'):
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)

        _txt = tokenizer.decode(output[2:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results = 0
        for i in output:
            results = scores[0][i].detach().cpu().numpy()
        if end_token in _txt:
            _end_token_index = _txt.index(end_token)
            return _txt[:_end_token_index], results
        else:
            return _txt, results
    
    def encode_decode_masked_sentence(self, sent, model):
        encoded = tokenizer.encode_plus(sent[0], add_special_tokens=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)   
        outputs = model.generate(input_ids=input_ids, 
                          max_new_tokens=5,
                          return_dict_in_generate=True, 
                          output_scores=True)

        results = list(map(self.filter, outputs['sequences'], outputs['scores']))
        return results
    
    def calculate_probs(self, masked_sentences, model):
        probs = []
        for s in masked_sentences:
            prob = 0
            for sent, word in s: 
                results = self.encode_decode_masked_sentence(sent, model)
                for result in results:
                    if result[0] == word:
                        prob += int(result[1])
            probs.append(prob)
        return probs

    def calculate_accuracy(self, probs):
        accuracy = 0
        for i in range(0, len(probs), 2):
            if probs[i] < probs[i + 1]:
                  accuracy += 1
        accuracy = accuracy*2/len(probs)
        return accuracy
    
    def save_probs(self, probs, checkpoint):
        with open(f'probes_T5_{self.emb_name}_{checkpoint+self.delay}.txt', 'w', encoding='utf-8') as file:
            for i in range(0, len(probs), 2):
                prob_str = str(probs[i]) + '\t' + str(probs[i+1]) + '\n'
                file.write(prob_str)

    def calculate(self):
        masked_sentences = self.mask_sentences()
        metrics = []
        for checkpoint in tqdm(range(self.checkpoints)):
            path = self.path[checkpoint]
            model = self.load_model(path)
            print('Model is loaded')
            probs = self.calculate_probs(masked_sentences, model)
            print('Probabilities are calculated')
            self.save_probs(probs, checkpoint)
            accuracy = self.calculate_accuracy(probs)
            metrics.append(accuracy)
        return metrics
