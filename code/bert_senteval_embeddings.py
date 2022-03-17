from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import pickle
import numpy as np

device = 'cuda:0'

class Embeddings(object):
    def __init__(self, path, sentences, checkpoints, size, test_name, delay=0):
        self.path = path
        self.sentences = sentences
        self.checkpoints = checkpoints
        self.size = size
        self.test_name = test_name
        self.delay = delay

    def load_model(self, checkpoint_path):
        """
        Loads a transformer model
        :return: a model and a tokenizer
        """
        #m = torch.load(checkpoint_path)
        #model = T5ForConditionalGeneration(T5Config(output_hidden_states=True))
        #model.load_state_dict(m['model_state_dict'])
        model = BertModel.from_pretrained(checkpoint_path, output_hidden_states=True)
        model.to(device)
        return model


    def get_emb(self, sent, model, tokenizer):
        """
        Encodes a sentence and returns an embedding
        :param sent: a sentence, str
        :param model: a transformer model
        :param tokenizer:
        :return:
        """
        with torch.no_grad():
            enc = tokenizer(sent, padding=True, truncation=True, max_length=512, return_tensors='pt')
            enc.to(device)
            output = model(
              input_ids=enc['input_ids'],
              return_dict=True
            )
        emb = output.hidden_states
        mean_pool = np.zeros((self.size, 13))
        for num, e in enumerate(emb):
            mean_pool[:,num] = torch.mean(e, 1).squeeze(0).cpu().numpy()
        return mean_pool


    def calculate_embeddings(self, path, tokenizer):
        """
        Calculates embeddings for all sentences
        :param path: a path to a checkpoint
        :param sentences: a corpus of texts
        :return: a matrix of embeddings
        """
        model = self.load_model(path)
        print('Model is loaded')
        embeddings = np.zeros((len(self.sentences), self.size, 13))
        for i, sentence in enumerate(self.sentences):
            embeddings[i] = self.get_emb(sentence, model, tokenizer)
        print('Embeddings are calculated')
        return embeddings

    def save_embeddings(self, embeddings, checkpoint):
        """
        Saves embeddings to a pickle file
        :param embeddings: a matrix of embeddings
        :param checkpoint: a checkpoint
        :return: a pickle file
        """
        with open(f'embeddings/BERT_checkpoints_{self.test_name}_{checkpoint+self.delay}.pickle', 'wb') as f:
            pickle.dump(embeddings, f, protocol=4)

    def calculate(self):
        tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
        for checkpoint in tqdm(range(self.checkpoints)):
            path = self.path[checkpoint]
            embs = self.calculate_embeddings(path, tokenizer)
            self.save_embeddings(embs, checkpoint)
