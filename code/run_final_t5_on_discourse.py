from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from tqdm import tqdm
import pickle
import numpy as np
import copy
from utils import load_model

device = 'cuda:0'

class Embeddings(object):
    def __init__(self, sentences, size, emb_name, task_name):
        self.sentences = sentences
        self.size = size
        self.emb_name = emb_name
        self.task_name = task_name

    def encode(self, tokenizer):
        batches = []
        for text in self.sentences:
              tokenized_text = tokenizer.batch_encode_plus(text,
                                                  #max_length=512,
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=True)
              batches.append(tokenized_text)
        return batches

    def mean_pooling(self, model_output, attention_mask, emb_number):
        final_tokens = np.zeros((model_output[0].shape[0], self.size, 7))
        tokens = model_output[0].cpu().detach().numpy()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(tokens.shape).float().cpu().detach().numpy() 
        normalized = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        for num, i in enumerate(model_output):
            tokens = i.cpu().detach().numpy() # batch_size x seq_len x emb_size
            final_tokens[:,:,num] = np.sum(tokens * input_mask_expanded, axis=1)/normalized
        return final_tokens
    
    def so_concat(self, emb):
        x1 = emb[0]
        x2 = x1 - emb[1]
        x3 = x1 - emb[2]
        x4 = x1 - emb[3]
        x5 = x1 - emb[4]
        return np.concatenate([x1, x2, x3, x4, x5])

    def get_emb(self, batch, model, emb_number):
        """
        Encodes a sentence and returns an embedding
        :param batch: a batch
        :param model: a transformer model
        :return:
        """
        input_ids = torch.tensor(batch['input_ids'], dtype=torch.long).to(device)
        attention_mask = torch.tensor(batch['attention_mask'], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(
              input_ids=input_ids,
              decoder_input_ids=input_ids,
              attention_mask=attention_mask,
              return_dict=True
            )
        emb = output.decoder_hidden_states
        mean_pool = self.mean_pooling(emb, attention_mask, emb_number)
        if self.task_name == 'so':
            embedding = self.so_concat(mean_pool)
        else:
            embedding = np.concatenate(mean_pool, axis=0) 
        return embedding

    def calculate_embeddings(self, tokenizer):
        """
        Calculates embeddings for all sentences
        :param path: a path to a checkpoint
        :param sentences: a corpus of texts
        :return: a matrix of embeddings
        """
        labels = []
        model = T5ForConditionalGeneration.from_pretrained('t5-small', output_hidden_states=True)
        model.to(device)
        print('Model is loaded')
        batches = self.encode(tokenizer)
        emb_number = len(self.sentences[0])
        embeddings = np.zeros((len(self.sentences), self.size * emb_number, 7))
        for i, batch in enumerate(batches):
            embeddings[i] = self.get_emb(batch, model, emb_number)
        print('Embeddings are calculated')
        return embeddings

    def save_embeddings(self, embeddings):
        """
        Saves embeddings to a pickle file
        :param embeddings: a matrix of embeddings
        :param checkpoint: a checkpoint
        :return: a pickle file
        """
        with open(f'embeddings/T5_checkpoints_{self.emb_name}_final.pickle', 'wb') as f:
            pickle.dump(embeddings, f)

    def calculate(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        embs = self.calculate_embeddings(tokenizer)
        self.save_embeddings(embs)
 
def run_on_task(task_name):
    X_train, y_train, X_test, y_test = load_files(task_name)
    embeddings = Embeddings(X_train, 512, f'{task_name}_TRAIN', task_name.lower())
    embeddings.calculate()
    embeddings = Embeddings(X_test, 512, f'{task_name}_TEST', task_name.lower())
    embeddings.calculate()
    
def main():
    task_name = "person"
    run_on_task(task_name)
    run_logreg(task_name)
        
if __name__ == "__main__":
    main()
