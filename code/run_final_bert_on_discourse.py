from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import pickle
from utils import load_files
import numpy as np
from logreg import LogRegClassification
import warnings
warnings.filterwarnings("ignore")

device = 'cuda:0'


class Embeddings(object):
    def __init__(self, sentences, size, emb_name, delay=0):
        self.sentences = sentences
        self.size = size
        self.emb_name = emb_name
        self.delay = delay
    
    def load_model(self):
        """
        Loads a transformer model
        :return: a model and a tokenizer
        """
        model = BertModel.from_pretrained('MultiBertGunjanPatrick/multiberts-seed-0', output_hidden_states=True)
        model.to(device)
        return model

    def encode(self, tokenizer):
        batches = []
        for text in self.sentences:
              tokenized_text = tokenizer.batch_encode_plus(text,
                                                  max_length=512,
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  return_attention_mask=True)
              batches.append(tokenized_text)
        return batches

    def mean_pooling(self, model_output, attention_mask, emb_number):
        final_tokens = np.zeros((model_output[0].shape[0], self.size, 13))
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
              attention_mask=attention_mask,
              return_dict=True
            )
        emb = output.hidden_states
        mean_pool = self.mean_pooling(emb, attention_mask, emb_number)
        if self.emb_name.lower().startswith('sp'):
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
        model = self.load_model()
        print('Model is loaded')
        batches = self.encode(tokenizer)
        emb_number = len(batches[0]['input_ids'])
        embeddings = np.zeros((len(self.sentences), self.size * emb_number, 13))
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
        with open(f'embeddings/BERT_checkpoints_{self.emb_name}_final.pickle', 'wb') as f:
            pickle.dump(embeddings, f, protocol=4)

    def calculate(self):
        tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
        embs = self.calculate_embeddings(tokenizer)
        self.save_embeddings(embs)
        
        
 def run_on_task(task_name):
    X_train, y_train, X_test, y_test = load_files(task_name)
    embeddings = Embeddings(X_train, 768, f'{task_name}_TRAIN')
    embeddings.calculate()
    embeddings = Embeddings(X_test, 768, f'{task_name}_TEST')
    embeddings.calculate()
    
    
def run_logreg(task_name):
    X_train, y_train, X_test, y_test = load_files(task_name)
    embeddings = [[f'./embeddings/BERT_checkpoints_{task_name}_TRAIN_final.pickle',
                  f'./embeddings/BERT_checkpoints_{task_name}_TEST_final.pickle']]
    logreg = LogRegClassification(y_train, y_test, embeddings, task_name, 'final_BERT')
    predictions, scores = logreg.probe()
    return predictions, scores
  
  
def main():
    task_name = "SO"
    run_on_task(task_name)
    run_logreg(task_name)
    

if __name__ == "__main__":
    main()
