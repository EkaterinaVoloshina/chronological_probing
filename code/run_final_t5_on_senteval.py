from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from tqdm import tqdm
import pickle
import numpy as np

device = 'cuda:0'

class Embeddings(object):
    def __init__(self, sentences, size, test_name):
        self.sentences = sentences
        self.size = size
        self.test_name = test_name

    def load_model(self):
        """
        Loads a transformer model
        :return: a model and a tokenizer
        """
        model = T5ForConditionalGeneration.from_pretrained('t5-small', output_hidden_states=True)
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
            enc = tokenizer(sent, padding=True, truncation=True, return_tensors='pt')
            enc.to(device)
            output = model(
              input_ids=enc['input_ids'],
              decoder_input_ids=enc['input_ids'],
              return_dict=True
            )
        emb = output.decoder_hidden_states
        mean_pool = np.zeros((self.size, 7))
        for num, e in enumerate(emb):
            mean_pool[:,num] = torch.mean(e, 1).squeeze(0).cpu().numpy()
        return mean_pool


    def calculate_embeddings(self, tokenizer):
        """
        Calculates embeddings for all sentences
        :param path: a path to a checkpoint
        :param sentences: a corpus of texts
        :return: a matrix of embeddings
        """
        model = self.load_model()
        print('Model is loaded')
        embeddings = np.zeros((len(self.sentences), self.size, 7))
        for i, sentence in enumerate(self.sentences):
            embeddings[i] = self.get_emb(sentence, model, tokenizer)
        print('Embeddings are calculated')
        return embeddings

    def save_embeddings(self, embeddings):
        """
        Saves embeddings to a pickle file
        :param embeddings: a matrix of embeddings
        :param checkpoint: a checkpoint
        :return: a pickle file
        """
        with open(f'embeddings/T5_checkpoints_{self.test_name}_final.pickle', 'wb') as f:
            pickle.dump(embeddings, f)

    def calculate(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        embs = self.calculate_embeddings(tokenizer)
        self.save_embeddings(embs)
        
        
def run_on_task(task_name):
    X_train, y_train, X_test, y_test = load_files(task_name)
    embeddings = Embeddings(X_train, self.size, f'{task_name}_TRAIN')
    embeddings.calculate()
    embeddings = Embeddings(X_test, self.size, f'{task_name}_TEST')
    embeddings.calculate()
    
def run_logreg(task_name):
    X_train, y_train, X_test, y_test = load_files(task_name)
    embeddings = [[f'./embeddings/T5_checkpoints_{task_name}_TRAIN_final.pickle',
                  f'./embeddings/T5_checkpoints_{task_name}_TEST_final.pickle']]
    logreg = LogRegClassification(y_train, y_test, embeddings, task_name, 'final_T5')
    predictions, scores = logreg.probe()
    return predictions, scores


def main():
    task_name = "person"
    run_on_task(task_name)
    run_logreg(task_name)
        
        
if __name__ == "__main__":
    main()
