from logreg import LogRegClassification
from utils import load_files
import warnings
warnings.filterwarnings("ignore")

def get_embeddings(task, number):
    embeddings = []
    for i in range(number):
        embeddings.append([f'./embeddings/{model}_checkpoints_{task}_TRAIN_{i}.pickle',
                           f'./embeddings/{model}_checkpoints_{task}_TEST_{i}.pickle'])
    return embeddings

def main():
    task = 'DC'
    model = 'T5'
    X_train, y_train, X_test, y_test = load_files(task, model)
    embeddings = get_embeddings(task)
    logreg = LogRegClassification(y_train, y_test, embeddings, 'DC')
    predictions, scores = logreg.probe()

if __name__=='__main__':
    main()
