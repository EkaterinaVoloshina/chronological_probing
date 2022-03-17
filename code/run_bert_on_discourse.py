from utils import load_files
from tqdm.notebook import tqdm
from bert_discourse_embeddings import Embeddings


def get_checkpoints(num):
    checkpoints = num
    path = []
    for i in range(checkpoints):
        num = i * 100000
        link = f'../../small_wiki_bs_128/model_{num}.pth'  # path to a model
        path.append(link)
    return checkpoints, path
        
def main():
    task_name = 'Conn'  # change the filename
    X_train, y_train, X_test, y_test = load_files(task_name)
    checkpoints, path = get_checkpoints(20)
    embeddings = Embeddings(path, X_train, checkpoints, 768, f'{task_name}_TRAIN', 'conn')
    metrics = embeddings.calculate()
    embeddings = Embeddings(path, X_test, checkpoints, 768, f'{task_name}_TEST', 'conn')
    embeddings.calculate()

if __name__ == "__main__":
    main()
