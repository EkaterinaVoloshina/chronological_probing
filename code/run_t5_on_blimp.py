from utils import load_files
from tqdm.notebook import tqdm
from t5_blimp_embeddings import Embeddings

def get_checkpoints(num):
    checkpoints = num
    path = []
    for i in range(checkpoints):
        num = i * 100000
        link = f'../../small_wiki_bs_128/model_{num}.pth'  # path to a model
        path.append(link)
    return checkpoints, path
        
def main():
    task_name = 'adjunct_island'  # change the filename
    dataframe = load_files(task_name)
    checkpoints, path = get_checkpoints(12)
    embeddings = Embeddings(path, dataframe, checkpoints, task_name)
    metrics = embeddings.calculate()

if __name__ == "__main__":
    main()
