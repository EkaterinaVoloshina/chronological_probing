from utils import load_files
from tqdm.notebook import tqdm
from bert_blimp_embeddings import Embeddings

def get_checkpoints(number):
    checkpoints = number
    path = []
    for i in range(checkpoints):
        if i > 10:
            num += 100
        else:
            num = i * 20
        link = f'MultiBertGunjanPatrick/multiberts-seed-0-{num}k'
        path.append(link)
    return checkpoints, path
        
def main():
    task_name = 'adjunct_island'  # change the filename
    dataframe = load_files(task_name)
    checkpoints, path = get_checkpoints(20)
    embeddings = Embeddings(path, dataframe, checkpoints, task_name)
    metrics = embeddings.calculate()

if __name__ == "__main__":
    main()
