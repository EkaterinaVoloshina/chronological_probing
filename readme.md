# Is language acquisition similar in language models and humans? A chronological probing study


`code`: code for experiments. To calculate embeddings on morphological and syntactic tasks, use the file `run_t5_on_senteval.py`/`run_bert_on_senteval.py`. For discourse tasks use `run_t5_on_discourse.py`/`run_bert_on_discourse.py`. For score calculations use `run_logreg.py`. Baselines can be calculated with the file `run_random_logreg.py`. The scores of final models were counted with the files starting with `run_final`

`scores`: scores of  Logistic regression on T5 and BERT embeddings (`scores - T5,csv` and `scores - BERT.csv` respectively) and embeddings with shuffled labels  (`scores - random T5.csv` and `scores - random BERT.csv` respectively)
