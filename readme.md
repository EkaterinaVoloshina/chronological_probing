# Is language acquisition similar in language models and humans? A chronological probing study

###  How to use

First, you need to clone this repository: 

```{python}
git clone https://github.com/EkaterinaVoloshina/chronological_probing
```

To calculate embeddings you should use the following code:

```{python}
from prober import Prober

prober = Prober(dir_path=PATH_TO_CHECKPOINTS, 
                tokenizer_path="bert-base-cased", 
                output_path=OUTPUT_PATH, 
                device="cuda:0")
                
prober.run_probe()
```
To run Logistic Regression over embeddings, import Logistic Regression Classifier. To use the baseline mode, change to `random=True`.

```{python}
from logreg import LogRegClassification

logreg = LogRegClassification(dir_path=PATH_TO_EMBEDDINGS, 
                              output_file=OUTPUT_FILE,  
                              random=False)
```


