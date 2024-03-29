# Is language acquisition similar in language models and humans? A chronological probing study

### Language transformer-based models for language acquisition

This work addresses the problem of language acquisition in state-of-the-art models and answers several research questions: 
- when do models acquire the linguistic information? 
- which factors influence the language acquisition process?
- how different grammar categories are acquired in comparison with one another?

As our results show, the linguistic information is acquired pretty early. The language data seem to be learnt during the first 600,000 steps. Whereas morphology and syntax tasks show the similar acquisition patterns, discourse is significantly different, as results on discourse tend to be lower and might not be learnt at all.

We also experimented with the Minimum Length Description method to check whether it would show any difference with results of logistic regression. We find out that the 'discourse'-based task is barely solved, while the model shows good quality on 'morphological' and 'syntactic' tasks.


Regarding architectures of models, both T5 and MultiBERT demonstrate comparable results considering the quality of the language level acquisition. To display correlation between language acquisition and different model parameters, we trained four models: one with the minimal hidden size and minimal number of layers and attention heads and three models with one parameter increased and others frozen. These experiments reveal that hidden size appears to be the most essential parameter for language acquisition, whereas attention heads do not significantly increase a model's performance. 

Finally, we compared all tasks with several criteria: selectivity (the difference between pre-trained and randomly initialised models), the number of iterations needed to reach the level of fully trained model, and the size of a model that shows the quality comparable with the base model used before. The idea behind this comparison is to find any correlation between different language levels and probing measures. As a result, models distinguish discourse from morphology and syntax but there is almost no difference between 'morphological' and 'syntactic' tasks.

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

### Reference


```
@article{voloshina2022neural,
  title={Is neural language acquisition similar to natural? A chronological probing study},
  author={Voloshina, Ekaterina and Serikov, Oleg and Shavrina, Tatiana},
  journal={arXiv preprint arXiv:2207.00560},
  year={2022}
}
```


