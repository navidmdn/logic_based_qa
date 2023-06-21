[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/answering-questions-over-knowledge-graphs/question-answering-on-metaqa)](https://paperswithcode.com/sota/question-answering-on-metaqa?p=answering-questions-over-knowledge-graphs)

# logic_based_qa

Implementation of the paper [Answering Questions Over Knowledge Graphs Using Logic Programming Along with Language Models](https://arxiv.org/abs/2303.02206).
In this repo I have implemented the code for translating MetaQA 
questions into logical predicates and then using Prolog to build a 
knowledge base over the MetaQA knowledge graph and answer the questions.

## Try the whole pipeline in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://raw.githubusercontent.com/navidmdn/logic_based_qa/main/notebooks/run.ipynb)


## Installing requirements

This project has been tested on ubuntu 20.04 and macos 10.14 operating systems, with
python 3.10.

### python requirements

First [install pytorch](https://pytorch.org/get-started/locally/) based on your system's requirements. Then install requirements using this command:
```
pip install -r requirements.txt
```

### prolog

For installation guide check [pyswip github page](https://github.com/yuce/pyswip). A sample installation 
guide can be found inside the colab notebook.


## Training the language to predicate model

run the commands below inside the project base directory

### preparing dataset and training model

create a random sample of 1000 training examples with equal number of samples from each hops:
```
!PYTHONPATH=. python nl2log/data_loader.py --data_path=./data --dataset=metaqa --sample_size 1000
```

Previous command creates a dataset called train_1000.json. Run the trainer on this sample:
```
bash translation_trainer.sh 1000
```

### evaluate the trained seq2seq model checkpoint

Evaluates the translation accuracy on all test samples:
 
```
!PYTHONPATH=. python nl2log/evaluation.py --model_cp=models/t5-small/checkpoint-x
```

## run the full question answering pipeline

### translate questions to predicates
```
!PYTHONPATH=. python qa/evaluation.py --model_path="./models/t5-small/checkpoint-x" --generate_predicates
```

### evaluate the question answering module on MetaQA test dataset
```
!PYTHONPATH=. python qa/evaluation.py --model_path="./models/t5-small/checkpoint-x"
```

## Manually test the model

After training the seq2seq model, you can manually test the model by running on
custom questions as follow:

```python

from qa.question_answering import QuestionAnswering
from qa.data_loader import MetaQADataLoader

data_loader = MetaQADataLoader('./data')
qa = QuestionAnswering('./models/t5-small/checkpoint-5000', data_loader)

qa.answer_question(
    "the films that share actors with the film [Creepshow] were in which languages"
)
```

internally, this produces a Prolog query and then fetches the answers:

```Prolog
Query:
starred_actors(Creepshow,X), starred_actors_reverse(X,Y), in_language(Y,Z)

Answer:
['English', 'Polish']

```

## Fixing bugs in the dataset

the data in `data/` file is the fixed version of the original MetaQA dataset.

modifications to the 1-hop questions in test set:
The answer set to the following question in 1hop qa_test modified from 

1- 
```
[Joseph L. Mankiewicz] directed which movies
	
All About Eve|Sleuth|Cleopatra|Guys and Dolls|Suddenly|Last Summer|Julius Caesar|The Barefoot Contessa|A Letter to Three Wives|People Will Talk|No Way Out|5 Fingers|There Was a Crooked Man...|Dragonwyck|House of Strangers|Somewhere in the Night|The Honey Pot|The Quiet American|A Carol for Another Christmas
```
to:

```
All About Eve|Sleuth|Cleopatra|Guys and Dolls|Suddenly, Last Summer|Julius Caesar|The Barefoot Contessa|A Letter to Three Wives|People Will Talk|No Way Out|5 Fingers|There Was a Crooked Man...|Dragonwyck|House of Strangers|Somewhere in the Night|The Honey Pot|The Quiet American|A Carol for Another Christmas
```

Suddenly, Last Summer is one movie but in the test set wrongly stated as two seperate movies.

2-

```
which films can be described by [nastassja kinski]

Paris|Texas|Cat People|Unfaithfully Yours|Maria's Lovers
```
```
what movies can be described by [dean stockwell]	

Paris|Texas|Compulsion
```

changed to:
```
which films can be described by [nastassja kinski]

Paris, Texas|Cat People|Unfaithfully Yours|Maria's Lovers
```
```
what movies can be described by [dean stockwell]	

Paris, Texas|Compulsion
```

because `Paris, Texas` is one movie in the knowledge graph but here wrongly stated
as separate movies.