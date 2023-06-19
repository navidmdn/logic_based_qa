[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/answering-questions-over-knowledge-graphs/question-answering-on-metaqa)](https://paperswithcode.com/sota/question-answering-on-metaqa?p=answering-questions-over-knowledge-graphs)

# logic_based_qa

Implementation of the paper [Answering Questions Over Knowledge Graphs Using Logic Programming Along with Language Models](https://arxiv.org/abs/2303.02206).
In this repo I have implemented the code for translating MetaQA 
questions into logical predicates and then using Prolog to build a 
knowledge base over the MetaQA knowledge graph and answer the questions.

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

```
!PYTHONPATH=. python nl2log/data_loader.py --data_path=./data --dataset=metaqa --sample_
bash translation_trainer.sh
```

### evaluate the trained seq2seq model checkpoint
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
starred_actors(Creepshow,X), starred_actors_reverse(X,Y), in_language(Y,Z), not(Y==Creepshow)

Answer:
['English', 'Polish']

```
