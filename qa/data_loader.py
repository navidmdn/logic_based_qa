import os
import re

import pandas as pd
from typing import Dict
from knowledge_handler.prolog import PrologDA
from knowledge_handler.kb import MetaQAKB
import random

import argparse


class MetaQADataLoader:
    def __init__(self, base_path, split='test'):
        self.base_path = base_path
        self.prolog_da = PrologDA()

        kb_path = os.path.join(base_path, 'kb.txt')
        self.kb = MetaQAKB(kb_path)

        self.prolog_da.register_kb(self.kb)
        self.dataset = self.load_question_answers(base_path, split)

    def load_question_answers(self, base_path, split='test') -> Dict:
        multi_hop_paths = ['1hop', '2hop', '3hop']
        dataset = {}

        for multi_hop_path in multi_hop_paths:
            hop_path = os.path.join(base_path, multi_hop_path)

            questions_path = os.path.join(hop_path, f'qa_{split}.txt')
            questions = []
            answers = []

            with open(questions_path, 'r') as f:
                lines = f.read().strip().split('\n')
                for line in lines:
                    q, a = line.split('\t')
                    question_concept = re.findall(r'\[(.+)\]', q)[0]
                    question_concept_cleaned = self.kb.regex.sub(self.kb.SPECIAL_CHAR, question_concept)
                    q = q.replace(question_concept, question_concept_cleaned)
                    questions.append(q)
                    answers.append(a.split('|'))

            dataset[multi_hop_path] = list(zip(questions, answers))
        return dataset
