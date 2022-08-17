import os
import re
import pickle

import pandas as pd
from typing import List, Tuple
from knowledge_handler.prolog import PrologDA
from knowledge_handler.kb import MetaQAKB
import random

import argparse

class MetaQADataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.prolog_da = PrologDA()

        kb_path = os.path.join(base_path, 'kb.txt')
        self.kb = MetaQAKB(kb_path)

        self.prolog_da.add_kb_entities_and_relations(self.kb, add_reverse_rel=True)

        self.hop_step_to_predicate_dict = {
            'actor_movie': 'starred_actors_reverse',
            'director_movie': 'directed_by_reverse',
            'movie_actor': 'starred_actors',
            'movie_director': 'directed_by',
            'movie_genre': 'has_genre',
            'movie_imdbrating': 'has_imdb_rating',
            'movie_imdbvotes': 'has_imdb_votes',
            'movie_language': 'in_language',
            'movie_tags': 'has_tags',
            'movie_writer': 'written_by',
            'movie_year': 'release_year',
            'tag_movie': 'has_tags_reverse',
            'writer_movie': 'written_by_reverse'
        }
        
        self.raw_train, self.raw_test, self.raw_dev = self.load_raw_data(base_path)
        random.shuffle(self.raw_train)

    def save_jsonl(self, base_path):
        questions, logical_steps = zip(*self.raw_train)
        train_df = pd.DataFrame({'question': questions, 'logical_steps': logical_steps})
        train_df.to_json(os.path.join(base_path, 'train.json'), orient='records', lines=True)

        questions, logical_steps = zip(*self.raw_dev)
        dev_df = pd.DataFrame({'question': questions, 'logical_steps': logical_steps})
        dev_df.to_json(os.path.join(base_path, 'dev.json'), orient='records', lines=True)

        questions, logical_steps = zip(*self.raw_test)
        test_df = pd.DataFrame({'question': questions, 'logical_steps': logical_steps})
        test_df.to_json(os.path.join(base_path, 'test.json'), orient='records', lines=True)

    def save_vocabs(self, base_path):
        entity_vocab = self.prolog_da.ent_vocab
        relation_vocab = self.prolog_da.rel_vocab

        with open(os.path.join(base_path, 'entity_vocab.pkl'), 'wb') as f:
            pickle.dump(entity_vocab, f)

        with open(os.path.join(base_path, 'relation_vocab.pkl'), 'wb') as f:
            pickle.dump(relation_vocab, f)


    @staticmethod
    def logics_str_to_steps(logics_str):
        logics_str = logics_str.replace("_to_", "_")
        logics_str += "_"
        steps = []
        logic_steps = []
        i = 0
        prev_und_loc = -1
        while i < len(logics_str):
            if logics_str[i] == '_' or i == len(logics_str) - 1:
                steps.append(logics_str[prev_und_loc + 1:i])
                prev_und_loc = i
            i += 1
        for i in range(len(steps) - 1):
            logic_steps.append(f"{steps[i]}_{steps[i + 1]}")
        return logic_steps

    def create_query_string(self, question_str, question_steps, logic_to_predicate_dict, entity_vocab, relation_vocab):
        # extract question concept
        question_concept = re.findall(r'\[(.+)\]', question_str)[0]
        # print(question_concept)

        # define prolog variables
        prolog_vars = ['X', 'Y', 'Z']
        logic_steps = self.logics_str_to_steps(question_steps)
        predicate_steps = [logic_to_predicate_dict[x] for x in logic_steps]

        # create predicates
        p1 = f'{predicate_steps[0]}({question_concept},X)'
        p1_query = f'{relation_vocab[predicate_steps[0]]}({entity_vocab[question_concept]},X)'
        i = 1
        predicates = [p1, ]
        predicates_query = [p1_query, ]
        while i < len(predicate_steps):
            predicates.append(f"{predicate_steps[i]}({prolog_vars[i - 1]},{prolog_vars[i]})")
            predicates_query.append(
                f'{relation_vocab[predicate_steps[i]]}({prolog_vars[i - 1]},{prolog_vars[i]})'
            )
            i += 1

        query_string = ', '.join(predicates_query)
        return predicates, query_string

    def load_raw_data(self, base_path) -> Tuple[List, List, List]:
        multi_hop_paths = ['1hop', '2hop', '3hop']
        train_raw_data = []
        test_raw_data = []
        dev_raw_data = []

        for multi_hop_path in multi_hop_paths:
            hop_path = os.path.join(base_path, multi_hop_path)
            for split, ds in zip(['train', 'test', 'dev'], [train_raw_data, test_raw_data, dev_raw_data]):

                questions_path = os.path.join(hop_path, f'qa_{split}.txt')
                logical_steps_path = os.path.join(hop_path, f'qa_{split}_qtype.txt')
                questions = []
                logical_steps = []

                with open(questions_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                    for line in lines:
                        q, a = line.split('\t')
                        questions.append(q)

                with open(logical_steps_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                    assert len(lines) == len(questions)

                    for logic_step, question in zip(lines, questions):
                        predicates, _ = self.create_query_string(
                            question,
                            logic_step,
                            self.hop_step_to_predicate_dict,
                            self.prolog_da.ent_vocab,
                            self.prolog_da.rel_vocab
                        )
                        logical_steps.append(", ".join(predicates))
                ds.extend(list(zip(questions, logical_steps)))
        
        print(f'train set size:{len(train_raw_data)}, test set size: {len(test_raw_data)}, '
              f'dev set size: {len(dev_raw_data)}')

        return train_raw_data, test_raw_data, dev_raw_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='metaqa')

    args = parser.parse_args()

    if args.dataset == 'metaqa':
        loader = MetaQADataLoader(args.data_path)
    else:
        raise NotImplementedError()

    loader.save_jsonl(args.data_path)
    loader.save_vocabs(args.data_path)
