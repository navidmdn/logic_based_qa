from pyswip import Prolog
from tqdm import tqdm
from knowledge_handler.kb import KB
from rapidfuzz.distance import Levenshtein


class PrologDA:
    def __init__(self):
        self.prolog = Prolog()
        self.ent_vocab = {}
        self.rel_vocab = {}
        self.inv_ent_vocab = {}

    def add_kb_entities_and_relations(self, kb: KB):

        for triple in tqdm(kb.triplets):
            entities = [triple[0], triple[2]]
            relations = [triple[1]]

            for entity in entities:
                if entity not in self.ent_vocab:
                    self.ent_vocab[entity] = f"ent_{len(self.ent_vocab)}"
            for relation in relations:
                if relation not in self.rel_vocab:
                    self.rel_vocab[relation] = f"rel_{len(self.rel_vocab)}"

        self.inv_ent_vocab = {v: k for k, v in self.ent_vocab.items()}

    def register_kb(self, kb: KB):
        self.add_kb_entities_and_relations(kb)
        for triple in tqdm(kb.triplets):
            e1, e2 = triple[0], triple[2]
            e1_idx, e2_idx = self.ent_vocab[e1], self.ent_vocab[e2]
            rel_idx = self.rel_vocab[triple[1]]
            self.prolog.assertz(f"{rel_idx}({e1_idx},{e2_idx})")

    def find_entity_by_ed(self, entity: str):
        for ent in self.ent_vocab:
            d = Levenshtein.distance(entity, ent, score_cutoff=2)
            if d < 3:
                return ent
        return None

    def transform_query_with_vocab(self, query: str, question_entity: str):
        rels_sorted_by_len = sorted(list(self.rel_vocab.keys()), key=lambda x: -len(x))
        for rel in rels_sorted_by_len:
            if rel in query:
                query = query.replace(rel, self.rel_vocab[rel])

        try:
            query = query.replace(question_entity, self.ent_vocab[question_entity])
        except KeyError:
            question_ent_rec = self.find_entity_by_ed(question_entity)
            if question_ent_rec is None:
                raise Exception("entity not found!")

            query = query.replace(question_entity, self.ent_vocab[question_ent_rec])

        return query

    def query(self, query: str, question_entity: str):
        query = self.transform_query_with_vocab(query, question_entity)
        results = list(self.prolog.query(query))

        answers = []
        for res in results:
            for k, v in res.items():
                res[k] = self.inv_ent_vocab[v]
            if len(res) == 1:
                answers.append(res['X'])
            elif len(res) == 2:
                answers.append(res['Y'])
            elif len(res) == 3:
                answers.append(res['Z'])
        return [x for x in list(set(answers))]
