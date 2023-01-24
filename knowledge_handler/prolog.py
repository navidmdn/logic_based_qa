from pyswip import Prolog
from tqdm import tqdm
from knowledge_handler.kb import KB


class PrologDA:
    def __init__(self):
        self.prolog = Prolog()
        self.ent_vocab = {}
        self.rel_vocab = {}

    def add_kb_entities_and_relations(self, kb: KB):
        # register direct and reverse relations as predicates

        for triple in tqdm(kb.triplets):
            entities = [triple[0], triple[2]]
            relations = [triple[1]]

            for entity in entities:
                if entity not in self.ent_vocab:
                    self.ent_vocab[entity] = f"ENT_{len(self.ent_vocab)}"
            for relation in relations:
                if relation not in self.rel_vocab:
                    self.rel_vocab[relation] = f"REL_{len(self.rel_vocab)}"

    def register_kb(self, kb: KB):
        self.add_kb_entities_and_relations(kb)
        for triple in tqdm(kb.triplets):
            e1, e2 = triple[0], triple[2]
            e1_idx, e2_idx = self.ent_vocab[e1], self.ent_vocab[e2]
            rel_idx = self.rel_vocab[triple[1]]
            self.prolog.assertz(f"{rel_idx}({e1_idx}, {e2_idx})")