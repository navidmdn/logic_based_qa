from pyswip import Prolog
from tqdm import tqdm
from knowledge_handler.kb import KB


class PrologDA:
    def __init__(self):
        self.prolog = Prolog()
        self.ent_vocab = {}
        self.rel_vocab = {}

    def add_kb_entities_and_relations(self, kb: KB, add_reverse_rel: bool = True):
        # register direct and reverse relations as predicates

        for triple in tqdm(kb.triplets):
            entities = [triple[0], triple[2]]
            relations = [triple[1]]

            if add_reverse_rel:
                relations.append(triple[1] + '_reverse')

            for entity in entities:
                if entity not in self.ent_vocab:
                    self.ent_vocab[entity] = f"ENT_{len(self.ent_vocab)}"
            for relation in relations:
                if relation not in self.rel_vocab:
                    self.rel_vocab[relation] = f"REL_{len(self.rel_vocab)}"

    def register_kb(self, kb: KB, add_reverse_rel: bool = True):
        self.add_kb_entities_and_relations(kb, add_reverse_rel)
        for triple in tqdm(kb.triplets):
            e1, e2 = triple[0], triple[2]
            e1_idx, e2_idx = self.ent_vocab[e1], self.ent_vocab[e2]
            rel_idx = self.rel_vocab[triple[1]]
            self.prolog.assertz(f"{rel_idx}({e1_idx}, {e2_idx})")
            if add_reverse_rel:
                self.prolog.assertz(f"{rel_idx}_reverse({e2_idx}, {e1_idx})")
