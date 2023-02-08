from tqdm import tqdm
from typing import Set, List
import re


class KB:
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.entities, self.relations, self.triplets = self.load_kb()

    def load_kb(self) -> (Set[str], Set[str], List[List[str]]):
        raise NotImplementedError()


class MetaQAKB(KB):
    SPECIAL_CHAR = 'SPC'

    def __init__(self, kb_path: str, add_reverse_rel: bool = True):
        self.add_reverse_rel = add_reverse_rel
        # self.regex = re.compile("[^a-zA-Z0-9\\s*.!?',_\\-]")
        super().__init__(kb_path)

    # def normalize_chars(self, strl: List[str]) -> List[str]:
    #     return [self.regex.sub(self.SPECIAL_CHAR, x) for x in strl]

    def load_kb(self) -> (Set, Set, List):
        """
        Loads the knowledge base from the given path
        :return: set of entities, set of relations, list of triplets
        """
        entities = set()
        relations = set()
        triplets = []

        with open(self.kb_path, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in tqdm(lines):
                triplet = line.split('|')
                # e1, e2 = self.normalize_chars([triplet[0], triplet[2]])
                e1, e2 = triplet[0], triplet[2]
                r = triplet[1]

                triplets.append([e1, r, e2])
                if self.add_reverse_rel:
                    rel = r + '_reverse'
                    triplets.append([e2, rel, e1])
                    relations.add(rel)

                entities.add(e1)
                entities.add(e2)
                relations.add(r)

        print(f"loaded {len(triplets)} triplets with {len(entities)} entities and {len(relations)} relations")
        return entities, relations, triplets
