from tqdm import tqdm
from typing import Set, List


class KBHandler:
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.entities, self.relations, self.triplets = self.load_kb()

    def load_kb(self) -> (Set[str], Set[str], List[List[str]]):
        raise NotImplementedError()


class MetaQAKBHandler(KBHandler):
    def __init__(self, kb_path: str):
        super().__init__(kb_path)

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
                triplets = line.split('|')
                triplets.append(triplets)
                entities.add(triplets[0])
                entities.add(triplets[2])
                relations.add(triplets[1])

        print(f"loaded {len(triplets)} triplets with {len(entities)} entities and {len(relations)} relations")
        return entities, relations, triplets
