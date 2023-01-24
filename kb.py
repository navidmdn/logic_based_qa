class KB:
    def __init__(self):
        self.triples = []
        self.rels = set()

        with open('data/kb.txt', 'r') as f:
            kb_data = f.read().strip().split('\n')
            for line in kb_data:
                elems = line.split('|')
                self.triples.append(elems)
                self.rels.add(elems[1])

    def get_movies_by_directors(self, directors):
        results = []
        for director in directors:
            for triple in self.triples:
                if triple[2] == director and triple[1] == 'directed_by':
                    results.append(triple[0])
        return results
    
    def get_directors_by_movies(self, movies):
        results = []
        for movie in movies:
            for triple in self.triples:
                if triple[0] == movie and triple[1] == 'directed_by':
                    results.append(triple[2])
        return results
    
    def get_starred_actors_by_movies(self, movies):
        results = []
        for movie in movies:
            for triple in self.triples:
                if triple[0] == movie and triple[1] == 'starred_actors':
                    results.append(triple[2])
        return results
    
    def get_movies_by_starred_actors(self, starred_actors):
        results = []
        for starred_actor in starred_actors:
            for triple in self.triples:
                if triple[2] == starred_actor and triple[1] == 'starred_actors':
                    results.append(triple[0])
        return results
    
    def get_tags_by_movies(self, movies):
        results = []
        for movie in movies:
            for triple in self.triples:
                if triple[0] == movie and triple[1] == 'has_tags':
                    results.append(triple[2])
        return results
    
    def get_movies_by_tags(self, tags):
        results = []
        for tag in tags:
            for triple in self.triples:
                if triple[2] == tag and triple[1] == 'has_tags':
                    results.append(triple[0])
        return results
    
    def get_genres_by_movies(self, movies):
        results = []
        for movie in movies:
            for triple in self.triples:
                if triple[0] == movie and triple[1] == 'has_genre':
                    results.append(triple[2])
        return results
    
    def get_release_year_by_movies(self, movies):
        results = []
        for movie in movies:
            for triple in self.triples:
                if triple[0] == movie and triple[1] == 'release_year':
                    results.append(triple[2])
        return results
    
    def get_movies_by_writers(self, writers):
        results = []
        for writer in writers:
            for triple in self.triples:
                if triple[2] == writer and triple[1] == 'written_by':
                    results.append(triple[0])
        return results
    
    def get_writers_by_movies(self, movies):
        results = []
        for movie in movies:
            for triple in self.triples:
                if triple[0] == movie and triple[1] == 'written_by':
                    results.append(triple[2])
        return results
    
kb = KB()