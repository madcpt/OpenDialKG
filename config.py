

class Config(object):
    def __init__(self):
        self.dataset = 'opendialkg'
        self.dataset_folder = './opendialkg/data/'
        self.dial_file = self.dataset_folder + 'opendialkg.csv'
        self.entities_file = self.dataset_folder + 'opendialkg_entities.txt'
        self.relations_file = self.dataset_folder + 'opendialkg_relations.txt'
        self.triples_file = self.dataset_folder + 'opendialkg_triples.txt'


cfg = Config()
