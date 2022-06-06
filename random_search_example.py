import numpy as np
from torch import cartesian_prod
import itertools

class RandomSearch:

    def __init__(self, search_space : dict):

        self.search_space = search_space

        self.cartesian_prod_of_configurations = list(itertools.product(*tuple(search_space.values())))
        self.hp_names =  list(search_space.keys())
        self.valid_configurations = [dict(zip(self.hp_names, x)) for x in self.cartesian_prod_of_configurations]
        self.observed_config = []
        self.pending_config =  np.arange(len(self.valid_configurations))

    
    def observe_and_suggest(self, conf: dict):

        ix = self.cartesian_prod_of_configurations.index(tuple(conf.values()))
        self.pending_config.remove(ix)
        self.observed_config.append(ix)

        next_conf_ix = np.random.choice(self.pending_config, 1)

        return self.valid_configurations[next_conf_ix]



