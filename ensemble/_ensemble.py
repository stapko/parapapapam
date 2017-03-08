from . import BaseEnsemble

from grid_search import GranularGridSearchCVSave

class Blender(BaseEnsemble):
    def __init__(self, gran_gsearch=None):
        if gran_gsearch:
            self.gran_gsearch = gran_gsearch
            self.X = gran_gsearch.X
            self.y = gran_gsearch.y
            #self.hash = gran_gsearch.
        else:
            raise ValueError('"gran_gsearch" is None')

    def make_greedy_blend(self, n, models=None):
        pass
