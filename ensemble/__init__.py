from ._ensemble import Blender
from grid_search import GranularGridSearchCVSave
__all__ = (Blender,)


class BaseEnsemble:
    def __init__(self, gran_gsearch=None):
        self.gran_gsearch = gran_gsearch if gran_gsearch else GranularGridSearchCVSave()
