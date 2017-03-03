import time
import pandas as pd
from sklearn.grid_search import GridSearchCV

class GridSearchCVSave(GridSearchCV):
    def fit_and_save(X, y, filename=None):
        if not filename:
            filename = str(self.estimator).partition('(')[0] + time.asctime()
        
        gsearch_res = self.fit(X, y)
        self._save_results(gsearch_res, filename)
        return gsearch_res
