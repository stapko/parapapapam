import time
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV


__all__ = ['GridSearchCVSave', 'GranularGridSearchCVSave']


class GridSearchCVSave(GridSearchCV):
    def fit_and_save(self, X, y, filename=None):
        if not filename:
            filename = str(self.estimator).partition('(')[0] + time.asctime() + '.csv'
	elif filename:
	    if filename.rpartition('.')[-1] != 'csv':
		raise ValueError

        gsearch_res = self.fit(X, y)
        self._save_results(self.grid_scores_, filename)
        return gsearch_res

    @staticmethod
    def _save_results(gsearch_res, filename):

        length = len(gsearch_res)
        cv = len(gsearch_res[0].cv_validation_scores) * np.ones(length)
        mean = np.zeros(length)
        std = np.zeros(length)
        params = np.array([None] * length)

        for i, iteration in enumerate(gsearch_res):
            mean[i] = iteration.mean_validation_score
            std[i] = np.std(iteration.cv_validation_scores)
            params[i] = iteration.parameters

        df_result = pd.DataFrame()
        df_result['mean'] = mean
        df_result['std'] = std
        df_result['cv'] = cv
        df_result['params'] = params
        df_result.to_csv(filename, index=False)


# If change name it also need to change it above in '__all__' list
class GranularGridSearchCVSave(GridSearchCV):
    def fit_and_save(self, X, y, filename=None):
        if not filename:
            filename = str(self.estimator).partition('(')[0] + time.asctime() + '.csv'
	elif filename:
	    if filename.rpartition('.')[-1] != 'csv':
		raise ValueError

	# TODO: granular grid serach
