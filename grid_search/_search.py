import time
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from ast import literal_eval
from sklearn.model_selection import cross_val_score

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
class GranularGridSearchCVSave:
    def fit_and_save(self, estimator, X, y, params, scoring, filename=None, verbose=False, cv=5):
        h_e_a_d_e_r = "mean\tstd\tcv\tparams\n"

        params_combinations = self._params_combinations(params)
        header_flag = self._check_header(filename, header=h_e_a_d_e_r)
        if header_flag:
            start_position = self._start_position(filename, params_combinations)
            params_combinations = params_combinations[start_position + 1:]

        if not filename:
            filename = str(estimator).partition('(')[0] + time.asctime().replace(' ', '_').replace(':', '-') + '.txt'
        elif filename:
            if filename.rpartition('.')[-1] != 'txt':
                raise ValueError

        if not header_flag:
            with open(filename, 'a') as the_file:
                the_file.write(h_e_a_d_e_r)
        for params in params_combinations:
            estimator.set_params(**params)
            result = cross_val_score(estimator, X=X, y=y, scoring=scoring, cv=cv, n_jobs=-1)
            mean = np.mean(result)
            std = np.std(result)
            result_string = str(mean) + '\t' + str(std) + '\t' + str(cv) + '\t' + str(
                params) + '\n'
            with open(filename, 'a') as the_file:
                the_file.write(result_string)

            if verbose:
                print('mean:{:2.5f}\tstd:{:2.5f}\tcv:{:3d}\tparams:{}'.format(mean, std, cv, params))

    def _check_header(self, filename, header):
        try:
            with open(filename, 'r') as f:
                first_line = f.readline()
            return first_line == header
        except:
            return False

    def _params_combinations(self, params):

        key_names = params.keys()
        param_values = params.values()

        def my_len(x):
            if type(x) == list:
                return len(x)
            return 1

        number_of_params = len(key_names)
        number_of_values = map(my_len, param_values)

        composition = 1
        for number in number_of_values:
            composition *= number

        all_combinations = map(lambda x: [None] * number_of_params, [None] * composition)

        special_counter = composition
        for position in xrange(number_of_params):
            special_counter /= number_of_values[position]
            for number in xrange(composition):
                param_number = (number / special_counter) % number_of_values[position]
                try:
                    all_combinations[number][position] = param_values[position][param_number]
                except TypeError:
                    all_combinations[number][position] = param_values[position]
        params_dict = map(lambda x: dict(zip(key_names, x)), all_combinations)
        return params_dict

    def _start_position(self, filename, params_combinations):
        data = pd.read_csv(filename, delimiter='\t')
        try:
            last_params = literal_eval(data.params.tolist()[-1])
            return params_combinations.index(last_params)
        except AttributeError, ValueError:
            return 0
