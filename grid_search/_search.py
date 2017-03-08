import os
import time
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from ast import literal_eval
from sklearn.model_selection import cross_val_score
from hashlib import sha224
from sklearn.base import BaseEstimator
from journal import WorkJournal

__all__ = ['GridSearchCVSave', 'GranularGridSearchCVSave', 'TaskManager']


class TaskManager:
    def __init__(self, X, y, scoring, cv=5):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.gsearch = GranularGridSearchCVSave(X, y)
        wj_filename = self.gsearch.data_hash + '.wj'
        self.wjournal = WorkJournal(wj_filename)

    def add_work(self, *args, **kwargs):
        return self.wjournal.add_work(*args, **kwargs)

    def perform_work(self, n_jobs=-1, verbose=True):
        done_work = self.wjournal.done_work
        if done_work and verbose:
            print("This work has already done:{}".format(done_work))

        for task in self.wjournal:
            if task not in done_work:
                self._perform_task(task, n_jobs)
                self.wjournal.task_done(task)
                if verbose:
                    print("The task {} done".format(task))

    def get_done_work(self):
        return self.wjournal.done_work

    def _perform_task(self, task, n_jobs):
        model_class, params = task
        est = model_class()
        try:
            self.gsearch.fit_and_save(est, params=params, scoring=self.scoring, cv=self.cv)
        except Exception as e:
            print('###\nException occured during task {} execution:\n{}'.format(task, e))
            if n_jobs != 1:
                print('Trying to run the task in 1 thread..')
                try:
                    self.gsearch.fit_and_save(est, params=params, scoring=self.scoring, cv=self.cv)
                    return
                except Exception as e:
                    print('###\nAnother exception occured during task {} execution in 1 thread\n{}'.format(task, e))
            print('Task execution terminated. Going to the next task.')


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
    def __init__(self, X, y, folder='grid_search_results'):
        self.X = X
        self.y = y
        self.data_hash = sha224(str(self.X) + str(self.y)).hexdigest()
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def get_score_from_file(self, estimator_class, cv=5, filename=None):

        if not filename:
            filename = self._get_default_name(estimator_class, cv)
        elif filename:
            if filename.rpartition('.')[-1] != 'txt':
                raise ValueError

        filename = self.folder + os.sep + filename
        data = pd.read_csv(filename, delimiter='\t')
        size = data.shape[0]

        if not size:
            return None

        result = [None] * size
        data.params = data.params.apply(literal_eval)
        data.sort_values('mean', inplace=True, ascending=False)

        for i, row in enumerate(data.itertuples()):
            result[i] = row.mean, row.std, estimator_class().set_params(**row.params)
        return result

    def fit_and_save(self, estimator, params, scoring, filename=None, verbose=False, cv=5, cvs_n_jobs=1):

        h_e_a_d_e_r = 'mean\tstd\tcv\tparams\n'

        params_combinations = self._params_combinations(params)

        if not filename:
            filename = self._get_default_name(estimator, cv)
        elif filename:
            if filename.rpartition('.')[-1] != 'txt':
                raise ValueError

        filename = self.folder + os.sep + filename

        header_flag = self._check_header(filename, header=h_e_a_d_e_r)

        if header_flag:
            start_position = self._start_position(filename, params_combinations)
            params_combinations = params_combinations[start_position + 1:]

        if not header_flag:
            with open(filename, 'a') as the_file:
                the_file.write(h_e_a_d_e_r)

        for params in params_combinations:
            estimator.set_params(**params)
            result = cross_val_score(estimator, X=self.X, y=self.y, scoring=scoring, cv=cv, n_jobs=cvs_n_jobs)
            mean = np.mean(result)
            std = np.std(result)
            result_string = '\t'.join(map(str, [mean, std, cv, params])) + '\n'
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
            return -1
        except IndexError:
            return -1

    def _get_default_name(self, estimator, cv):
        try:
            if issubclass(estimator, BaseEstimator):
                model_name = str(estimator).rpartition('.')[-1][:-2]
            else:
                raise AttributeError
        except TypeError:
            model_name = str(estimator).partition('(')[0]
        finally:
            return '_'.join([self.data_hash, model_name, str(cv)]) + '.txt'
