import ast
import time
import os
import pickle
from moments import journal

DONE_TAG = 'done'
WORK_KWG = 'work'
LATTICE = '#'
WJ_EXTENSION = '.wj'


class WorkJournal(journal.Journal):
    """
    Work journal class.
    It used to save all tasks and then made them one by one saving the progress.
    You can add new tasks to it and previous will be saved.
    """

    def __init__(self, filename=None):
        super(self.__class__, self).__init__()
        if not filename:
            filename = time.asctime() + WJ_EXTENSION
        elif os.path.isfile(filename):
            self.load(filename)

        self.file = filename
        self.pickle_file = filename.rpartition('.')[0] + '_pickled_models' + WJ_EXTENSION
        with open(self.pickle_file, 'w') as pf:
            pass

        self.size = len(self.entries())
        self.save(filename)

    def __iter__(self):
        return self

    def next(self):
        next_task = self._get_next_undone_work()
        if not next_task:
            raise StopIteration
        return next_task

    def task_done(self, task):
        model, params = task
        task_text = self._make_task_text(model, params)
        tag = str(hash(task_text))
        if not self._check_task_in_work(task_text, tag):
            raise ValueError('No such task in the work journal : ', task_text)

        task = self.tag(tag)[0]
        task.tags += [DONE_TAG]

    def add_work(self, *args, **kwargs):
        work = self._get_work(*args, **kwargs)

        size = len(self.entries())
        newmodels = []
        for pos, (model, params) in work:
            task_text = self._make_task_text(model, params)
            tag = str(hash(task_text))
            newmodels.append(model)
            if self._check_task_in_work(task_text, tag):
                print("Task in work already exist:" + task_text)
                continue

            self.make(task_text, position=size + pos, tags=[tag])

        self._pickle_models(newmodels)
        self.save(self.file)

    @property
    def done_work(self):
        done_work = []
        for entry in self.entries():
            if DONE_TAG in entry.tags:
                done_work.append(self._get_model_and_params_from_text_task(entry.data))
        return done_work

    def _get_next_undone_work(self):
        for entry in self.entries():
            if DONE_TAG not in entry.tags:
                return self._get_model_and_params_from_text_task(entry.data)

    def _get_model_and_params_from_text_task(self, text_task):
        text_model, _lattice, text_params = text_task.partition(LATTICE)
        model = self._unpickle_models()[text_model]
        with open('blablabla.txt','w') as the_file:
            the_file.write(text_task)
            
        params = ast.literal_eval(text_params)
        return model, params

    def _pickle_models(self, models):
        if not models:
            return

        output_dict = {model.__name__: model for model in models}
        with open(self.pickle_file, 'w') as pf:
            pickle.dump(output_dict, pf)

    def _unpickle_models(self):
        models_dict = None
        with open(self.pickle_file, 'r') as pf:
            models_dict = pickle.load(pf)
        return models_dict

    def _check_task_in_work(self, task_text, tag):
        if tag not in self.tags():
            return False

        texts = list(map(lambda x: x.data.strip(), self.tag(tag)))
        if task_text not in texts:
            return False

        return True

    def _get_work(self, *args, **kwargs):
        if 'work' in kwargs.keys():
            if isinstance(kwargs[WORK_KWG], dict):
                work = enumerate(kwargs[WORK_KWG].items())
            else:
                work = enumerate(kwargs[WORK_KWG])
        elif len(args) == 1 and isinstance(args[0][0], tuple):  # tuple or list of tasks
            work = enumerate(args[0])
        else:
            work = enumerate(args)

        return work

    def _make_task_text(self, model, params):
        return model.__name__ + LATTICE + str(params)