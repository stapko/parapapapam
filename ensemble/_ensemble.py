import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from ..metrics import METRICS


class Blender:

    def make_greedy_blend(self, X, y, models, scores=None, scoring=None,
                          cv=3, proba=True, random_state=42, verbose=False):
        """
        This func makes greedy blend for many models.

        Attributes
        ----------
        X : array-like
            The data to fit.
        y : array-like
            The target variable to try to predict.
        models : list
            List of models to blend.
        scores : list, optional, default: None
            List of scores for this models on the X, y data.
        scoring : string or callable, optional
            Scoring function from sklearn.
        cv : int, cross validation generator, optional, default: 3
            Cross validation from sklearn or number of folds.
        proba : bool, optional, default: True
            If true than probabilities were predicted else labels.
        random_state : int, optional, default: 42

        Returns
        -------

        """
        if not scores:
            scores = self._evaluate_models(X, y, models, scoring, cv)

        models = np.array(models)
        scores = np.array(scores)
        isorted = np.argsort(scores)[::-1]

        blend = BlendClassifier((models[isorted][0],), (1,))
        best_scores = np.array([scores[isorted][0]])
        if verbose:
            print('First blending model:\n{}\nScore: {}'
                  .format(models[isorted][0], scores[isorted][0]))

        for model in models[isorted][1:]:
            score, alpha = self.blend_two_models(X, y, blend, model, scoring=scoring, cv=cv,
                                                 proba=proba, random_state=random_state)
            print('score : {}\nalpha : {}'.format(score, alpha))
            if score > best_scores[-1]:
                blend = blend.get_updated_classifier((model, ), (1 - alpha, ))
                np.append(best_scores, score)
                if verbose:
                    print('The model added to blending:\n{}\nCoef: {}\nNew score: {}'
                          .format(model, 1 - alpha, score))
            elif verbose:
                print('The model is not added to blending:\n{}'
                      .format(model))

        return blend, best_scores

    def blend_two_models(self, X, y, est1, est2, scoring, cv,
                         proba=True, random_state=42):
        """
        This func blends two estimators as the following combination:
            alpha * est1_prediction + (1 - alpha) * est2_prediction
        and finds the best combination.

        Attributes
        ----------
        X : array-like
            The data to fit.
        y : array-like
            The target variable to try to predict.
        est1 : estimator
            The first estimator to blend.
        est2 : estimator
            The second estimator to blend.
        scoring : string or callable, optional
            Scoring function from sklearn.
        cv : int, cross validation generator, optional, default: 3
            Cross validation from sklearn or number of folds.
        proba : bool, optional, default: True
            If true than probabilities were predicted else labels.
        random_state : int, optional, default: 42

        Returns
        -------
        best_score : float
            The best score of blending.
        best_alpha : float
            The alpha parameter of best blending combination.
        """
        try:
            metric = METRICS[scoring]
        except KeyError:
            metrics = [metric for metric in METRICS]
            raise ValueError('%r is not a valid scoring value. '
                             'Valid options are %s'
                             % (scoring, sorted(metrics)))

        weights = np.linspace(0, 1, 101)
        method = 'predict_proba' if proba else 'predict'

        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, random_state=random_state)

        preds1 = cross_val_predict(est1, X, y, cv=cv, method=method)
        preds2 = cross_val_predict(est2, X, y, cv=cv, method=method)
        if proba:
            preds1, preds2 = preds1[:, 1], preds2[:, 1]

        best_score, best_alpha = metric(y, preds1), 1
        for idx, alpha in enumerate(weights):
            preds = alpha * preds1 + (1 - alpha) * preds2
            score = metric(y, preds)
            if score > best_score:
                best_score = score
                best_alpha = alpha

        return best_score, best_alpha

    def _evaluate_models(self, X, y, models, scoring, cv):
        scores = []
        for model in models:
            scores.append(cross_val_score(model, X, y, scoring=scoring, cv=cv).mean())
        return scores

    def _get_n_best_estimators_from_each_class(self, task_manager, n, classes):
        if classes is None:
            classes = task_manager.get_done_model_classes()

        models = []
        for cls in classes:
            models.extend(task_manager.get_best_models(cls, n))

        return list(reversed(sorted(models, key=lambda x: x[0])))

    def _get_models_with_scores(self, models, scores):
        return np.array(list(reversed(sorted(zip(scores, models)))))


class BlendClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators=(SVC(),), coefs=(1,)):
        """
        Estimator which result is blending of many models.

        Parameters
        ----------
        estimators : tuple of classifiers, optional, default: (SVC(),)
            The tuple of classifiers to blend.
        coefs : tuple of numbers (float, int), optional, default: (1,)
            The tuple of coefficients for classifiers blending.
        """

        self._check_params(estimators, coefs)
        self.estimators = estimators
        self.coefs = coefs

    def __repr__(self):
        output = ''
        for idx, (est, coef) in enumerate(zip(self.estimators, self.coefs)):
            output += 'step {}. {} : {}\n'.format(idx + 1, est, coef)
        return output

    @property
    def n_estimators(self):
        return len(self.estimators)

    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X):
        result = np.zeros(len(X))
        for coef, est in zip(self.coefs, self.estimators):
            result += coef * est.predict(X)
        return result

    def predict_proba(self, X):
        result = np.zeros((len(X), 2))
        for coef, est in zip(self.coefs, self.estimators):
            result += coef * est.predict_proba(X)
        return result

    def get_updated_classifier(self, new_estimators, new_coefs):
        """
        This func makes updated blending classifier
        and returns new blending classifier.

        Parameters
        ----------
        new_estimators : tuple of estimators
            The tuple of new models to blend.
        new_coefs : tuple of numbers (float, int)
            The tuple of coefficients of new models
            If sum of coefficients were equal to 1 then all models will be added
            with saving that rule. I.e. models will be added one by one and coefs
            will be updated by the following algorithm:
                - we have coefs array and new_coef to be added
                - coefs = coefs * (1 - new_coef)
                - coefs.append(new_coef)

        Returns
        -------
        blend_clf : BlendClassifier
            Updated blend classifier
        """

        _estimators = self.estimators + new_estimators
        _coefs = self.coefs
        if self._check_coefs_sum(verbose=True):
            # Append each coefficient saving whole sum equal to 1
            for coef in new_coefs:
                _coefs = tuple(map(lambda x: x * (1 - coef), self.coefs))
                _coefs = _coefs + (coef,)
        else:
            _coefs += new_coefs

        blend_clf = BlendClassifier(_estimators, _coefs)
        return blend_clf

    def _check_coefs_sum(self, verbose=True):
        if len(self.coefs) > 0 and sum(self.coefs) != 1:
            if verbose:
                print('WARNING: the sum of coefficients is not equal to 1.')
            return False
        return True

    def _check_array_elems_type(self, arr, types):
        for elem in arr:
            if not isinstance(elem, types):
                return False
        return True

    def _get_models_and_coefs(self, *args):
        if len(args) % 2:
            raise ValueError('Number of model and number of coefficients must be equal.')

        if len(args) == 2 and isinstance(args[0], (list, tuple)):
            models, coefs = args[0], args[1]
        else:
            models, coefs = args[:len(args) / 2], args[len(args) / 2:]

        if not self._check_array_elems_type(models, BaseEstimator):
            raise ValueError('All models must have some of estimator type.')
        if not self._check_array_elems_type(coefs, (float, int)):
            raise ValueError('All coefficients must be float.')

        return models, coefs

    def _check_params(self, estimators, coefs):
        if len(estimators) != len(coefs):
            raise ValueError('Number of estimators and number of coefficients must be the same.\n'
                             'Given estimators parameter has len {} and coefs parameter has len {}'
                             .format(len(estimators), len(coefs)))
        if not isinstance(estimators, tuple):
            raise ValueError('The estimators parameter must be a tuple type.\n'
                             'Given estimators parameter have {} type'.format(type(estimators)))
        if not isinstance(coefs, tuple):
            raise ValueError('The coefs must be a tuple type.\n'
                             'Given coefs parameter have {} type'.format(type(coefs)))