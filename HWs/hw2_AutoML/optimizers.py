import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from distributions import BaseDistribution, NumericalDistribution


class BaseOptimizer(BaseEstimator, ABC):
    '''
    A base class for all hyperparameter optimizers
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, n_jobs: Optional[int] = None,
                 verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - estimator: sklearn model instance
          - param_distributions: a dictionary of parameter distributions,
            e.g. param_distributions['num_epochs'] = IntUniformDistribution(100, 200)
          - scoring: sklearn scoring object, see
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            if left None estimator must have 'score' attribute
          - cv: number of folds to cross-validate
          - num_runs: total number of iterations to fit hyperparameters (including num_dry_runs)
          - num_dry_runs: number of dry runs (i.e. random strategy steps) to gather initial statistics
          - num_samples_per_run: number of hyperparameters set to sample each iteration
          - n_jobs: number of parallel processes to fit algorithms
          - verbose: whether to print debugging information (you can configure debug as you wish)
          - random_state: RNG seed to control reproducibility
        '''
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.num_runs = num_runs
        self.num_samples_per_run = num_samples_per_run
        self.num_dry_runs = num_dry_runs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.reset()

    def reset(self):
        '''
        Reset fields used for fitting
        '''
        self.splitter = None
        self.best_score = None
        self.best_params = None
        self.best_estimator = None
        self.params_history = {
            name: np.array([]) for name in self.param_distributions
        }
        self.scores_history = np.array([])
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def sample_params(self) -> Dict[str, np.array]:
        '''
        Sample self.num_samples_per_run set of hyperparameters
        Returns:
          - sampled_params: dict of arrays of parameter samples,
            e.g. sampled_params['num_epochs'] = np.array([178, 112, 155])
        '''
        sampled_params = {}
        for name, distr in self.param_distributions.items():
            sampled_params[name] = distr.sample(self.num_samples_per_run)
        return sampled_params

    @abstractmethod
    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        '''
        Select new set of parameters according to a specific search strategy
        Params:
          - params_history: list of hyperparameter values from previous interations
          - scores_history: corresponding array of CV scores
          - sampled_params: dict of arrays of parameter samples to select from
        Returns:
          - new_params: a dict of new hyperparameter values
        '''
        msg = f'method \"select_params\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def cross_validate(self, X: np.array, y: Optional[np.array],
                       params: Dict[str, Any]) -> float:
        '''
        Calculate cross-validation score for a set of params
        Consider using sklearn.model_selection.cross_validate()
        Also use self.splitter as a cv parameter in cross_validate
        Params:
          - X: object features
          - y: object labels
          - params: a set of params to score
        Returns:
          - score: mean cross-validation score
        '''
        params_to_set = self.estimator.get_params()
        params_to_set.update(params)
        if hasattr(self.estimator, 'random_state'):
            params_to_set['random_state'] = self.random_state

        estimator = type(self.estimator)(**params_to_set)
        result = cross_validate(estimator, X, y, scoring=self.scoring, cv=self.splitter)
        score = result['test_score'].mean()
        return score

    def fit(self, X_train: np.array, y_train: Optional[np.array] = None) -> BaseEstimator:
        '''
        Find the best set of hyperparameters with a specific search strategy
        using cross-validation and fit self.best_estimator on whole training set
        Params:
          - X_train: array of train features of shape (num_samples, num_features)
          - y_train: array of train labels of shape (num_samples, )
            if left None task is unsupervised
        Returns:
          - self: (sklearn standard convention)
        '''
        self.reset()
        if y_train is not None and np.issubdtype(y_train.dtype, np.integer):
            self.splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.random_state)
        else:
            self.splitter = KFold(n_splits=self.cv, shuffle=True,
                                  random_state=self.random_state)

        if self.verbose:
            print(f'Starting new trial for {self.num_runs} iterations with {self.cv} CV folds')

        for i in range(self.num_runs):
            sampled_params = self.sample_params()
            new_params = self.select_params(self.params_history, self.scores_history,
                                            sampled_params)
            score = self.cross_validate(X_train, y_train, new_params)

            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_params = new_params

            self.scores_history = np.append(self.scores_history, [score])
            for name, value in new_params.items():
                self.params_history[name] = np.append(
                    self.params_history[name], [value]
                )

            if self.verbose:
                print(f'Iteration {i + 1}/{self.num_runs}: fitted params {new_params} with score {score:.4f}. '
                      f'Current best score {self.best_score:.4f}')

        params_to_set = self.estimator.get_params()
        params_to_set.update(self.best_params)
        if hasattr(self.estimator, 'random_state'):
            params_to_set['random_state'] = self.random_state

        self.best_estimator = type(self.estimator)(**params_to_set)
        self.best_estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.array) -> np.array:
        '''
        Generate a prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test predictions of shape (num_samples, )
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        y_pred = self.best_estimator.predict(X_test)
        return y_pred

    def predict_proba(self, X_test: np.array) -> np.array:
        '''
        Generate a probability prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test probabilities of shape (num_samples, num_classes)
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        if not hasattr(self.best_estimator, 'predict_proba'):
            raise ValueError('Estimator does not support predict_proba')

        y_pred = self.best_estimator.predict_proba(X_test)
        return y_pred


class RandomSearchOptimizer(BaseOptimizer):
    '''
    An optimizer implementing random search strategy
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        super().__init__(
            estimator, param_distributions, scoring, cv=cv,
            num_runs=num_runs, num_dry_runs=0, num_samples_per_run=1,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        return new_params


class GPOptimizer(BaseOptimizer):
    '''
    An optimizer implementing gaussian process strategy
    '''
    @staticmethod
    def calculate_expected_improvement(y_star: float, mu: np.array,
                                       sigma: np.array) -> np.array:
        '''
        Calculate EI values for passed parameters of normal distribution
        hint: consider using scipy.stats.norm
        Params:
          - y_star: optimal (maximal) score value
          - mu: array of mean values of normal distribution of size (num_samples_per_run, )
          - sigma: array of std values of normal distribution of size (num_samples_per_run, )
        Retuns:
          - ei: array of EI values of size (num_samples_per_run, )
        '''
        ei = np.zeros_like(mu)
        # Your code here (⊃｡•́‿•̀｡)⊃
        return ei

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        return new_params
