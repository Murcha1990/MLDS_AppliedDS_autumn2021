import numpy as np
from typing import Union
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    '''
    A base class for all hyperparameter distributions
    '''
    @abstractmethod
    def sample(self, size: int) -> np.array:
        '''
        Generate "size" samples from the distribution
        Params:
          - size: number of samples
        Returns:
          - sample: generated samples np.array of shape (size, )
        '''
        msg = f'method \"sample\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)
    
    def __repr__(self):
        return 'BaseDistribution()'


class NumericalDistribution(BaseDistribution, ABC):
    '''
    A base class for all numerical hyperparameter distributions
    '''
    @abstractmethod
    def scale(self, sample: np.array) -> np.array:
        '''
        Scale sample from distribution to [0, 1] uniform range
        Params:
          - sample: np.array sample drawn from the distribution of arbitrary shape
        Returns:
          - scaled_sample: np.array scaled sample of the same shape
        '''
        msg = f'method \"scale\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def __repr__(self):
        return 'NumericalDistribution()'


class CategoricalDistribution(BaseDistribution):
    '''
    A categorical distribution over a finite set of elements
    '''
    def __init__(self, categories: Union[list, tuple, np.array]):
        '''
        Params:
          - categories: finite set of categories
          - p: probabilitites of categories (uniform by default)
        '''
        super().__init__()
        self.categories = np.array(categories)
        self.p = np.ones(len(categories)) / len(categories)
    
    def sample(self, size: int) -> np.array:
        # Hint: consider np.random.choice
        return super().sample(size)  # replace with your code

    def __repr__(self):
        return f'CategoricalDistribution({self.categories.tolist()})'


class UniformDistribution(NumericalDistribution):
    '''
    A uniform continuous distribution x ~ U[low, high]
    '''
    def __init__(self, low: float, high: float):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound
        '''
        super().__init__()
        assert high > low
        self.low = low
        self.high = high
    
    def sample(self, size: int) -> np.array:
        # Hint: consider np.random.uniform
        return super().sample(size)  # replace with your code

    def scale(self, sample: np.array) -> np.array:
        # Hint: scale sample to [0, 1] using self.low and self.high
        return super().scale(sample)  # replace with your code
    
    def __repr__(self):
        return f'UniformDistribution({self.low}, {self.high})'


class LogUniformDistribution(NumericalDistribution):
    '''
    A log-uniform continuous distribution x ~ exp(U[ln(a), ln(b)])
    '''
    def __init__(self, low: float, high: float):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound
        '''
        super().__init__()
        assert high > low and low > 0
        self.log_low = np.log(low)
        self.log_high = np.log(high)
    
    def sample(self, size: int) -> np.array:
        # Hint: use np.random.uniform and exponentiate sample
        return super().sample(size)  # replace with your code

    def scale(self, sample: np.array) -> np.array:
        # Hint: apply logarithm to sample and scale it to [0, 1] using self.log_low and self.log_high
        return super().scale(sample)  # replace with your code

    def __repr__(self):
        return f'LogUniformDistribution({np.exp(self.log_low):.2e}, {np.exp(self.log_high):.2e})'


class IntUniformDistribution(NumericalDistribution):
    '''
    A uniform discrete distribution x ~ U[{low, low + 1, ..., high}]
    '''
    def __init__(self, low: int, high: int):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound (included in range)
        '''
        super().__init__()
        assert type(low) == int and type(high) == int
        assert high >= low
        self.low = low
        self.high = high

    def sample(self, size: int) -> np.array:
        # Hint: use np.random.randint
        return super().sample(size)  # replace with your code

    def scale(self, sample: np.array) -> np.array:
        # Hint: scale sample to [0, 1] using self.low and self.high
        return super().scale(sample)  # replace with your code

    def __repr__(self):
        return f'IntUniformDistribution({self.low}, {self.high})'


class IntLogUniformDistribution(NumericalDistribution):
    '''
    A log-uniform discrete distribution x ~ int(exp(U[ln(a), ln(b)]))
    '''
    def __init__(self, low: int, high: int):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound (included in range)
        '''
        super().__init__()
        assert type(low) == int and type(high) == int
        assert high > low and low > 0
        self.log_low = np.log(low)
        self.log_high = np.log(high)

    def sample(self, size: int) -> np.array:
        # Hint: use np.random.uniform, exponentiate sample and cast to int
        return super().sample(size)  # replace with your code

    def scale(self, sample: np.array) -> np.array:
        # Hint: apply logarithm to sample and scale it to [0, 1] using self.log_low and self.log_high
        return super().scale(sample)  # replace with your code

    def __repr__(self):
        return f'IntLogUniformDistribution({self.low}, {self.high})'