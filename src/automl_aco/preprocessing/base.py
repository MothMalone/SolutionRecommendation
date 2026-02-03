"""Base interfaces for preprocessing steps."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStep(ABC):
    @abstractmethod
    def fit(self, X, y=None) -> "BaseStep":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)
