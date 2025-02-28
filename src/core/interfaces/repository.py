"""Abstract base class for repositories following the Repository Pattern."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional

T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """
    Generic repository interface defining standard CRUD operations.

    This abstract base class ensures all repositories follow the same contract.
    """

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        """Retrieve an entity by ID."""
        pass

    @abstractmethod
    def list(self) -> List[T]:
        """List all entities."""
        pass

    @abstractmethod
    def create(self, entity: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete an entity by ID."""
        pass
