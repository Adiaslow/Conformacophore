"""Base service class implementing common business logic patterns."""

from typing import Generic, TypeVar
from ..interfaces.repository import Repository

T = TypeVar("T")


class BaseService(Generic[T]):
    """
    Base service class providing common business logic operations.

    Implements the Service Layer pattern and follows dependency injection.
    """

    def __init__(self, repository: Repository[T]):
        """Initialize service with repository dependency."""
        self._repository = repository

    def get_by_id(self, id: str) -> T:
        """
        Retrieve entity by ID with business logic validation.

        Args:
            id: Entity identifier

        Returns:
            Entity instance

        Raises:
            ValueError: If entity not found
        """
        entity = self._repository.get(id)
        if not entity:
            raise ValueError(f"Entity with id {id} not found")
        return entity
