class RAGAPIError(Exception):
    """base exception class"""

    def __init__(self, message: str = "Service is unavailable", name: str = "RAGAPIError"):
        self.message = message
        self.name = name
        super().__init__(self.message, self.name)


class ServiceError(RAGAPIError):
    """failures in external services or APIs, like a database or a third-party service"""

    pass


class EntityDoesNotExistError(RAGAPIError):
    """database returns nothing"""

    pass


class EntityAlreadyExistsError(RAGAPIError):
    """conflict detected, like trying to create a resource that already exists"""

    pass


class InvalidOperationError(RAGAPIError):
    """invalid operations like trying to delete a non-existing entity, etc."""

    pass


class AuthenticationFailed(RAGAPIError):
    """invalid authentication credentials"""

    pass


class InvalidTokenError(RAGAPIError):
    """invalid token"""

    pass


class TypeError(RAGAPIError):
    """invalid type"""

    pass
