from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from typing import Callable
from loguru import logger

from config.settings import api
from config.settings import api
from routes.router import base_router as router
from exceptions.exceptions import RAGAPIError, EntityDoesNotExistError, InvalidOperationError, AuthenticationFailed, InvalidTokenError, ServiceError, TypeError

app = FastAPI(
    title=api.project_name,
    debug=api.debug,
    version=api.version
)
app.include_router(router, prefix=api.prefix)


def create_exception_handler(
    status_code: int, initial_detail: str
) -> Callable[[Request, RAGAPIError], JSONResponse]:
    detail = {"message": initial_detail}

    async def exception_handler(_: Request, exc: RAGAPIError) -> JSONResponse:
        if exc.message:
            detail["message"] = exc.message

        if exc.name:
            detail["message"] = f"{detail['message']} [{exc.name}]"

        logger.error(exc)
        return JSONResponse(
            status_code=status_code, content={"detail": detail["message"]}
        )

    return exception_handler


app.add_exception_handler(
    exc_class_or_status_code=EntityDoesNotExistError,
    handler=create_exception_handler(
        status.HTTP_404_NOT_FOUND, "Entity does not exist."
    ),
)

app.add_exception_handler(
    exc_class_or_status_code=InvalidOperationError,
    handler=create_exception_handler(
        status.HTTP_400_BAD_REQUEST, "Can't perform the operation."
    ),
)

app.add_exception_handler(
    exc_class_or_status_code=AuthenticationFailed,
    handler=create_exception_handler(
        status.HTTP_401_UNAUTHORIZED,
        "Authentication failed due to invalid credentials.",
    ),
)

app.add_exception_handler(
    exc_class_or_status_code=InvalidTokenError,
    handler=create_exception_handler(
        status.HTTP_401_UNAUTHORIZED, "Invalid token, please re-authenticate again."
    ),
)

app.add_exception_handler(
    exc_class_or_status_code=ServiceError,
    handler=create_exception_handler(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "A service seems to be down, try again later.",
    ),
)

app.add_exception_handler(
    exc_class_or_status_code=TypeError,
    handler=create_exception_handler(
        status.HTTP_400_BAD_REQUEST,
        "An unexpected type error occurred.",
    ),
)
