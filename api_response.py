from typing import Any, Dict, List, Union


class Response:
    """
    Represents an API response.

    Attributes:
        success (bool): Indicates whether the API call was successful or not.
        data (Union[List[Any], Dict[str, Any], None]): The data returned by the API call, if any.
        message (Union[str, None]): A message describing the result of the API call, if any.
    """

    def __init__(self, success: bool, data: Union[List[Any], Dict[str, Any], None] = None, message: Union[str, None] = None):
        self.success = success
        self.data = data
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the response to a dictionary format.

        Returns:
            Dict[str, Any]: A dictionary representing the response.
        """
        response = {"success": self.success}
        if self.data is not None:
            response["data"] = self.data
        if self.message is not None:
            response["message"] = self.message
        return response

    @classmethod
    def success(cls, data: Union[List[Any], Dict[str, Any], None] = None, message: Union[str, None] = None) -> 'Response':
        """
        Creates a successful API response.

        Args:
            data (Union[List[Any], Dict[str, Any], None]): The data to be included in the response.
            message (Union[str, None]): A message to be included in the response.

        Returns:
            Response: An instance of ApiResponse representing a successful response.

        Raises:
            ValueError: If neither data nor message is provided.
        """
        if data is None and message is None:
            raise ValueError(
                "Either data or message must be provided for a successful response")
        return cls(success=True, data=data, message=message)

    @classmethod
    def failure(cls, error: str, status_code: int = 400) -> 'Response':
        """
        Creates a failed API response.

        Args:
            error (str): The error message describing the failure.
            status_code (int): The HTTP status code to be returned, defaults to 400.

        Returns:
            Response: An instance of ApiResponse representing a failed response.
        """
        return cls(success=False, message=error), status_code
