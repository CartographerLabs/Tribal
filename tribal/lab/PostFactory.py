from tribal.posts.Post import Post
from typing import List, Dict, Any

class PostFactory:
    """
    Base class for generating Post objects from a list of dictionaries.

    Attributes
    ----------
    data : List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents a post.

    Methods
    -------
    input_format() -> Dict[str, str]:
        Returns the expected format of the input data.
    generate_posts() -> List[Post]:
        Generates a list of Post objects from the input data.
    """

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    @staticmethod
    def input_format() -> Dict[str, str]:
        """
        Returns the expected format of the input data.

        Returns
        -------
        Dict[str, str]
            A dictionary describing the expected keys and their value types.
        """
        return {
            "post": "str - The content of the post.",
            "username": "str - The username of the post author.",
            "time": "Optional - Timestamp of the post.",
            "replying_to": "Optional - Indicates if the post is a reply to another post.",
            "post_id": "Optional - A unique identifier for the post.",
        }

    def generate_posts(self) -> List[Post]:
        """
        Generate a list of Post objects from the input data.
        """
        posts = [Post(**entry) for entry in self.data]
        return posts
