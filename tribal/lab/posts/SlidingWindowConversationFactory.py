from typing import List, Dict, Any
from tribal.lab.posts.PostFactory import PostFactory, Post

class SlidingWindowConversationFactory(PostFactory):
    """
    A factory class to create conversation groups from posts using specified patterns.

    Methods
    -------
    generate_conversations(window_size: int, slide: int) -> List[List[Post]]:
        Generates conversations using a sliding window pattern.
    """

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        super().__init__(data)

    def generate_conversations(self, window_size: int = 10, slide: int = 0) -> List[List[Post]]:
        """
        Generate conversations using a sliding window pattern.

        Parameters
        ----------
        window_size : int
            The size of the sliding window.
        slide : int
            The overlap between windows. A slide of 0 means no overlap,
            and a slide equal to the window size means contiguous windows.

        Returns
        -------
        List[List[Post]]
            A list of conversations, each containing a subset of Post objects.
        """
        posts = self.generate_posts()
        conversations = []

        # Add windows of the specified size
        for i in range(0, len(posts) - window_size + 1, max(window_size - slide, 1)):
            conversations.append(posts[i:i + window_size])

        # Add remaining posts as a single conversation, if not already included
        if len(posts) % window_size != 0 and len(posts) > window_size:
            conversations.append(posts[-(len(posts) % window_size):])

        return conversations
