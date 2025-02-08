from tribal.lab.extractors.BaseFeatureExtractor import BaseFeatureExtractor
import spacy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Set, Tuple
from collections import Counter
from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from nrclex import NRCLex
import en_core_web_sm
from rich.progress import Progress
# Import necessary libraries
import json
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm
import torch
import gc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from nrclex import NRCLex
from TRUNAJOD import surface_proxies, ttr
import spacy
import pytextrank
from tribal.lab.posts.Post import Post
from jsonformer import Jsonformer

class RoleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(property_name="role", model=model, tokenizer=tokenizer)

    def get_roles_for_all_users(self, posts: List[Post], users: Set[str]) -> Dict[str, str]:
        """
        Given a list of Post objects and a set of unique users, determine the roles for 
        all users by making a single LLM call that returns the role for each user.
        """
        if not self.model or not self.tokenizer:
            return {user: None for user in users}

        prompt = """You are an expert in social media analysis of user roles. 
Given the following set of posts, define the most apparent role of each user. 

Possible roles (in order of priority if a user fits multiple categories):
- People Leader: Directs, recruits and mobilises members.
- Leader Influencer: Directs conversation as a knowledge source or gatekeeper.
- Engager Negator: Negative, berating, or attempts to reduce discussion.
- Engager Supporter: Positive, encourages further discussion and ideological success.
- Engager Neutral: Neutral interaction, learning or socially interacting.
- Bystander: Not engaged with main topics, but present.
- NATTAC: Not Applicable or unable to determine.

Posts:
"""

        schema = {
            "type": "object",
            "properties": {},
            "required": list(users),
            "additionalProperties": False
        }

        for username in users:
            schema["properties"][username] = {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": [
                            "People Leader",
                            "Leader Influencer",
                            "Engager Negator",
                            "Engager Supporter",
                            "Engager Neutral",
                            "Bystander",
                            "NATTAC"
                        ],
                        "description": "The user's primary role in the community"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Detailed explanation of why this role was assigned"
                    }
                },
                "required": ["role", "explanation"],
                "additionalProperties": False
            }

        # Add all posts to the prompt
        for post in posts:
            prompt += f"\n{post.username}: {post.text}"

        jsonformer = Jsonformer(self.model, self.tokenizer, schema, prompt, max_number_tokens=300)
        structured_response = jsonformer()
        gc.collect()
        torch.cuda.empty_cache()

        return structured_response

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract roles for each user from the given list of Post objects and update
        each post with the corresponding user's role.
        """
        # Get unique users
        users = set(post.username for post in posts)

        # Get roles for all users
        user_roles = self.get_roles_for_all_users(posts, users)

        # Update each post with the user's role
        for post in posts:
            role = user_roles.get(post.username, None)
            post.update_property(self.property_name, role)

