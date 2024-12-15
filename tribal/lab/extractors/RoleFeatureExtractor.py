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
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk import pos_tag, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from nrclex import NRCLex
from easyLLM.easyLLM import EasyLLM
from TRUNAJOD import surface_proxies, ttr
import spacy
import pytextrank
from easyLLM.easyLLM import EasyLLM
from tribal.lab.posts.Post import Post

class RoleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, llm=None):
        super().__init__(property_name="role", llm=llm)

    def get_roles_for_all_users(self, posts: List[Post], users: Set[str]) -> Dict[str, str]:
        """
        Given a list of Post objects and a set of unique users, determine the roles for 
        all users by making a single LLM call that returns the role for each user.
        """
        if not self.llm:
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

        # Add all posts to the prompt
        schema_parts = {}
        for post in posts:
            prompt += f"\n{post.username}: {post.text}"
            schema_parts = {**schema_parts, post.username: {"role": "users_role", "explanation": "rational"}}
        # Create the schema
        schema = json.dumps(schema_parts)
        schema_model = self.llm.generate_pydantic_model_from_json_schema("RolesSchema", schema)
        structured_prompt = self.llm.generate_json_prompt(schema_model, prompt)

        response = self.llm.ask_question(structured_prompt)
        self.llm._unload_model()
        self.llm.reset_dialogue()
        gc.collect()
        torch.cuda.empty_cache()

        return response

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
            if role is not None:
                role = role["role"]
            post.update_property(self.property_name, role)

