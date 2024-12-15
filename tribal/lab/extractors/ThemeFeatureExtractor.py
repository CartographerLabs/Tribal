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

class ThemeFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, llm=None):
        super().__init__(property_name="topic", llm=llm)

    def get_theme_for_all_users(self, posts: List[Post], users: Set[str]) -> Dict[str, Optional[str]]:
        """
        Identify a single-word theme for each user's posts.
        If USE_LLM is False, return None for all.
        """
        if not self.llm:
            return {user: None for user in users}

        prompt = f"""You are an expert in summarizing user posts into a single broad theme.
Given a set of posts by multiple users, determine the primary one-word theme for each user.

Examples:
- Travel, Food, Gratitude, Fitness, Economy, Entertainment, Technology, etc.

Return a JSON object:
- Keys: usernames
- Values:
  {{
    "topic": "a single word topic",
    "rational": "Explanation referencing the user's posts"
  }}

Posts:
"""
        user_posts_map = {u: [] for u in users}
        for post in posts:
            user_posts_map[post.username].append(post.text)

        schema_parts = {}
        for username, user_posts in user_posts_map.items():
            for text in user_posts:
                prompt += f"\n{username}: {text}"
            schema_parts[username] = {
                "topic": "string",
                "rational": "string"
            }

        schema = json.dumps(schema_parts)
        schema_model = self.llm.generate_pydantic_model_from_json_schema("ThemeSchema", schema)
        structured_prompt = self.llm.generate_json_prompt(schema_model, prompt)

        response = self.llm.ask_question(structured_prompt)
        self.llm._unload_model()
        self.llm.reset_dialogue()
        gc.collect()
        torch.cuda.empty_cache()

        # response is {username: {"topic": str, "rational": str}}
        result = {}
        for user in users:
            if user in response and "topic" in response[user]:
                result[user] = response[user]["topic"]
            else:
                result[user] = None
        return result

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract the main theme for each user and update each post.
        """
        users = set(post.username for post in posts)
        user_themes = self.get_theme_for_all_users(posts, users)

        for post in posts:
            topic = user_themes.get(post.username, None)
            post.update_property(self.property_name, topic)