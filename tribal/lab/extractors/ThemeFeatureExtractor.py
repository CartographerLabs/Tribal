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
from TRUNAJOD import surface_proxies, ttr
import spacy
from tribal.lab.posts.Post import Post
from jsonformer import Jsonformer

class ThemeFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(property_name="topic", model=model, tokenizer=tokenizer)

    def get_theme_for_all_users(self, posts: List[Post], users: Set[str]) -> Dict[str, Optional[str]]:
        """
        Identify a single-word theme for each user's posts.
        If USE_LLM is False, return None for all.
        """
        if not self.model or not self.tokenizer:
            return {user: None for user in users}

        prompt = f"""You are an expert in summarizing user posts into a single broad theme.
Given a set of posts by multiple users, determine the primary one-word theme for each user.

Examples:
- Travel, Food, Gratitude, Fitness, Economy, Entertainment, Technology, etc.

Return a JSON object:
- Keys: usernames
- Values: An object containing the topic and rationale
"""
        user_posts_map = {u: [] for u in users}
        for post in posts:
            user_posts_map[post.username].append(post.text)

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
                    "topic": {
                        "type": "string",
                        "description": "Single word theme describing the user's posts"
                    },
                    "rational": {
                        "type": "string",
                        "description": "Explanation of why this theme was chosen"
                    }
                },
                "required": ["topic", "rational"],
                "additionalProperties": False
            }

        for username, user_posts in user_posts_map.items():
            for text in user_posts:
                prompt += f"\n{username}: {text}"

        jsonformer = Jsonformer(self.model, self.tokenizer, schema, prompt, max_number_tokens=300)
        structured_response = jsonformer()
        gc.collect()
        torch.cuda.empty_cache()

        return structured_response


    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract the main theme for each user and update each post.
        """
        users = set(post.username for post in posts)
        user_themes = self.get_theme_for_all_users(posts, users)

        for post in posts:
            topic = user_themes.get(post.username, None)
            post.update_property(self.property_name, topic)