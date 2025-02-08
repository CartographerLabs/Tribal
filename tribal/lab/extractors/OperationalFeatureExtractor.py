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
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from nrclex import NRCLex
from TRUNAJOD import surface_proxies, ttr
import spacy
import pytextrank
from jsonformer import Jsonformer

from tribal.lab.posts.Post import Post

class OperationalFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(property_name="operational", model=model, tokenizer=tokenizer)

    def get_operational_for_all_users(self, posts: List[Post], users: Set[str]) -> Dict[str, Optional[str]]:
        """
        Given posts, determine the operational planning indicator for each user.
        Return values: "none", "weak", "moderate", "strong", or "extreme".
        If USE_LLM is False, return None for all.
        """
        if not self.model or not self.tokenizer:
            return {user: None for user in users}

        prompt = f"""You are an expert in detecting operational planning in extremist contexts.
Given a set of posts by multiple users, identify for each user the degree of operational planning.

Possible output indicators: "None", "Weak", "Moderate", "Strong", "Extreme".

Return a JSON object:
- Keys: usernames
- Values:
  {{
    "indicator": one of "None", "Weak", "Moderate", "Strong", "Extreme",
    "rational": "Explanation with examples."
  }}

Posts:
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
                    "indicator": {
                        "type": "string",
                        "enum": ["None", "Weak", "Moderate", "Strong", "Extreme"],
                        "description": "The level of operational planning detected in the user's posts"
                    },
                    "rational": {
                        "type": "string",
                        "description": "Detailed explanation with examples of why this indicator was assigned"
                    }
                },
                "required": ["indicator", "rational"],
                "additionalProperties": False
            }

        jsonformer = Jsonformer(self.model, self.tokenizer, schema, prompt, max_number_tokens=300)
        structured_response = jsonformer()
        gc.collect()
        torch.cuda.empty_cache()

        return structured_response

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract operational planning indicators for each user and update each post.
        """
        users = set(post.username for post in posts)
        user_indicators = self.get_operational_for_all_users(posts, users)

        for post in posts:
            indicator = user_indicators.get(post.username, None)
            post.update_property(self.property_name, indicator)