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
from tribal.lab.posts.Post import Post
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

class ExtremismFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, llm=None):
        super().__init__(property_name="extremism", llm=llm)

    def get_extremism_for_all_users(self, posts: List[Post], users: Set[str]) -> Dict[str, Optional[str]]:
        """
        Given a list of Post objects and a set of users, determine the degree of extremism for each user.
        Instead of returning a boolean, we now return a category indicating extremism level:
        Possible categories: "none", "low", "moderate", "high", "extreme"
        
        If USE_LLM is False, return None for all.
        
        The returned JSON object format for each user is now:
          {
            "extremism_level": "none|low|moderate|high|extreme",
            "rational": "Explanation of why this level was assigned"
          }
        """
        if not self.llm:
            return {user: None for user in users}

        prompt = f"""You are an expert in social media analysis for extremist content.
Given a set of posts from multiple users, determine the degree of extremism each user exhibits.

Definition of extremism (for reference):
Extremism is the promotion or advancement of an ideology[footnote 3] based on violence, hatred or intolerance[footnote 4], that aims to:

- negate or destroy the fundamental rights and freedoms[footnote 5] of others; or
- undermine, overturn or replace the UK’s system of liberal parliamentary democracy[footnote 6] and democratic rights[footnote 7]; or
- intentionally create a permissive environment for others to achieve the results in (1) or (2).

Assign an extremism level based on the user's overall content:
Possible categories: "none", "low", "moderate", "high", "extreme".

Posts:
"""
        # Organize posts by user
        user_posts_map = {u: [] for u in users}
        for post in posts:
            user_posts_map[post.username].append(post.text)

        schema_parts = {}
        for username, user_posts in user_posts_map.items():
            for text in user_posts:
                prompt += f"\n{username}: {text}"
            schema_parts[username] = {
                "extremism_level": "string",
                "rational": "string"
            }

        schema = json.dumps(schema_parts)
        schema_model = self.llm.generate_pydantic_model_from_json_schema("ExtremismSchema", schema)
        structured_prompt = self.llm.generate_json_prompt(schema_model, prompt)

        response = self.llm.ask_question(structured_prompt)
        self.llm._unload_model()
        self.llm.reset_dialogue()
        gc.collect()
        torch.cuda.empty_cache()

        # response should be a dict {username: {"extremism_level": str, "rational": str}}
        result = {}
        for user in users:
            if user in response and "extremism_level" in response[user]:
                result[user] = response[user]["extremism_level"]
            else:
                result[user] = None
        return result

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract extremism levels for each user and update each post with that level.
        """
        users = set(post.username for post in posts)
        user_extremism = self.get_extremism_for_all_users(posts, users)

        for post in posts:
            extremism_value = user_extremism.get(post.username, None)
            post.update_property(self.property_name, extremism_value)