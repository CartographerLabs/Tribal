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

class EngagementFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, llm=None):
        super().__init__(property_name="engagement", llm=llm)

    def get_engagement_for_all_users(self, posts: List[Post], users: Set[str]) -> Dict[str, Optional[str]]:
        """
        Given posts and a set of users, determine the engagement level of each user.
        Possible categories: "none", "low", "medium", "high"
        """
        if not self.llm:
            return {user: None for user in users}

        prompt = """You are an expert in analyzing user engagement in a conversation.
Given multiple users and their posts, determine each user's engagement level in the conversation.

Possible categories: "none", "low", "medium", "high".

Posts:
"""
        user_posts_map = {u: [] for u in users}
        for post in posts:
            user_posts_map[post.username].append(post.text)

        schema_parts = {}
        for username, texts in user_posts_map.items():
            for text in texts:
                prompt += f"\n{username}: {text}"
            schema_parts[username] = {
                "engagement_level": "string",
                "rational": "string"
            }

        schema = json.dumps(schema_parts)
        schema_model = self.llm.generate_pydantic_model_from_json_schema("EngagementSchema", schema)
        structured_prompt = self.llm.generate_json_prompt(schema_model, prompt)

        response = self.llm.ask_question(structured_prompt)
        self.llm._unload_model()
        self.llm.reset_dialogue()
        gc.collect()
        torch.cuda.empty_cache()

        results = {}
        for user in users:
            if user in response and "engagement_level" in response[user]:
                results[user] = response[user]["engagement_level"]
            else:
                results[user] = None

        return results

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract engagement level for each user and update each post.
        """
        users = set(post.username for post in posts)
        user_engagement = self.get_engagement_for_all_users(posts, users)

        for post in posts:
            engagement_level = user_engagement.get(post.username, None)
            post.update_property(self.property_name, engagement_level)