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

class RecruitmentFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(property_name="recruitment", model=model, tokenizer=tokenizer)

    def get_recruitment_for_all_posts(self, posts: List[Post]) -> Dict[int, Optional[str]]:
        """
        Given a list of posts, determine the extent of recruitment effort for each post.
        Returns a dict keyed by the index of the post or post.id if available.
        Each value is one of: "none", "weak", "moderate", "strong", "extreme".
        If USE_LLM is False, return None for all posts.
        """
        if not self.model or not self.tokenizer:
            return {i: None for i, _ in enumerate(posts)}

        prompt = """You are an expert in analyzing recruitment signals in social media posts.
Given the following posts, classify the extent to which each post is actively recruiting others to the poster's cause.

Possible output categories: "none", "weak", "moderate", "strong", "extreme".

Posts:
"""
        schema = {
            "type": "object",
            "properties": {},
            "required": [f"post_{i}" for i in range(len(posts))],
            "additionalProperties": False
        }

        for i, post in enumerate(posts):
            prompt += f"\npost_{i}: {post.text}"
            schema["properties"][f"post_{i}"] = {
                "type": "object",
                "properties": {
                    "recruitment_level": {
                        "type": "string",
                        "enum": ["none", "weak", "moderate", "strong", "extreme"],
                        "description": "The level of recruitment effort detected in the post"
                    },
                    "rational": {
                        "type": "string",
                        "description": "Explanation of why this recruitment level was assigned"
                    }
                },
                "required": ["recruitment_level", "rational"],
                "additionalProperties": False
            }

        jsonformer = Jsonformer(self.model, self.tokenizer, schema, prompt, max_number_tokens=300)
        structured_response = jsonformer()
        gc.collect()
        torch.cuda.empty_cache()

        return structured_response

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract recruitment level for each post and update the post.
        """
        post_recruitment = self.get_recruitment_for_all_posts(posts)

        for post in posts:
            recruitment_value = post_recruitment.get(post.username, None)
            post.update_property(self.property_name, recruitment_value)