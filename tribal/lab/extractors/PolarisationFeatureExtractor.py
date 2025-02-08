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
from nrclex import NRCLex
from TRUNAJOD import surface_proxies, ttr
import spacy
import pytextrank
from tribal.lab.posts.Post import Post
from jsonformer import Jsonformer

class PolarisationFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(property_name="polarisation", model=model, tokenizer=tokenizer)

    def get_polarisation_for_all_posts(self, posts: List[Post]) -> Dict[int, Optional[str]]:
        """
        Given posts, determine the polarisation degree of each post.
        Possible categories: "none", "low", "moderate", "high", "extreme"
        """
        if not self.model or not self.tokenizer:
            return {i: None for i, _ in enumerate(posts)}

        prompt = """You are an expert in analyzing polarization in conversation posts.
For each post, determine how polarized it is. Consider polarization as the degree to which the post expresses a strongly one-sided, divisive, or extreme viewpoint.

Possible categories: "none", "low", "moderate", "high", "extreme".

Posts:
"""
        schema_parts = {}
        for i, post in enumerate(posts):
            prompt += f"\npost_{i}: {post.text}"
            schema_parts[f"post_{i}"] = {
                "polarisation_level": "string",
                "rational": "string"
            }

        jsonformer = Jsonformer(self.model, self.tokenizer, schema_parts, prompt)
        structured_response = jsonformer()
        gc.collect()
        torch.cuda.empty_cache()

        return structured_response

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract the polarisation level for each post and update the post.
        """
        post_polarisation = self.get_polarisation_for_all_posts(posts)
        for i, post in enumerate(posts):
            post.update_property(self.property_name, post_polarisation[i])