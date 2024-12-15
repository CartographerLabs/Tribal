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

class PolarisationFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, llm=None):
        super().__init__(property_name="polarisation", llm=llm)

    def get_polarisation_for_all_posts(self, posts: List[Post]) -> Dict[int, Optional[str]]:
        """
        Given posts, determine the polarisation degree of each post.
        Possible categories: "none", "low", "moderate", "high", "extreme"
        """
        if not self.llm:
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

        schema = json.dumps(schema_parts)
        schema_model = self.llm.generate_pydantic_model_from_json_schema("PolarisationSchema", schema)
        structured_prompt = self.llm.generate_json_prompt(schema_model, prompt)

        response = self.llm.ask_question(structured_prompt)
        self.llm._unload_model()
        self.llm.reset_dialogue()
        gc.collect()
        torch.cuda.empty_cache()

        results = {}
        for i, _ in enumerate(posts):
            key = f"post_{i}"
            if key in response and "polarisation_level" in response[key]:
                results[i] = response[key]["polarisation_level"]
            else:
                results[i] = None

        return results

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract the polarisation level for each post and update the post.
        """
        post_polarisation = self.get_polarisation_for_all_posts(posts)
        for i, post in enumerate(posts):
            post.update_property(self.property_name, post_polarisation[i])