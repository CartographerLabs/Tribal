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
import pytextrank
from tribal.lab.posts.Post import Post

class PositiveWordFrequencyExtractor(BaseFeatureExtractor):
    def __init__(self) -> None:
        super().__init__(property_name="positive_word_frequency")
        self.positive_words = {
            "good", "happy", "joy", "excellent", "great", "positive", "fortunate", "correct", "superior",
            "love", "wonderful", "awesome", "blessed", "success", "smile", "win", "beautiful", "nice",
            "amazing", "fantastic", "delight", "pleasure", "grateful", "best", "brilliant", "bright",
            "cheerful", "content", "enjoy", "glad", "hopeful", "lucky", "proud", "relaxed", "satisfy",
        }

    def extract_features(self, posts: List[Post]) -> None:
        for post in posts:
            tokens = word_tokenize(post.text.lower())
            positive_word_count = sum(1 for w in tokens if w in self.positive_words)
            total_words = len(tokens)
            val = 0 if total_words == 0 else positive_word_count / total_words
            post.update_property(self.property_name, val)
