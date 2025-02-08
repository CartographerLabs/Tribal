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

class NGramsExtractor(BaseFeatureExtractor):
    def __init__(self, n: int = 2, top_n: int = 10) -> None:
        super().__init__(property_name="n_grams")
        self.n = n
        self.top_n = top_n

    def extract_features(self, posts: List[Post]) -> None:
        for post in posts:
            tokens = word_tokenize(post.text.lower())
            n_gram_seq = ngrams(tokens, self.n)
            n_gram_counts = Counter(n_gram_seq)
            most_common_n_grams = n_gram_counts.most_common(self.top_n)
            result = [' '.join(gram) for gram, _ in most_common_n_grams]
            post.update_property(self.property_name, result)