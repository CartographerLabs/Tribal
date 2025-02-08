from tribal.lab.extractors.BaseFeatureExtractor import BaseFeatureExtractor
from tribal.lab.posts.Post import Post
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

class SentimentFeatureExtractor(BaseFeatureExtractor):
    """
    A sentiment analysis feature extractor that uses VADER to analyze the sentiment
    of a given Post object's text and updates the post with the sentiment scores.
    """

    def __init__(self) -> None:
        """
        Initialize the SentimentFeatureExtractor with a VADER sentiment intensity analyzer.
        Sets property_name to 'sentiment'.
        """
        super().__init__(property_name="sentiment")
        self.analyzer = SentimentIntensityAnalyzer()

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract sentiment features from each Post object in the provided list and update
        the Post objects with the sentiment data under the property_name attribute.

        Parameters
        ----------
        posts : List[Post]
            A list of Post objects.
        """
        for post in posts:
            sentiment_scores = self.analyzer.polarity_scores(post.text)["compound"]
            post.update_property(self.property_name, sentiment_scores)