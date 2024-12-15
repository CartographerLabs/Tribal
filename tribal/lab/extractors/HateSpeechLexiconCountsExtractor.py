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

class HateSpeechLexiconCountsExtractor(BaseFeatureExtractor):
    def __init__(self) -> None:
        super().__init__(property_name="hate_speech_lexicon_counts")
        self.hate_speech_terms = {"hate", "slur", "offensive"}

    def extract_features(self, posts: List[Post]) -> None:
        for post in posts:
            tokens = word_tokenize(post.text.lower())
            hate_terms_in_text = [t for t in tokens if t in self.hate_speech_terms]
            hate_term_counts = dict(Counter(hate_terms_in_text))
            post.update_property(self.property_name, hate_term_counts)
