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

class LexicalDiversityExtractor(BaseFeatureExtractor):
    def __init__(self) -> None:
        super().__init__(property_name="lexical_diversity")
        self.nlp = en_core_web_sm.load()
        if "textrank" not in self.nlp.pipe_names:
            self.nlp.add_pipe("textrank")

    def extract_features(self, posts: List[Post]) -> None:
        for post in posts:
            doc = self.nlp(post.text)
            try:
                val = ttr.lexical_diversity_mtld(doc)
            except ZeroDivisionError:
                val = 0.0
            post.update_property(self.property_name, float(val))