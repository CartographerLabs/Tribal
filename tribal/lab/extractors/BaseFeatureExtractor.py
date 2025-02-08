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

# Ensure required nltk packages are downloaded
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

class BaseFeatureExtractor:
    """
    A base class for feature extractors that operate on one or more Post objects.
    Derived classes should implement the `extract_features` method.
    It also holds a property_name attribute that can be overridden by subclasses.
    """

    def __init__(self, property_name: str, model = None, tokenizer = None) -> None:
        """
        Initialize the BaseFeatureExtractor with a given property name.

        Parameters
        ----------
        property_name : str
            The attribute name under which to store extracted features.
        """
        self.property_name: str = property_name

        if model:
            self.model = model

        if tokenizer:
            self.tokenizer = tokenizer

        if model and not tokenizer:
            raise ValueError("If a model is provided, a tokenizer must also be provided.")
        if tokenizer and not model:
            raise ValueError("If a tokenizer is provided, a model must also be provided.")

    def extract_features(self, posts: List[Post]) -> None:
        """
        Extract features from a list of Post objects.

        Parameters
        ----------
        posts : List[Post]
            A list of Post objects to process.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by a subclass.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")