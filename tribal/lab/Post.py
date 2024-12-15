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

class Post:
    """
    A class representing a post object.
    It starts with a required text attribute, and additional attributes
    can be dynamically added at runtime via feature extractors.
    """

    def __init__(self, post: str, username: str, time=None, replying_to=None, post_id=None) -> None:
        """
        Initialize a Post object.

        Parameters
        ----------
        text : str
            The text content of the post.
        """
        self.text: str = post
        self.username: str = username
        self.time = time
        self.replying = replying_to
        self.id = post_id

    def update_property(self, key: str, value: Any) -> None:
        """
        Dynamically add or update an attribute on the Post object.
        This method attaches a new attribute directly to the instance.

        Parameters
        ----------
        key : str
            The attribute name to set on the Post instance.
        value : Any
            The attribute value to assign.
        """
        setattr(self, key, value)

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieve a dynamically added attribute from the Post object.
        If the attribute does not exist, return the provided default.

        Parameters
        ----------
        key : str
            The attribute name to retrieve from the Post instance.
        default : Any, optional
            The value to return if the attribute does not exist.

        Returns
        -------
        Any
            The attribute value or the default if not found.
        """
        return getattr(self, key, default)