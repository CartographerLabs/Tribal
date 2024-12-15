import re
import networkx as nx
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
from easyLLM.easyLLM import EasyLLM
from TRUNAJOD import surface_proxies, ttr
import spacy
import pytextrank
from easyLLM.easyLLM import EasyLLM

class CentralityFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__(property_name="centrality")
        self.mention_pattern = re.compile(r"@([A-Za-z0-9]+)")

    def build_graph(self, posts: List[Post]) -> nx.DiGraph:
        """
        Build a directed graph where each user is a node and mentions form edges.
        If user A mentions user B in a post, create a directed edge A -> B.
        """
        G = nx.DiGraph()
        # Add nodes for all known users (optional, nodes will be added as edges appear)
        users = set(post.username for post in posts)
        G.add_nodes_from(users)

        for post in posts:
            # Find all mentions in the post's text
            mentions = self.mention_pattern.findall(post.text)
            for mentioned_user in mentions:
                if mentioned_user != post.username:  # Avoid self loops if desired
                    # Add an edge from the post's author to the mentioned user
                    G.add_edge(post.username, mentioned_user)

        return G

    def compute_centrality(self, G: nx.DiGraph) -> Dict[str, float]:
        """
        Compute centrality for each user (node in the graph).
        Here we use degree centrality, but you can choose other measures.
        """
        # For a directed graph, degree_centrality is computed based on in+out degree.
        # networkx.degree_centrality works on undirected graphs, so we may choose another measure.
        # For simplicity, let's convert to undirected or just use in-degree centrality.
        
        # Convert to undirected for a simple measure:
        UG = G.to_undirected()
        centrality_scores = nx.degree_centrality(UG)
        return centrality_scores

    def extract_features(self, posts: List[Post]) -> None:
        """
        Build the mention graph, compute centrality, and update each post with the user's centrality.
        """
        G = self.build_graph(posts)
        centrality_scores = self.compute_centrality(G)

        # Update each post with its author's centrality score
        for post in posts:
            score = centrality_scores.get(post.username, 0.0)
            post.update_property(self.property_name, score)
