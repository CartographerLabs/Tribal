import pickle
import os
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyvis.network import Network
import hashlib
import random
import numpy as np
import json  # Make sure to import the json module

class TextCluster:
    text_data = []
    dictionary = corpora.Dictionary()  # Initialize to an empty dictionary object
    corpus = None
    ldamodel = None

    def __init__(self, text_data=None, num_topics=3, random_seed=42, cache_file="text_cluster_cache.pkl"):
        self.num_topics = num_topics
        self.random_seed = random_seed
        self.cache_file = cache_file
        self.stop_words = set(stopwords.words('english'))
        self.topic_colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6','#c4e17f','#76D7C4','#F7DC6F','#F0B27A']
        
        self.text_data = text_data
        self.initialize_from_scratch()

    def initialize_from_scratch(self):
        self.texts = self.preprocess_text(self.text_data)
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.ldamodel = models.LdaModel(self.corpus, num_topics=self.num_topics, id2word=self.dictionary, random_state=self.random_seed)
        self.net = Network(notebook=True, width='1000px', height='800px')
        self.initialize_network()

    def initialize_network(self):
        topic_keywords = self.get_topic_keywords()
        
        # add nodes for topics with more prominence
        for i in range(self.num_topics):
            self.net.add_node(f"Topic {i}", label=topic_keywords[i], color=self.topic_colors[i % len(self.topic_colors)], size=20, shape='dot')

        # add nodes for documents and their dominant topics
        document_nodes = {}
        for i, text_bow in enumerate(self.corpus):
            topic_distribution = self.ldamodel.get_document_topics(text_bow)
            dominant_topic, weight = max(topic_distribution, key=lambda x: x[1])  # get the topic with the highest weight
            document_hash = self.hash_text(self.text_data[i])
            document_nodes[i] = document_hash
            self.net.add_node(document_hash, label=document_hash, color=self.topic_colors[dominant_topic % len(self.topic_colors)])
            # add edge between document and its dominant topic
            self.net.add_edge(document_hash, f"Topic {dominant_topic}", value=float(weight), title=f"Topic {dominant_topic}")

    def preprocess_text(self, text_data):
        return [[word for word in word_tokenize(document.lower()) if word not in self.stop_words] for document in text_data]

    def hash_text(self, text):
        return hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]  # using SHA-1 for hashing and taking first 8 characters

    def add_new_text(self, new_text):
        if self.ldamodel is None:
            print("Error: LDA model not initialized. Cannot add new text.")
            return None
        
        self.text_data.append(new_text)
        new_text_preprocessed = [word for word in word_tokenize(new_text.lower()) if word not in self.stop_words]
        new_text_bow = self.dictionary.doc2bow(new_text_preprocessed)
        new_text_topics = self.ldamodel.get_document_topics(new_text_bow)
        
        # Find dominant topic for the new text
        dominant_topic, weight = max(new_text_topics, key=lambda x: x[1])
        return dominant_topic
    
    def visualize_graph(self):
        # Clear previous network data if any
        self.net.clear()
        
        # Initialize the network with topics and existing documents
        self.initialize_network()
        
        # Visualize the graph
        self.net.show("topics_graph.html")


    def load_from_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                cached_data = json.load(f)
                self.num_topics = cached_data['num_topics']
                self.random_seed = cached_data['random_seed']
                self.text_data = cached_data['text_data']
                self.stop_words = set(cached_data['stop_words']) if 'stop_words' in cached_data else set()  # Convert list back to set
                self.topic_colors = cached_data['topic_colors']
                self.dictionary = corpora.Dictionary()
                self.dictionary.token2id.update(cached_data['dictionary']) if 'dictionary' in cached_data else None  # Update dictionary token2id mapping
                self.corpus = [corpora.MmCorpus.serialize(None, doc) for doc in cached_data['corpus']] if 'corpus' in cached_data else None  # Deserialize corpus
                self.ldamodel = models.ldamodel.LdaModel.load(cached_data['ldamodel']) if 'ldamodel' in cached_data else None  # Load ldamodel
                self.net = Network()  # Reinitialize the network
                self.initialize_network()  # Reinitialize the network data
        except (EOFError, FileNotFoundError) as e:
            print(f"Error: {type(e).__name__} encountered while loading cache file '{self.cache_file}'")
        except Exception as e:
            print(f"Error loading cache file '{self.cache_file}': {str(e)}")

    def get_topic_keywords(self):
        topic_keywords = []
        for i in range(self.num_topics):
            topic_words = [word for word, _ in self.ldamodel.show_topic(i, topn=3)]
            topic_keywords.append('-'.join(topic_words))
        return topic_keywords

    def print_topics_and_keywords(self):
        topics = self.ldamodel.print_topics(num_topics=self.num_topics, num_words=3)
        for topic in topics:
            print(topic)
        
        for i in range(self.num_topics):
            topic_keywords = [word for word, _ in self.ldamodel.show_topic(i, topn=10)]
            print(f"Topic {i} keywords: {topic_keywords}")

# Example usage:
if __name__ == "__main__":
    # Initialize TextCluster object
    text_cluster = TextCluster(text_data=["Sample text 1", "Sample text 2"], num_topics=3)
    
    # Add new text
    dominant_topic = text_cluster.add_new_text("New sample text")
    print(f"Dominant topic for new text: {dominant_topic}")
    
    # Visualize the graph
    text_cluster.visualize_graph()
    
    # Save to cache
    text_cluster.save_to_cache()
