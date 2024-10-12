from tkinter import NO
from sentence_transformers import SentenceTransformer
import spacy
import pytextrank
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.text_cluster import TextCluster
from detoxify import Detoxify
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec
import os
import glob
from tqdm import tqdm
import json 
import csv 
import numpy as np
from nltk.tokenize import word_tokenize
from utils.easy_llm import EasyLLM

class FeatureExtractor():

    _idf_model = None
    _word2vec_model = None
    llm = EasyLLM()

    def __init__(self, list_of_text_for_topic_creation, list_of_baseline_posts_for_vec_model):
        self._build_idf_model(list_of_text_for_topic_creation)
        self._build_word_2_vec_model(list_of_baseline_posts_for_vec_model)

    ######################################
    
    def get_text_topic(self, text):

        dominant_topic = self._idf_model.add_new_text(text)
        return dominant_topic

    def get_words_per_sentance(self, text):
        return ""
    
    def get_syntactical_complexity(self, text):
        return ""
    
    def get_lexical_diversity(self, text):
        return ""

    def get_readability(self, text):
        return ""

    def get_toxicity(self, text):
        return Detoxify('original').predict(text)

    def get_sentiment(self, text):
        sentiment = SentimentIntensityAnalyzer()
        return sentiment.polarity_scores(text)
    
    def get_embeddings(self, text):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        sentences = [text]
        embeddings = model.encode(sentences)

        return embeddings

    def get_entities(self, text):
        return [""] #TODO

    def get_keywords(self, text, top_n=10):
        # Load a spaCy model, depending on language, scale, etc.
        nlp = spacy.load("en_core_web_sm")

        # Add PyTextRank to the spaCy pipeline
        nlp.add_pipe("textrank")

        # Process the text
        doc = nlp(text)

        # Extract phrases and their ranks
        phrases = [(phrase.text, phrase.rank) for phrase in doc._.phrases]
        
        # Sort phrases by rank in descending order
        sorted_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
        
        # Get the top N phrases
        top_phrases = sorted_phrases[:top_n]

        top_words = [phrase[0] for phrase in top_phrases]

        return top_words
        
    def get_text_vector(self, text):

        tokens = word_tokenize(text.lower())
        
        # Initialize vector
        vector = np.zeros((self.VECTOR_SIZE,))
        count = 0
        
        # Aggregate vectors for each token
        for token in tokens:
            if token in self.word2vec_model.wv:
                vector += self.word2vec_model.wv[token]
                count += 1
        
        # Average the vectors
        if count > 0:
            vector /= count
        
        # Ensure vector is exactly of size VECTOR_SIZE (500)
        if len(vector) > self.VECTOR_SIZE:
            vector = vector[:self.VECTOR_SIZE]
        elif len(vector) < self.VECTOR_SIZE:
            vector = np.pad(vector, (0, self.VECTOR_SIZE - len(vector)), 'constant')
        
        return vector
    
    def if_is_text_valid(text):
       
        if len(ds.get_words(text)) > 5:
            return True
        else:
            return False

    ################################

    def _build_idf_model(self, list_of_all_text, num_topics=5):
        self._idf_model = TextCluster(list_of_all_text, num_topics)

    def _build_word_2_vec_model(self, list_of_all_text):
        model = Word2Vec(min_count=1, window=5)  # vector size of 100 and window size of 5?
        model.build_vocab(list_of_all_text)  # prepare the model vocabulary
        model.model_trimmed_post_training = False
        model.train(list_of_all_text, total_examples=model.corpus_count,
                    epochs=model.epochs)  # train word vectors

        self._word2vec_model = model

