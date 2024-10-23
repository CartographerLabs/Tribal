from abc import ABC, abstractmethod
import json
import numpy as np
import gc 
# Assuming FeatureExtractor is imported correctly
from Tribal.utils.feature_extractor import FeatureExtractor
from Tribal.utils.config_manager import ConfigManager
import pycuda.driver as cuda
import pycuda.autoinit


class PostObject(ABC):

    _topic = None
    _words_per_sentence = None
    _syntactical_complexity = None
    _lexical_diversity = None
    _readability = None
    _toxicity = None
    _sentiment = None
    _embeddings = None
    _entities = None
    _keywords = None
    _operational = None
    _theme = None

    # New features
    _capital_letter_word_frequency = None
    _pos_counts = None
    _n_grams = None
    _emotion_scores = None
    _hate_speech_lexicon_counts = None
    _tf_idf_vector = None
    _text_vector = None

    ##########
    _post = None
    _post_id = None
    _time = None
    _replying = None
    _username = None

    ########

    _feature_extractor = None

    def __init__(self, feature_extractor) -> None:
        super().__init__()
        self._feature_extractor = feature_extractor

    #################################

    @property
    def post(self):
        return self._post

    @post.setter
    def post(self, value):
        self._post = value

    @property
    def post_id(self):
        return self._post_id

    @post_id.setter
    def post_id(self, value):
        self._post_id = value

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def replying(self):
        return self._replying

    @replying.setter
    def replying(self, value):
        self._replying = value

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, value):
        self._username = value

    #####
    @property
    def topic(self):
        if self._topic is None:
            self._topic = self._feature_extractor.get_text_topic(self.post)
        return self._topic

    @property
    def words_per_sentence(self):
        if self._words_per_sentence is None:
            self._words_per_sentence = self._feature_extractor.get_words_per_sentence(
                self.post
            )
        return self._words_per_sentence

    @property
    def syntactical_complexity(self):
        if self._syntactical_complexity is None:
            self._syntactical_complexity = (
                self._feature_extractor.get_syntactical_complexity(self.post)
            )
        return self._syntactical_complexity

    @property
    def lexical_diversity(self):
        if self._lexical_diversity is None:
            self._lexical_diversity = self._feature_extractor.get_lexical_diversity(
                self.post
            )
        return self._lexical_diversity

    @property
    def readability(self):
        if self._readability is None:
            self._readability = self._feature_extractor.get_readability(self.post)
        return self._readability

    @property
    def toxicity(self):
        if self._toxicity is None:
            self._toxicity = self._feature_extractor.get_toxicity(self.post)
        return self._toxicity

    @property
    def sentiment(self):
        if self._sentiment is None:
            self._sentiment = self._feature_extractor.get_sentiment(self.post)
        return self._sentiment

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = self._feature_extractor.get_embeddings(self.post)
        return self._embeddings

    @property
    def entities(self):
        if self._entities is None:
            self._entities = self._feature_extractor.get_entities(self.post)
        return self._entities

    @property
    def keywords(self):
        if self._keywords is None:
            self._keywords = self._feature_extractor.get_keywords(self.post)
        return self._keywords

    # New properties

    @property
    def capital_letter_word_frequency(self):
        if self._capital_letter_word_frequency is None:
            self._capital_letter_word_frequency = self._feature_extractor.get_capital_letter_word_frequency(
                self.post
            )
        return self._capital_letter_word_frequency

    @property
    def pos_counts(self):
        if self._pos_counts is None:
            self._pos_counts = self._feature_extractor.get_pos_counts(self.post)
        return self._pos_counts

    @property
    def n_grams(self):
        if self._n_grams is None:
            self._n_grams = self._feature_extractor.get_n_grams(self.post)
        return self._n_grams

    @property
    def emotion_scores(self):
        if self._emotion_scores is None:
            self._emotion_scores = self._feature_extractor.get_emotion_scores(self.post)
        return self._emotion_scores

    @property
    def hate_speech_lexicon_counts(self):
        if self._hate_speech_lexicon_counts is None:
            self._hate_speech_lexicon_counts = self._feature_extractor.get_hate_speech_lexicon_counts(
                self.post
            )
        return self._hate_speech_lexicon_counts

    @property
    def tf_idf_vector(self):
        if self._tf_idf_vector is None:
            self._tf_idf_vector = self._feature_extractor.get_tf_idf_vector(self.post)
        return self._tf_idf_vector

    @property
    def text_vector(self):
        if self._text_vector is None:
            self._text_vector = self._feature_extractor.get_text_vector(self.post)
        return self._text_vector

    @property
    def operational(self):
        if self._operational:
            return self._operational

        prompt = """You are an AI assistant tasked with analysing social media posts to identify content related to operational planning for extremist activities. Using the information from the research paper "Online Signals of Extremist Mobilization" and the rules established in our previous conversation, evaluate the given text and answer the following:1.
Does the post contain any keywords or phrases related to violent action? If so, what are they? Consider terms like "terrorist", "gun", "attack", "punch", "police", or any other language that suggests violence or harm.2.
Are there any indicators that the user is seeking to acquire knowledge or skills related to carrying out violence? Look for signs of the user attempting to learn about weapons, tactics, operational security, or ways to evade law enforcement.3.
Does the post include discussion about logistics, timelines, specific targets, or coordination with others? Identify any mentions of arranging meetups, planning travel, or sharing maps and schedules.4.
Does the text contain any "leakage" where the user might be unintentionally revealing their intentions or plans? Look for seemingly offhand remarks about future actions, boasts about capabilities, or expressions of being ready to act.5.
Assess the overall tone and emotional intensity of the post. Are there excessive punctuation marks, strong expressions of anger or excitement, or a sense of urgency that could indicate an individual is close to taking action?
Based on your analysis of these five points, provide a final assessment of whether the post exhibits strong, moderate, weak, or no signs of operational planning for extremist activities. Explain your reasoning, citing specific examples from the text and referencing the research paper where applicable.
Important Considerations:●
Context is key. Analyse the language and content in relation to the user's online activity, group affiliations, and any known offline behaviour.●
Avoid bias. Do not base your assessment solely on the user's ideology or group affiliation. Focus on the specific content and behaviours exhibited in the post.●
This is a complex task with ethical implications. False positives can have serious consequences. Emphasize that this is a tool for triaging data and should be used in conjunction with other investigative methods.
"""

        prompt = prompt + "post: " + self.post

        response_schema = json.dumps(
            {
                "is_operational_planning": "single word response on if the post contains strong, medium, weak, or no operational planning",
                "reasoning": "Your reasoning for why the post contains that degree of operational planning",
            }
        )
        schema_model = (
            self._feature_extractor.llm.generate_pydantic_model_from_json_schema(
                "Default", response_schema
            )
        )
        structured_prompt = self._feature_extractor.llm.generate_json_prompt(
            schema_model, prompt
        )
        response = self._feature_extractor.llm.ask_question(structured_prompt)
        self._feature_extractor.llm._unload_model()
        self._feature_extractor.llm.reset_dialogue()
        gc()
        cuda.Context.pop()
        self._operational = response["is_operational_planning"]
        return response["is_operational_planning"]

    @property
    def theme(self):

        if self._theme:
            return self._theme

        prompt = """Given the following social media post, identify the main theme or topic of the content in one word. Focus on the most prominent subject being discussed or conveyed."

Example: Post: 'Excited to visit the beach this weekend and enjoy some sunshine!'
Response: 'Travel'"""

        prompt = prompt + "post: " + self.post

        response_schema = json.dumps(
            {
                "theme": "single word response on the theme/ topic of the post",
                "reasoning": "A short summary of your reasoning",
            }
        )
        schema_model = (
            self._feature_extractor.llm.generate_pydantic_model_from_json_schema(
                "Default", response_schema
            )
        )
        structured_prompt = self._feature_extractor.llm.generate_json_prompt(
            schema_model, prompt
        )
        response = self._feature_extractor.llm.ask_question(structured_prompt)
        self._feature_extractor.llm._unload_model()
        self._feature_extractor.llm.reset_dialogue()
        gc.collect()
        cuda.Context.pop()

        self._theme = response["theme"]
        return response["theme"]

    ##################

    def get_dict(self):
        return {
            "username": self.username,
            "post_id": self.post_id,
            "replying": self.replying,
            "time": self.time,
            "post": self.post,
            "topic": self.topic,
            "words_per_sentence": str(self.words_per_sentence),
            "syntactical_complexity": str(self.syntactical_complexity),
            "lexical_diversity": str(self.lexical_diversity),
            "readability": str(self.readability),
            "toxicity": str(self.toxicity),
            "sentiment": str(self.sentiment),
            "embeddings": self.embeddings.tolist() if isinstance(self.embeddings, np.ndarray) else self.embeddings,
            "entities": self.entities,
            "keywords": self.keywords,
            "capital_letter_word_frequency": str(self.capital_letter_word_frequency),
            "pos_counts": self.pos_counts,
            "n_grams": self.n_grams,
            "emotion_scores": self.emotion_scores,
            "hate_speech_lexicon_counts": self.hate_speech_lexicon_counts,
            "tf_idf_vector": self.tf_idf_vector.toarray().tolist() if hasattr(self.tf_idf_vector, 'toarray') else self.tf_idf_vector,
            "text_vector": self.text_vector.tolist() if isinstance(self.text_vector, np.ndarray) else self.text_vector,
            "operational": str(self.operational),
            "theme": str(self.theme),
        }
