from abc import ABC, abstractmethod
import json
import numpy as np
import gc

# Assuming FeatureExtractor is imported correctly
from Tribal.utils.feature_extractor import FeatureExtractor
from Tribal.utils.config_manager import ConfigManager
import torch


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
            self._capital_letter_word_frequency = (
                self._feature_extractor.get_capital_letter_word_frequency(self.post)
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
            self._hate_speech_lexicon_counts = (
                self._feature_extractor.get_hate_speech_lexicon_counts(self.post)
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

        prompt = """
        As an AI model, you are tasked with analyzing social media posts to detect and classify online behaviors associated with operational planning for extremist mobilization. Your primary goal is to identify whether the post contains signals that suggest a likelihood of transitioning from radicalized beliefs to actionable planning. Based on the framework from “Online Signals of Extremist Mobilization”, evaluate the post and address the following:
        
        1. **Language of Violent Action**: Does the post contain keywords or themes associated with preparing for violence? Identify terms indicating intent to harm, such as "weapons," "target," "attack," or "security" that may align with operational planning rather than ideology alone.
           - *Example*: "Planning to gather supplies for the mission; anyone have resources for building what we need?"
           - *Example*: "I've got the target location mapped out; just need a few more volunteers."
        
        2. **Knowledge Acquisition and Capability**: Are there signs that the user is acquiring skills or resources necessary for violent actions? Look for indicators of users seeking knowledge on tactics, weapon acquisition, operational security, or techniques for evading law enforcement—aligning with capabilities that support mobilization.
           - *Example*: "Anyone have experience with encrypted communication methods? Need to secure our plans."
           - *Example*: "Looking for a reliable source for tactical gear and training manuals."
        
        3. **Logistics and Coordination**: Does the content reference logistical planning, including specifics about timing, locations, targets, or collaborative efforts? This may include discussions about coordinating with others, arranging meetups, gathering supplies, or planning routes, which indicate preparation.
           - *Example*: "We’ll meet at the south entrance at 8 PM—don’t bring phones or anything traceable."
           - *Example*: "Got the route mapped out; sharing the link in a private group."
        
        4. **“Leakage” of Intentions**: Identify any unintended or indirect revelations of the user's plans or intentions. This may include expressions that hint at readiness for action, capability boasts, or discussing future steps in a way that goes beyond expressing ideology.
           - *Example*: "Soon, they’ll see what we’re capable of—no more waiting around."
           - *Example*: "We’ve been prepping for months, and everything is set for the day."
        
        5. **Tone and Intensity Indicators**: Assess the post for heightened emotional cues, such as excessive punctuation or expressions of urgency, anger, or excitement that may signal increased readiness or psychological escalation toward action.
           - *Example*: "This is it!! No more talk—time for action!!!"
           - *Example*: "We have no time left, everyone needs to be ready NOW."
        
        **Final Classification**: Based on the criteria above, classify the post as exhibiting “none,” “weak,” “moderate,” “strong,” or “extreme” indicators of mobilization for extremist operational planning. Support your assessment with examples from the post, referencing specific criteria from “Online Signals of Extremist Mobilization”.
        
        **Key Considerations**:
        - **Contextual Analysis**: Evaluate the language and behaviors within the broader context of the user’s online history and affiliations.
        - **Minimize Bias**: Avoid basing decisions solely on ideology. Focus on behaviors associated with mobilization processes, as defined in the study.
        - **Ethical Awareness**: Recognize that false positives can have serious impacts; use this classification to support, not replace, investigative methods.
        """

        
        prompt = prompt + "\n Post: " + self.post

        response_schema = json.dumps(
            {
                "is_operational_planning": "One-word response: 'none', 'weak', 'moderate', 'strong', or 'extreme'.",
                "reasoning": "Detailed explanation of the post's classification, with relevant examples and references to mobilization indicators",
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

        print(structured_prompt)
        response = self._feature_extractor.llm.ask_question(structured_prompt)
        self._feature_extractor.llm._unload_model()
        self._feature_extractor.llm.reset_dialogue()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            is_op = response["is_operational_planning"]
            self._operational = is_op
        except KeyError as e:
            return self.operational()
            
        return response["is_operational_planning"]

    @property
    def theme(self):

        if self._theme:
            return self._theme

        prompt = """
        You are given a social media post. Identify the primary theme or topic of the post in one word. Your response should focus on the most prominent subject or sentiment conveyed, capturing the main idea in broad terms.
        
        When selecting the topic, avoid specific details or minor points. Aim for general categories that summarize the post's main content, such as 'Travel,' 'Health,' 'Food,' 'Technology,' etc.
        
        Examples:
        
        Post: 'Excited to visit the beach this weekend and enjoy some sunshine!'
        Response: 'Travel'
        
        Post: 'Just had the most amazing pasta dinner at my favorite restaurant!'
        Response: 'Food'
        
        Post: 'Feeling grateful for my family and friends today.'
        Response: 'Gratitude'
        
        Post: 'Just finished my workout, feeling strong and ready to tackle the day!'
        Response: 'Fitness'
        
        Post: 'Can’t believe the prices at the grocery store lately, everything is so expensive!'
        Response: 'Economy'
        
        Post: 'Watching the latest Marvel movie, the effects are mind-blowing!'
        Response: 'Entertainment'
        
        Focus on selecting a single word that best captures the main theme of the post.
        """

        prompt = prompt + "\n Post: " + self.post

        response_schema = json.dumps(
            {
                "theme": "One-word response on the theme/ topic of the post",
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

        print(structured_prompt)
        response = self._feature_extractor.llm.ask_question(structured_prompt)
        self._feature_extractor.llm._unload_model()
        self._feature_extractor.llm.reset_dialogue()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            extracted_theme = response["theme"]
            self._theme = extracted_theme
        except KeyError as e:
            return self.theme()

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
            "embeddings": (
                self.embeddings.tolist()
                if isinstance(self.embeddings, np.ndarray)
                else self.embeddings
            ),
            "entities": self.entities,
            "keywords": self.keywords,
            "capital_letter_word_frequency": str(self.capital_letter_word_frequency),
            "pos_counts": self.pos_counts,
            "n_grams": self.n_grams,
            "emotion_scores": self.emotion_scores,
            "hate_speech_lexicon_counts": self.hate_speech_lexicon_counts,
            "tf_idf_vector": (
                self.tf_idf_vector.toarray().tolist()
                if hasattr(self.tf_idf_vector, "toarray")
                else self.tf_idf_vector
            ),
            "text_vector": (
                self.text_vector.tolist()
                if isinstance(self.text_vector, np.ndarray)
                else self.text_vector
            ),
            "operational": str(self.operational),
            "theme": str(self.theme),
        }
