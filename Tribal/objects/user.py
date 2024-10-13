from abc import ABC
import numpy as np
from collections import Counter, defaultdict

# Assuming PostObject and DatasetWindowValues are imported correctly
from Tribal.objects.post import PostObject
from Tribal.data_set_managers.dataset_window_values import DatasetWindowValues


class UserObject(ABC):
    _posts = []
    _username = None
    _centrality = None
    _role = None
    _extremism = None
    _feature_extractor = None

    # Aggregated features
    _avrg_toxicity = None
    _avrg_sentiment = None
    _avrg_is_operational = None
    _avrg_theme = None
    _avrg_capital_letter_word_frequency = None
    _avrg_pos_counts = None
    _most_common_n_grams = None
    _avrg_emotion_scores = None
    _average_hate_speech_lexicon_counts = None  # Updated variable name
    _avrg_text_vector = None

    def __init__(self, feature_extractor) -> None:
        super().__init__()
        self._feature_extractor = feature_extractor

    @property
    def posts(self):
        return self._posts

    @posts.setter
    def posts(self, list_of_posts: list[DatasetWindowValues]):
        list_of_post_objects = []

        sentiment_scores = []
        toxicity_scores = []
        is_operational_scores = []
        theme_values = []
        capital_letter_freqs = []
        pos_counts_list = []
        n_grams_list = []
        emotion_scores_list = []
        hate_speech_counts_list = []
        text_vectors = []

        # Initialize counters for POS tags, emotion scores, and hate speech terms
        total_pos_counts = defaultdict(int)
        total_emotion_scores = defaultdict(float)
        total_hate_speech_counts = defaultdict(float)

        num_posts = len(list_of_posts)  # Number of posts

        for post in list_of_posts:
            post_object = PostObject(self._feature_extractor)
            post_object.post = post.post
            post_object.username = post.username
            post_object.time = post.time
            post_object.post_id = post.post_id
            post_object.replying = post.replying

            list_of_post_objects.append(post_object)

            # Collect features
            toxicity_scores.append(post_object.toxicity["toxicity"])
            sentiment_scores.append(post_object.sentiment["compound"])
            is_operational_scores.append(post_object.operational)
            theme_values.append(post_object.theme)
            capital_letter_freqs.append(post_object.capital_letter_word_frequency)
            pos_counts_list.append(post_object.pos_counts)
            n_grams_list.extend(post_object.n_grams)
            emotion_scores_list.append(post_object.emotion_scores)
            hate_speech_counts_list.append(post_object.hate_speech_lexicon_counts)
            text_vectors.append(post_object.text_vector)

        self._posts = list_of_post_objects

        # Assign average toxicity and sentiment to the user
        self._avrg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        self._avrg_toxicity = np.mean(toxicity_scores) if toxicity_scores else 0.0
        self._avrg_capital_letter_word_frequency = (
            np.mean(capital_letter_freqs) if capital_letter_freqs else 0.0
        )
        self._avrg_text_vector = (
            np.mean(text_vectors, axis=0) if text_vectors else np.zeros(self._feature_extractor.VECTOR_SIZE)
        )

        # Aggregate is_operational_scores and theme_values
        self._avrg_is_operational = (
            max(is_operational_scores, key=is_operational_scores.count)
            if is_operational_scores else None
        )
        self._avrg_theme = (
            max(theme_values, key=theme_values.count)
            if theme_values else None
        )

        # Aggregate POS counts
        for pos_counts in pos_counts_list:
            for pos_tag, count in pos_counts.items():
                total_pos_counts[pos_tag] += count
        self._avrg_pos_counts = dict(total_pos_counts)

        # Find most common n-grams
        n_gram_counter = Counter(n_grams_list)
        self._most_common_n_grams = n_gram_counter.most_common(10)

        # Aggregate emotion scores
        for emotion_scores in emotion_scores_list:
            for emotion, score in emotion_scores.items():
                total_emotion_scores[emotion] += score
        if num_posts > 0:
            self._avrg_emotion_scores = {
                emotion: total_emotion_scores[emotion] / num_posts
                for emotion in total_emotion_scores
            }
        else:
            self._avrg_emotion_scores = {}

        # Aggregate hate speech lexicon counts and compute average per post
        for hate_speech_counts in hate_speech_counts_list:
            for term, count in hate_speech_counts.items():
                total_hate_speech_counts[term] += count
        if num_posts > 0:
            self._average_hate_speech_lexicon_counts = {
                term: count / num_posts for term, count in total_hate_speech_counts.items()
            }
        else:
            self._average_hate_speech_lexicon_counts = {}

    @property
    def centrality(self):
        return self._centrality

    @centrality.setter
    def centrality(self, value):
        self._centrality = value

    @property
    def role(self):
        return self._role

    @role.setter
    def role(self, value):
        self._role = value

    @property
    def extremism(self):
        return self._extremism

    @extremism.setter
    def extremism(self, value):
        self._extremism = value

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, new_username):
        self._username = new_username

    @property
    def avrg_sentiment(self):
        return self._avrg_sentiment

    @property
    def avrg_toxicity(self):
        return self._avrg_toxicity

    @property
    def avrg_is_operational(self):
        return self._avrg_is_operational

    @property
    def avrg_theme(self):
        return self._avrg_theme

    @property
    def avrg_capital_letter_word_frequency(self):
        return self._avrg_capital_letter_word_frequency

    @property
    def avrg_pos_counts(self):
        return self._avrg_pos_counts

    @property
    def most_common_n_grams(self):
        return self._most_common_n_grams

    @property
    def avrg_emotion_scores(self):
        return self._avrg_emotion_scores

    @property
    def average_hate_speech_lexicon_counts(self):
        return self._average_hate_speech_lexicon_counts

    @property
    def avrg_text_vector(self):
        return self._avrg_text_vector

    ##################

    def get_dict(self, show_simple=False):
        if show_simple:
            return [str(self.centrality), str(self.role), str(self.extremism)]
        else:
            return {
                "username": self.username,
                "centrality": self.centrality,
                "role": self.role,
                "extremism": self.extremism,
                "avrg_sentiment": self.avrg_sentiment,
                "avrg_toxicity": self.avrg_toxicity,
                "avrg_is_operational": self.avrg_is_operational,
                "avrg_theme": self.avrg_theme,
                "avrg_capital_letter_word_frequency": self.avrg_capital_letter_word_frequency,
                "avrg_pos_counts": self.avrg_pos_counts,
                "most_common_n_grams": self.most_common_n_grams,
                "avrg_emotion_scores": self.avrg_emotion_scores,
                "average_hate_speech_lexicon_counts": self.average_hate_speech_lexicon_counts,
                "avrg_text_vector": self.avrg_text_vector.tolist()
                if isinstance(self.avrg_text_vector, np.ndarray)
                else self.avrg_text_vector,
                "posts": [data.get_dict() for data in self._posts],
            }
