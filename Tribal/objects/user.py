from abc import ABC
from objects.post import PostObject
from data_set_managers.dataset_window_values import DatasetWindowValues
import numpy as np


class UserObject(ABC):  
    _posts = []
    _username = None
    _centrality = None
    _role = None
    _extremism = None
    _feature_extractor = None
    _avrg_toxicity = None
    _avrg_sentiment = None
    _avrg_is_operational = None
    _avrg_theme = None

    def __init__(self, feature_extractor) -> None:
        super().__init__()
        self._feature_extractor = feature_extractor

    @property
    def posts(self):    
        return self._posts

    @posts.setter
    def posts(self, list_of_posts:list[DatasetWindowValues]):
        list_of_post_objects = []

        sentiment_scores = []
        toxicity_scores = []
        is_operational_scores = []
        theme_values = []

        for post in list_of_posts:
            post_object = PostObject(self._feature_extractor)
            post_object.post = post.post
            post_object.username = post.username
            post_object.time = post.time
            post_object.post_id = post.post_id
            post_object.replying = post.replying

            
            list_of_post_objects.append(post_object)

            toxicity_scores.append(post_object.toxicity["toxicity"])
            sentiment_scores.append(post_object.sentiment["compound"])
            is_operational_scores.append(post_object.operational)
            theme_values.append(post_object.theme)

        self._posts = list_of_post_objects

        # Assign average toxicity and sentiment to the user
        self._avrg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        self._avrg_toxicity = np.mean(toxicity_scores) if toxicity_scores else 0.0
        self.avrg_is_operational = max(is_operational_scores, key=is_operational_scores.count)
        self.avrg_theme = max(theme_values, key=theme_values.count)


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

    @avrg_sentiment.setter
    def avrg_sentiment(self, value):
        self._avrg_sentiment = value

    @property
    def avrg_toxicity(self):
        return self._avrg_toxicity
    
    @property
    def avrg_is_operational(self):
        return self._avrg_is_operational
    
    @avrg_is_operational.setter
    def avrg_is_operational(self, value):
        self._avrg_is_operational = value

    @property
    def avrg_theme(self):
        return self._avrg_theme

    @avrg_theme.setter
    def avrg_theme(self, value):
        self._avrg_theme = value

    @avrg_toxicity.setter
    def avrg_toxicity(self, value):
        self._avrg_toxicity = value

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
                "avrg_is_operational":self.avrg_is_operational,
                "avrg_theme":self.avrg_theme,
                "posts": [data.get_dict() for data in self._posts]
            }
