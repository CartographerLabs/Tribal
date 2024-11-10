from sympy import use
from Tribal.objects.user import UserObject
from Tribal.utils.random_word import generate_random_name
from Tribal.utils.window_feature_extractor import WindowFeatureExtractor
from Tribal.objects.conversation import ConversationObject
import json
import torch
import gc


class WindowObject:

    _window_name = None
    _posts = None
    _users = None
    _conversations = None
    _start_date = None
    _end_date = None

    _extremism = None
    _centrality = None
    _roles = None
    _graph = None

    _feature_extractor = None

    def __init__(
        self, feature_extractor, posts, start_date, end_date, window_name=None
    ):
        if window_name is None:
            window_name = f"{generate_random_name()}-Tribe"

        self._feature_extractor = feature_extractor

        self.window_name = window_name
        self.start_date = start_date
        self.end_date = end_date

        self._initialise_window_data(posts)

    def _initialise_window_data(self, posts):
        self.posts = posts

        # Filter posts within the time range
        unique_usernames = list(set([data.username for data in posts]))
        list_of_user_objects = []

        self._window_feature_extractor = WindowFeatureExtractor(self.posts)
        self._graph = self._window_feature_extractor.graph
        self._centralities = self._window_feature_extractor.centralities
        self._roles = self._window_feature_extractor.roles
        self._extremism = self._window_feature_extractor.extremism

        for username in unique_usernames:
            user = UserObject(self._feature_extractor)
            users_posts = [post for post in posts if post.username == username]
            user.posts = users_posts
            user.username = username
            user.centrality = self._get_centrality_for_user(username)

            list_of_user_objects.append(user)

        self.users = list_of_user_objects

        self._chop_window_into_conversations()

    @property
    def window_name(self):
        return self._window_name

    @window_name.setter
    def window_name(self, name):
        self._window_name = name

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, value):
        self._start_date = value

    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, value):
        self._end_date = value

    @property
    def posts(self):
        return self._posts

    @posts.setter
    def posts(self, posts):
        if isinstance(posts, list):
            self._posts = posts
        else:
            raise ValueError("Posts should be a list")

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, users):
        if isinstance(users, list):
            self._users = users
        else:
            raise ValueError("Users should be a list")

    @property
    def conversations(self):
        return self._conversations

    @conversations.setter
    def conversations(self, value):
        self._conversations = value

    def _get_centrality_for_user(self, user):
        return self._centralities[user]

    @conversations.setter
    def conversations(self, conversations):
        if isinstance(conversations, list):
            self._conversations = conversations
        else:
            raise ValueError("Conversations should be a list")

    def _is_time_difference_within(self, date1, date2, time_threshold=10):
        time_diff = date2 - date1
        time_threshold_seconds = time_threshold * 60
        return 0 < time_diff.total_seconds() <= time_threshold_seconds

    def _chop_window_into_conversations(self):
        conversations = []

        for post_object in self.posts:
            username = post_object.username
            post_message = post_object.post
            replying = post_object.replying
            time = post_object.time

            added_to_convo = False
            if len(conversations) == 0:
                conversations.append([post_object])
                added_to_convo = True
            else:
                reply_to = replying
                for conversation in conversations:
                    for conversation_post in conversation:
                        conversation_post_id = conversation_post.post_id
                        if conversation_post_id == reply_to:
                            conversation.append(post_object)
                            added_to_convo = True

            if not added_to_convo:
                latest_convo = conversations[-1]
                latest_message = latest_convo[-1]
                latest_message_post_time = latest_message.time

                is_same_convo = self._is_time_difference_within(
                    latest_message_post_time, time
                )
                if is_same_convo:
                    conversations[-1].append(post_object)
                else:
                    conversations.append([post_object])

        self.conversations = []
        for convo_thread_posts in conversations:
            self.conversations.append(
                ConversationObject(self._feature_extractor, convo_thread_posts)
            )

    def get_dict(self):
        data_dict = {
            "window_name": self.window_name,
            "users": [
                {
                    "username": user.username,
                    "centrality": user.centrality,
                    "role": user.role,
                    "extremism": user.extremism,
                    "avrg_sentiment": user.avrg_sentiment,
                    "avrg_toxicity": user.avrg_toxicity,
                }
                for user in self.users
            ],
            "conversations": [obj.get_dict() for obj in self.conversations],
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self._end_date.strftime("%Y-%m-%d"),
        }

        return data_dict
