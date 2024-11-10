from turtle import pos

from sympy import use
from Tribal.objects import user
import networkx as nx
import re
from datetime import datetime
import json
import gc
import torch

from Tribal.objects.user import UserObject
from Tribal.utils.window_feature_extractor import WindowFeatureExtractor


class ConversationObject:

    _posts = None
    _graph = None
    _users = None
    _feature_extractor = None
    _window_feature_extractor = None
    _centralities = None
    _roles = None
    _extremism = None
    _start_time = None
    _end_time = None

    def __init__(self, feature_extractor, posts) -> None:
        self._posts = posts
        self._feature_extractor = feature_extractor
        self._initialise()

    def _initialise(self):
        self._window_feature_extractor = WindowFeatureExtractor(self.posts)

        self._centralities = self._window_feature_extractor.centralities
        self._roles = self._window_feature_extractor.roles
        self._extremism = self._window_feature_extractor.extremism

        self._start_time, self._end_time = min(post.time for post in self.posts), max(
            post.time for post in self.posts
        )

        self._update_users()
        self._graph = self._window_feature_extractor.graph

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, value):
        self._graph = value

    @property
    def posts(self):
        return self._posts

    @posts.setter
    def posts(self, value):
        self._posts = value

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, value):
        self._users = value

    def add_post(self, post, index=None):

        if index == None:
            self._posts.apend(post)
        else:
            self._posts.insert(index, post)

    def _update_users(self):
        self.users = []
        users_dict = {}
        for post in self.posts:
            username = post.username
            if username in users_dict:
                users_dict[username].append(post)
            else:
                users_dict[username] = [post]

        users_list = []
        for user in users_dict:
            posts = users_dict[user]
            user_object = UserObject(self._feature_extractor)
            user_object.posts = posts
            user_object.username = user
            user_object.centrality = self._get_centrality_for_user(user)
            user_object.role = self._get_role_for_user(user)
            user_object.extremism = self._get_extremism_for_user(user)
            users_list.append(user_object)

        self.users = users_list

    def _get_role_for_user(self, user):
        prompt = f"""For user: '{user}' identify their role in the conversation. You are an expert in social media analysis amongst user roles on social media. Given the following block of social media content, your task is to define the (most apparent) role of each user based on the following categories/definitions.
 
If a user is identified as performing behaviour that falls into multiple roles, then, in order, the following roles take priority:
People Leader: Directs, recruits and mobilises block members either virtually or in the real world.
Leader Influencer: Directs the conversation as a knowledge source and/or gatekeeper, with other members reflecting their attitudes.
Engager Negator: Negative or berating interactions in an attempt to reduce discussion or offer a counter argument against a fundamental principle groups agreement.
Engager Supporter: Positively interacts with the topic, encouraging or promoting future further discussion and ideological success.
Engager Neutral: Neutral topic interaction to learn or socially interact with members.
Bystander: Does not engage with the main discussion but remains within the discussion block.
NATTC: The user is at the beginning or end of a block (outside of the time boxed discussion) discussing a topic that the content is not engaged with.

Posts:
"""

        for post in self._posts:
            prompt = prompt + "\n" + post.username + " : " + post.post

        schema = json.dumps(
            {
                f"role": "The role of 'People Leader', 'Leader Influencer', 'Engager Negator', 'Engager Supporter', 'Engager Neutral', 'Bystander', or 'NATTAC' for user {user}.",
                "rational": "the rational for why you have made this decision.",
            }
        )
        schema_model = (
            self._feature_extractor.llm.generate_pydantic_model_from_json_schema(
                "Default", schema
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
            role = response["role"]
        except KeyError as e:
            return self._get_role_for_user(user)

        return response["role"]

    def _get_extremism_for_user(self, user):
        prompt = f"""For user: '{user}' identify if their posts are extremist or non extremist. You are an expert in social media analysis for extremist content on social media. To follow is a block of social media text containing users and their related text. Based on the following definition please identify which users in the conversation are classified as extremist based on their content and provide your reasoning:
Extremism is the promotion or advancement of an ideology based on violence, hatred or intolerance, that aims to:
1.  negate or destroy the fundamental rights and freedoms of others; or
2.  undermine, overturn or replace the UK’s system of liberal parliamentary democracy and democratic rights; or
3.  intentionally create a permissive environment for others to achieve the results in (1) or (2).

Posts:
"""

        for post in self._posts:
            prompt = prompt + "\n" + post.username + " : " + post.post

        schema = json.dumps(
            {
                f"is_extremist": "A boolean ('true' or 'false') on if user {user}'s posts are extremist",
                "rational": "the rational for why you have made this decision.",
            }
        )
        schema_model = (
            self._feature_extractor.llm.generate_pydantic_model_from_json_schema(
                "Default", schema
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
            extremism = response["is_extremist"]
        except KeyError as e:
            return self._get_extremism_for_user(user)

        return response["is_extremist"]

    def _get_centrality_for_user(self, user):
        return self._centralities[user]

    def get_dict(self):
        data_dict = {
            "posts": [data.get_dict() for data in self.posts],
            "graph": f"{self.graph.nodes()}-{self.graph.edges()}",
            "users": [data.get_dict() for data in self.users],
            "start_time": self._start_time.strftime("%Y-%m-%d"),
            "end_time": self._end_time.strftime("%Y-%m-%d"),
        }

        return data_dict
