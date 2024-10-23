import networkx as nx
import re 
import json 
from datetime import datetime

from easyLLM.easyLLM import EasyLLM

class WindowFeatureExtractor():

    posts = None
    graph = None
    centralities = None
    roles = None
    extremism = None

    def __init__(self, posts) -> None:
        self.posts = posts
        self._calculate_graph()
        self._calculate_centralities()
        self._calculate_extremism()
        self._calculate_roles()

    def _calculate_centralities(self):
        self.centralities = {}
        for node, score in nx.degree_centrality(self.graph).items():
            self.centralities[node] = score

    def _calculate_roles(self):
        #TODO
        return 
        question = r"""You are an expert in social media analysis amongst user roles on social media. Given the following block of social media content, your task is to define the (most apparent) role of each user based on the following categories/definitions.
 
If a user is identified as performing behaviour that falls into multiple roles, then, in order, the following roles take priority:
Leader: Directs, recruits and mobilises block members either virtually or in the real world. or directs the conversation as a knowledge source and/or gatekeeper, with other members reflecting their attitudes.
Engager: Negative or berating interactions in an attempt to reduce discussion or offer a counter argument against a fundamental principle groups agreement.Or positively interacts with the topic, encouraging or promoting future further discussion and ideological success. Or neutral topic interaction to learn or socially interact with members.
Bystander: Does not engage with the main discussion but remains within the discussion block. Or the user is at the beginning or end of a block (outside of the time boxed discussion) discussing a topic that the content is not engaged with.

In addition to the above, also follow the below instructions:
Your response should contain only one instance of each user id.
Use of @<user> refers to a user being ‘mentioned’ in the post.
If a post only contains a link and there is no information provided in the link URL, then it should be defined as ‘bystander’
If a user attempts to start a new conversation, but no-one responds, then treat the user as a ‘bystander’.
If the post is identified as sarcasm, and there is no evidence one way or another for the role, then treat it as ‘neutral’
Please assess the following content based on the ruleset provided and following the template structure:
 
Please format your response with the JSON structure for each user present in the discussion:
{
    'username': The relevant user's username.
    'label': The user's overall discussion label
    'reasoning': Your reasoning behind the chosen label
}
Each username should only appear once in the json list!
[Social Media Discussion]
"""+str({data.username: data.post for data in self.posts})+"""
[/Social Media Discussion]
[Answer]"""
        
        answere = EasyLLM(max_new_tokens=500).ask_question(question)

        roles = json.loads(answere)

        self.roles = roles


    def _calculate_extremism(self):
        #TODO
        return
        question = r"""
You are an expert in social media analysis for extremist content on social media. To follow is a block of social media text containing users and their related text. Based on the following definition please identify which users in the conversation are classified as extremist based on their content and provide your reasoning:
Extremism is the promotion or advancement of an ideology based on violence, hatred or intolerance, that aims to:
1.  negate or destroy the fundamental rights and freedoms of others; or
2.  undermine, overturn or replace the UK’s system of liberal parliamentary democracy and democratic rights; or
3.  intentionally create a permissive environment for others to achieve the results in (1) or (2).

In addition to the above, also follow the below instructions:
•   Use of @<user> refers to a user being ‘mentioned’ in the post.
•   If a post does not clearly fall into the above definition of extremism, then lean towards non-extremism.
•   A 'permissive environment' may be characterised as being tolerant of behaviour or practices strongly disapproved of by others, such as an environment where radicalising ideologies are permitted to flourish.
 
Please format your response with the JSON structure for each user present in the discussion:
{
'username': The relevant username.
'label': The user's overall discussion label
'reasoning': Your reasoning behind the chosen label
}

All usernames should be listed and each username should only appear once in the json list! It is possible for no users to express extremist content.
 
[Social Media Discussion]
"""+str({data.username: data.post for data in self.posts})+"""
[/Social Media Discussion]
[Answer]"""
        
        answere = EasyLLM(max_new_tokens=500).ask_question(question)

        extremism = json.loads(answere)

        self.extremism = extremism

    def _is_time_difference_within(self, date1, date2, time_threshold=10):

        time_diff = date2 - date1
        time_threshold_seconds = time_threshold * 60
        return 0 < time_diff.total_seconds() <= time_threshold_seconds

    def _calculate_graph(self):
        dict_of_post_ids = {data.post_id: data.username for data in self.posts}
        G = nx.MultiDiGraph()

        known_users = []
        for i, post in enumerate(self.posts):
            known_users.append(post.username)

        for i, post in enumerate(self.posts):
            username = post.username
            message = post.post
            post_time = post.time

            G.add_node(username, type="user")

            replying_to = post.replying
            if replying_to != 'None' and replying_to is not None:
                if replying_to in dict_of_post_ids:
                    replying_to_user = dict_of_post_ids[replying_to]
                    G.add_node(replying_to_user, type="user")
                    G.add_edge(username, replying_to_user, type="reply")

            mention_pattern = r'@([A-Za-z0-9_]+)'
            mentions = re.findall(mention_pattern, message)
            for mention in mentions:
                
                if mention in known_users:
                    G.add_node(mention, type="user")
                    G.add_edge(username, mention, type="mention")

            for j in range(i + 1, len(self.posts)):
                next_post = self.posts[j]
                next_post_user = next_post.username
                next_post_time = next_post.time
                if username != next_post_user and self._is_time_difference_within(post_time, next_post_time, 10):
                    
                    if next_post_user in known_users:
                        G.add_node(next_post_user, type="user")
                        G.add_edge(username, next_post_user, type="temporal")


        # Due to meentions this is making users who don't exist in the posts
        self.graph = G
