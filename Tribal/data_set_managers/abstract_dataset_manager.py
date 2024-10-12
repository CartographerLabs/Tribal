# an abstract class for extracting data that is extended for data types (e.g. databases, json, etc)
from abc import ABC, abstractmethod
from utils.random_word import generate_random_name

class AbstractDatasetManager(ABC):
    @abstractmethod
    def get_all_user_data(self, start_time=None, end_time=None):
        raise NotImplementedError("get_all_user_data is not implemengted in the abstract class")

    @abstractmethod
    def get_all_data_for_user(self, start_time=None, end_time=None):
        raise NotImplementedError("get_all_data_for_user is not implemengted in the abstract class")
    
    @abstractmethod
    def get_list_of_posts(self, start_time=None, end_time=None):
        raise NotImplementedError("get_list_of_posts is not implemengted in the abstract class")
    
    @abstractmethod
    def get_all_users(self, start_time=None, end_time=None):
        raise NotImplementedError("get_all_users is not implemengted in the abstract class")

    @abstractmethod
    def get_timeframe(self):
        raise NotImplementedError("get_timeframe is not implemengted in the abstract class")
    
    def obfuscate_usernames(self):

        if not self._all_data:
            raise Exception("Data must be set before names can be obfuscated!")

        known_names = {}

        for post in self._all_data:
            username = post.username
            if username in known_names:
                post.username = known_names[username]
            else:
                obfuscated_name = generate_random_name()

                while obfuscated_name in list(known_names.keys()):
                    obfuscated_name = generate_random_name()

                obfuscated_name = obfuscated_name.replace("-","")
                known_names[username] = obfuscated_name
                post.username = obfuscated_name


        for post in self._all_data:

            for username in known_names:
                obfuscated_name = known_names[username]
                post.post = post.post.replace(username, obfuscated_name)