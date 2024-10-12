from data_set_managers.abstract_dataset_manager import AbstractDatasetManager
from data_set_managers.dataset_window_values import DatasetWindowValues
import json
from datetime import datetime

class JsonDatasetManager(AbstractDatasetManager):

    _path_to_json_file = None
    _time_key = None
    _post_key = None
    _username_key = None
    _replying_key = None
    _post_id_key = None

    _all_data = None
    _list_of_posts = None

    def __init__(self, path_to_json_file, time_key="time", post_key="post", username_key="username", post_id_key="post_id", replying_key="replying_to") -> None:
        super().__init__()

        self._path_to_json_file = path_to_json_file
        self._time_key = time_key
        self._post_key = post_key
        self._username_key = username_key
        self._replying_key = replying_key
        self._post_id_key = post_id_key

        # Load and cache data during initialization
        self._all_data = self._load_and_process_data()

    def _read_all_contents_from_json_file(self):
        try:
            with open(self._path_to_json_file, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            raise Exception(f"Error: The file at {self._path_to_json_file} was not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error: The file at {self._path_to_json_file} is not a valid JSON file.")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

    def _load_and_process_data(self):
        json_data = self._read_all_contents_from_json_file()
        post_objects = []
        for post in json_data:
            time = post[self._time_key]
            post_content = post[self._post_key]
            username = post[self._username_key]
            post_id = post[self._post_id_key]
            if self._replying_key in post:
                replying = post[self._replying_key]
            else:
                replying = "None"

            value = DatasetWindowValues(time, post_content, username, replying, post_id)
            post_objects.append(value)
        return post_objects

    def get_all_data_within_time_window(self, start_time, end_time):

        if start_time is not None:
            start_dt = start_time
        else:
            start_dt = datetime.min

        if end_time is not None:
            end_dt = end_time
        else:
            end_dt = datetime.max

        filtered_data = [data for data in self._all_data if start_dt <= data.time <= end_dt]
        return filtered_data
        
    def get_all_user_data(self, start_time=None, end_time=None):
        return self.get_all_data_within_time_window(start_time, end_time)

    def get_all_data_for_user(self, username=None, start_time=None, end_time=None):       
        if start_time is not None:
            start_dt = start_time
        else:
            start_dt = datetime.min

        if end_time is not None:
            end_dt = end_time
        else:
            end_dt = datetime.max

        filtered_data = [data for data in self._all_data
                        if (username is None or data.username == username) and start_dt <= data.time <= end_dt]
        return filtered_data

    def get_list_of_posts(self, username = None, start_time=None, end_time=None):       

        if username != None:
            filtered_data = self.get_all_data_for_user(username, start_time, end_time)
        else:
            filtered_data = self.get_all_data_within_time_window(start_time, end_time)
        posts = [data.post for data in filtered_data]
        return posts
    
    def get_all_users(self, start_time=None, end_time=None):
        if start_time is not None:
            start_dt = start_time
        else:
            start_dt = datetime.min

        if end_time is not None:
            end_dt = end_time
        else:
            end_dt = datetime.max

        filtered_data = [data for data in self._all_data if start_dt <= data.time <= end_dt]
        users = {data.username for data in filtered_data}
        return list(users)

    def get_timeframe(self):
        if not self._all_data:
            return None, None

        # Extract times from dataset
        times = [data.time for data in self._all_data]

        start_time = min(times)
        end_time = max(times)

        return start_time, end_time