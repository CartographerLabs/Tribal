from data_set_managers.dataset_window_values import DatasetWindowValues
from objects.post import PostObject
from objects.window import WindowObject

class TimelineObject():
    _posts = []
    _windows = []

    _feature_extractor = None

    def __init__(self, feature_extractor) -> None:
        self._feature_extractor = feature_extractor

    @property
    def posts(self):    
        return self._posts

    @posts.setter
    def posts(self, list_of_posts: list[DatasetWindowValues]):
        post_objects = []
        
        # Sort the list of posts by time
        sorted_posts = sorted(list_of_posts, key=lambda post: post.time)
        
        for post in sorted_posts:
            post_object = PostObject(self._feature_extractor)
            post_object.post = post.post
            post_object.username = post.username
            post_object.time = post.time
            post_object.post_id = post.post_id
            post_object.replying = post.replying

            post_objects.append(post_object)

        self._posts = post_objects

    @property
    def windows(self):
        return self._windows
    
    @windows.setter
    def windows(self, value):
        self._windows = value

    def clear_windows(self):
        self.windows = []

    def get_current_start_and_end_dates(self):
        return min(post.time for post in self.posts), max(post.time for post in self.posts)

    def make_new_window(self, start_time, end_time, window_name=None):
        filtered_posts = [post for post in self._posts if start_time <= post.time <= end_time]
        window = WindowObject(self._feature_extractor, filtered_posts, start_time, end_time, window_name)
        self._windows.append(window)
        return window

    def update_windows(self):
        old_windows = self.windows
        self.windows = []
        for window in old_windows:
            window_name = window.window_name
            start_date = window.start_date
            end_date = window.end_date

            filtered_posts = [post for time, post in self._posts if start_date <= time <= end_date]

            self.windows.append(WindowObject(self._feature_extractor, filtered_posts, window_name))


    def add_user(self):
        pass
        #TODO - ensure to update posts, and update windows

    def remove_user(self):
        pass
        #TODO - ensure to update posts, and update windows