from abc import abstractmethod
from datetime import datetime

class DatasetWindowValues():

    _time = None
    _post = None
    _username = None
    _replying = None
    _post_id = None

    def __init__(self, time, post, username, replying, post_id) -> None:
        self.time = time
        self.post = post
        self.username = username
        self.replying = replying
        self.post_id = post_id

    @property
    def time(self):
        return self._time
    
    @property
    def post(self):
        return self._post
    
    @property
    def username(self):
        return self._username
    
    @property
    def replying(self):
        return self._replying
    
    @property
    def post_id(self):
        return self._post_id

    ######################

    @post_id.setter
    def post_id(self, value):
        self._post_id = str(value)

    @time.setter
    def time(self, value):
        if isinstance(value, str):
            # Try to parse the string into a datetime object
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
                try:
                    dt = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
            else:
                # If no format worked, raise an exception
                raise ValueError("Unrecognized string format")
        
        elif isinstance(value, (int, float)):
            # Assume it's an epoch timestamp
            dt = datetime.fromtimestamp(value)
        
        elif isinstance(value, datetime):
            dt = value
        
        else:
            raise TypeError("Unsupported type for time")
        
        # Convert to the desired format
        self._time = dt

    @post.setter
    def post(self, value):
        self._post = str(value)

    @username.setter
    def username(self, value):
        self._username = str(value)

    @replying.setter
    def replying(self, value):
        self._replying = str(value)