from tribal.forge.base_nodes import MESSAGE_FORMAT, BaseProcessorNode
from tribal.lab.extractors import *
from tribal.lab.posts.Post import Post
import time
import random

class FeatureExtractorNode(BaseProcessorNode):
    def __init__(self, broadcast_manager, llm=None, sleep=10):

        super().__init__("Feature Extractor", broadcast_manager)
        self.extractors = [
            CentralityFeatureExtractor(),
            PolarizingWordFrequencyExtractor(),
            SentimentFeatureExtractor(),
            CapitalLetterWordFrequencyExtractor(),
            POSCountsExtractor(),
            NGramsExtractor(),
            EmotionScoresExtractor(),
            HateSpeechLexiconCountsExtractor(),
            WordsPerSentenceExtractor(),
            LexicalDiversityExtractor(),
            ToxicityFeatureExtractor(),
            EntitiesExtractor(),
            KeywordsExtractor(),
            PositiveWordFrequencyExtractor(),
            NegativeWordFrequencyExtractor(),
            ViolenceRelatedWordFrequencyExtractor(),
            ReligiousWordFrequencyExtractor()
        ]
        
        if llm:
            self.extractors += [
                EngagementFeatureExtractor(llm),
                RecruitmentFeatureExtractor(llm),
                RoleFeatureExtractor(llm),
                ExtremismFeatureExtractor(llm),
                ThemeFeatureExtractor(llm),
                OperationalFeatureExtractor(llm)
            ]

        random.shuffle(self.extractors)
        self.post_cache = []
        self.cache_limit = 10
        self.sleep_between_send = float(sleep)

    def _process_broadcast(self, act_message):
        message = act_message[MESSAGE_FORMAT["MESSAGE"]]
        post_content = message["post"]
        username = message["username"]
        msg_time = message["time"]
        replying_to = message["replying_to"]
        post_id = message["post_id"]

        post = Post(post_content, username, msg_time, replying_to, post_id)
        self.post_cache.append(post)

        if len(self.post_cache) < self.cache_limit:
            return

        posts = self.post_cache
        self.post_cache = []
        for extractor in self.extractors:
            extractor.extract_features(posts)

        for post in posts:
            message = {"username": post.username, "post":post.text,"time": post.time, "replying_to": post.replying, "post_id": post.id}

            for extractor in self.extractors:

                message[extractor.property_name] = post.get_property(extractor.property_name)
            message = self._construct_message(self.name, message)
            time.sleep(self.sleep_between_send)
            self.send_broadcast(message, self.name)
