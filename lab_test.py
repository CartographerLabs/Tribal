from tribal.lab.posts.PostFactory import PostFactory
from tribal.lab.posts.Post import Post
from tribal.lab.posts.SlidingWindowConversationFactory import SlidingWindowConversationFactory
from tribal.lab.extractors import *
from rich.progress import Progress
from easyLLM.easyLLM import EasyLLM
import random 
import json 
import pickle

llm = EasyLLM(model_name="unsloth/Llama-3.1-Storm-8B")

if __name__ == "__main__":

    extractors = [
        CentralityFeatureExtractor(),
        PolarizingWordFrequencyExtractor(),
        EngagementFeatureExtractor(llm),
        RecruitmentFeatureExtractor(llm),
        RoleFeatureExtractor(llm),
        ExtremismFeatureExtractor(llm),
        ThemeFeatureExtractor(llm),
        OperationalFeatureExtractor(llm),
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

    random.shuffle(extractors)

    data = {}
    with open(r"C:\Users\JS\Downloads\mini-tribe\content\output.json", 'r') as file:
        data = json.load(file)

    # Instantiate the conversation factory
    conversation_factory = SlidingWindowConversationFactory(data)

    # Generate conversations with a sliding window of size 10
    conversations = conversation_factory.generate_conversations(window_size=10)

    with Progress() as progress:
        for conversation in conversations:

            # Use Rich to show progress as we apply each extractor
            task = progress.add_task(f"Extracting features for convo {conversations.index(conversation)} of {len(conversations)}...", total=len(extractors))
            for extractor in extractors:
                extractor.extract_features(conversation)
                progress.advance(task)

    for conversation in conversations:
        for post in conversation:
            for extractor in extractors:
                print(f"{extractor.property_name}: {post.get_property(extractor.property_name)}")
            print("\n")

    with open('conversations_with_features.pkl', 'wb') as f:
        pickle.dump(conversations, f)