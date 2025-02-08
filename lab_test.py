from tribal.lab.posts.SlidingWindowConversationFactory import SlidingWindowConversationFactory
from tribal.lab.extractors import *
from rich.progress import Progress
import random 
import json 
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os


MODEL_NAME = "unsloth/Llama-3.1-Storm-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

if __name__ == "__main__":

    extractors = [
        CentralityFeatureExtractor(),
        PolarizingWordFrequencyExtractor(),
        EngagementFeatureExtractor(model, tokenizer),
        RecruitmentFeatureExtractor(model, tokenizer),
        RoleFeatureExtractor(model, tokenizer),
        ExtremismFeatureExtractor(model, tokenizer),
        ThemeFeatureExtractor(model, tokenizer),
        OperationalFeatureExtractor(model, tokenizer),
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
    with open(r"C:\Users\JS\Downloads\output.json", 'r') as file:
        data = json.load(file)

    # Instantiate the conversation factory
    conversation_factory = SlidingWindowConversationFactory(data)

    # Generate conversations with a sliding window of size 10
    conversations = conversation_factory.generate_conversations(window_size=10)
    total_conversations = len(conversations)

    # Process all conversations with a progress bar
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing conversations...", total=total_conversations)
        
        for i, conversation in enumerate(conversations):
            progress.update(task, description=f"Processing conversation {i+1}/{total_conversations}")
            # Apply all extractors to current conversation
            for extractor in extractors:
                extractor.extract_features(conversation)
            progress.advance(task)

    # Print results after all processing is complete
    print("\nExtracted Features Summary:")
    print("-" * 50)
    for i, conversation in enumerate(conversations):
        print(f"\nConversation {i+1}/{total_conversations}:")
        for j, post in enumerate(conversation):
            print(f"\nPost {j+1}:")
            for extractor in extractors:
                feature_value = post.get_property(extractor.property_name)
                print(f"  {extractor.property_name}: {feature_value}")

    # Save processed data
    with open('conversations_with_features.pkl', 'wb') as f:
        pickle.dump(conversations, f)
