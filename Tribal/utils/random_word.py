import random

def generate_random_name():
    verbs = [
        "Jumping", "Running", "Flying", "Swimming", "Dancing", "Singing", "Laughing", "Crawling",
        "Gliding", "Soaring", "Racing", "Hopping", "Bouncing", "Skipping", "Marching", "Climbing",
        "Sliding", "Swinging", "Diving", "Hiking", "Galloping", "Twirling", "Leaping", "Chasing"
    ]
    adjectives = [
        "Happy", "Sad", "Brave", "Clever", "Eager", "Jolly", "Kind", "Lively",
        "Gentle", "Bold", "Bright", "Calm", "Cheerful", "Energetic", "Friendly", "Graceful",
        "Joyful", "Mighty", "Playful", "Quiet", "Radiant", "Serene", "Strong", "Vibrant"
    ]
    nouns = [
        "Lion", "Mountain", "River", "Sky", "Tree", "Star", "Ocean", "Eagle",
        "Wolf", "Forest", "Desert", "Moon", "Sun", "Rain", "Thunder", "Flower",
        "Tiger", "Valley", "Canyon", "Lake", "Hawk", "Storm", "Field", "Peacock"
    ]

    verb = random.choice(verbs)
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)

    random_name = f"{adjective}-{verb}-{noun}"
    return random_name