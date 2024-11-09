from fasttext import FastText

# Train a text classification model
model = FastText.train_supervised(input="train.txt")

# Predict the class of a new text
labels, probabilities = model.predict("This is a test sentence.")
print(labels, probabilities)


#e.g. train off a user window to predict next words