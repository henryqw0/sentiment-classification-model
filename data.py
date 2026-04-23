import random
import pandas as pd

# Define some positive, negative, and neutral templates
positive_templates = [
    "I love this {}.",
    "This {} is fantastic.",
    "I would definitely recommend this {}.",
    "The {} exceeded my expectations."
]

negative_templates = [
    "I hate this {}.",
    "This {} is terrible.",
    "I regret buying this {}.",
    "The {} was a waste of money."
]

neutral_templates = [
    "This {} is okay.",
    "The {} is fine, nothing special.",
    "It does the job, but nothing more.",
    "The {} is acceptable."
]

# Ambiguous or mixed templates
ambiguous_templates = [
    "I thought this {} would be bad, but it's actually okay.",
    "The {} is good, but not perfect.",
    "Not terrible, but not great either, just an average {}.",
    "I expected more from this {}, but it's not bad."
]

# Things we are reviewing
objects = ["product", "service", "experience", "tool", "app", "device"]

# Generate data
data = []

# Create balanced positive, negative, neutral, and ambiguous examples
for _ in range(500):
    obj = random.choice(objects)
    data.append([random.choice(positive_templates).format(obj), 2])
    data.append([random.choice(negative_templates).format(obj), 0])
    data.append([random.choice(neutral_templates).format(obj), 1])
    data.append([random.choice(ambiguous_templates).format(obj), random.choice([0, 1, 2])])

# Shuffle the data to mix labels
random.shuffle(data)

# Convert to DataFrame and save
df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("sentiment_2000.csv", index=False)
