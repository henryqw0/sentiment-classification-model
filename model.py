import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle

# STEP 1 - LOAD DATA
df = pd.read_csv("sentiment_2000.csv")
texts = df['text']
labels = df['label']

# STEP 2 - SPLIT FIRST
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# STEP 3 - VECTORIZE
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1,2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train)   # learn + convert 
X_test  = vectorizer.transform(X_test)    # convert only
vocabs = vectorizer.get_feature_names_out()  # view vocabulary

# STEP 4 - TRAIN
model = LogisticRegression(max_iter=1000 , C=0.5)
model.fit(X_train, y_train)

# STEP 5 - PREDICT
preds = model.predict(X_test)
probas = model.predict_proba(X_test)

# STEP 6 - EVALUATE and SAVE
print("Predicted:", preds)
print("Actual:   ", y_test)
print('Accuracy:', model.score(X_test, y_test))
print (classification_report(y_test, preds))
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))