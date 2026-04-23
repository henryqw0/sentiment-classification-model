# Sentiment Classification Model

This project is a simple sentiment analysis system that classifies text as **negative**, **neutral**, or **positive**. It uses a **TF-IDF vectorizer** with a **Logistic Regression** classifier, then serves predictions through a small **Streamlit** web app.

The repository includes:

- `data.py` to generate a synthetic training dataset
- `model.py` to train and evaluate the classifier
- `ui.py` to run the Streamlit interface
- `sentiment_2000.csv` as the training dataset
- `model.pkl` and `vectorizer.pkl` as saved training artifacts

## How It Works

1. `data.py` builds a balanced dataset of short review-style sentences.
2. `model.py` loads the dataset and splits it into training and test sets.
3. Text is converted into numeric features with `TfidfVectorizer`.
4. A `LogisticRegression` model is trained on those features.
5. The trained model is evaluated and saved with `pickle`.
6. `ui.py` loads the saved files and predicts sentiment for user-entered text.

## Labels

The model uses numeric labels internally:

- `0` = Negative
- `1` = Neutral
- `2` = Positive

The Streamlit app converts these values into readable labels for the user.

## Model Details

- Algorithm: `LogisticRegression`
- Text features: `TfidfVectorizer`
- Max features: `500`
- N-grams: unigrams and bigrams (`(1, 2)`)
- Stop words: English
- Train/test split: `80/20`
- Random state: `42`

## Project Structure

```text
Sentiment Classification Model/
├── data.py
├── model.py
├── ui.py
├── sentiment_2000.csv
├── model.pkl
├── vectorizer.pkl
└── README.md
```

## Installation

Install the required packages:

```bash
pip install pandas scikit-learn streamlit
```

## Usage

### 1. Generate the dataset

If you want to recreate the dataset:

```bash
python data.py
```

This will generate `sentiment_2000.csv`.

### 2. Train the model

Run:

```bash
python model.py
```

The script will:

- load the dataset
- train the model
- print accuracy and a classification report
- save `model.pkl` and `vectorizer.pkl`

On the current dataset and split, the script produced an accuracy of about **0.84** before attempting to save the model files.

### 3. Start the web app

Run:

```bash
streamlit run ui.py
```

Then open the local Streamlit URL in your browser to test the classifier.

## App Features

- text input area for custom sentences
- one-click sentiment prediction
- confidence score display
- low-confidence warning for unfamiliar or weak-signal text
- clear button to reset the input

## Example Predictions

- `"I love this app."` → Positive
- `"This tool is okay."` → Neutral
- `"I regret buying this product."` → Negative

## Notes

- The training data is **synthetic**, so the model is best viewed as a learning project or demo rather than a production-ready sentiment system.
- Because the dataset is template-based, performance on real-world text may be lower.
- If `python model.py` raises a permission error while saving `model.pkl` or `vectorizer.pkl`, make sure those files are not open or locked by another running process such as the Streamlit app.

## Future Improvements

- train on a larger real-world dataset
- add preprocessing for punctuation, emojis, and slang
- save metrics more cleanly
- add automated tests
- package dependencies in a `requirements.txt` file
- deploy the Streamlit app online

## License

This project is licensed under the MIT License.
