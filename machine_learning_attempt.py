"""
Was an attempt at developing a semisupervised training model to determine whether a movie is sad or happy based on their movie synapses tf-idf vectors.
"""


import pandas as pd
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer # set up stemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import FunctionTransformer, normalize
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.base import TransformerMixin, BaseEstimator
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

stemmer = SnowballStemmer("english")

stopwords=[]
with open('mySQL_stopwords.txt','r') as f:
    for line in f:
        stopwords.append(line.split())
stopwords = [item for sublist in stopwords for item in sublist]
stopwords = set(stopwords)

summary_df = pd.read_json('IMDB_movie_details.json', lines=True)
summary_df.drop(columns=['plot_summary', 'duration', 'genre', 'rating', 'release_date'], inplace=True)
summary_df['movie_id'] = summary_df['movie_id'].astype('category')

movie_id_list = summary_df['movie_id'].cat.categories
nmovies = 20
movie_corpus = []

#  Sad movies: tt0332280=Notebook, tt0120338=Titanic, tt0102492=My Girl, tt2674426=Me before you, tt0395169=Hotel Rwanda, tt0070903=The way we were
# tt0180093=Requeim for a Dream, tt0405159=Million dollar baby, tt1255953=Incendies, tt0959337=Revolutionary Road, tt0108052=Schindler's list
# tt0109830= Forrest Gump, tt0118799=Life is beautiful, tt0120689=Green mile, tt2582846=The fault in our stars
# tt0095327=Grave of the fireflies, tt0398808=The bridge to Terabithia, tt0080678=The elephant man, tt4034228=Manchester by the sea

#  Happy movies: tt0319343=Elf, tt0795421=Mamma mia, tt0096283=My neighbor Totoro, tt0432283=Fantastic Mr. Fox, tt1109624=Paddington, tt0337741=Something's gotta give
# tt1675434=Intouchables, tt0211915=Amelie, tt0198781=Monsters Inc, tt1490017=Lego movie, tt0486655=Stardust, tt0038650=It's a wonderful life
# tt0093779=The princess bride, tt3783958= La la land, tt2278388=The grand budapest hotel, tt0045152=Singin' in the rain
# tt0126029=Shrek, tt1485796=The greateet showman, tt2948356=Zootopia

sad_movie_ids = ['tt0332280', 'tt0120338', 'tt0102492', 'tt2674426', 'tt0395169', 'tt0070903', 'tt0180093',
                'tt0405159', 'tt1255953', 'tt0959337', 'tt0108052', 'tt0109830', 'tt0118799', 'tt0120689', 'tt2582846',
                'tt0095327', 'tt0398808', 'tt0080678', 'tt4034228']
happy_movie_ids = ['tt0319343', 'tt0795421', 'tt0096283', 'tt0432283', 'tt1109624', 'tt0337741', 'tt1675434', 'tt0211915',
                'tt0198781', 'tt1490017', 'tt0486655', 'tt0038650', 'tt0093779', 'tt3783958', 'tt2278388', 'tt0045152',
                'tt0126029', 'tt1485796', 'tt2948356']

print(len(sad_movie_ids))
print(len(happy_movie_ids))
summary_df['label'] = np.select([summary_df['movie_id'].isin(sad_movie_ids), summary_df['movie_id'].isin(happy_movie_ids)],
                                [1, 0],
                                default=-1)


movie_label = []

labeled_movies_corpus = []

for movie_id in movie_id_list:
    mask = summary_df['movie_id'] == movie_id
    film_summary_string = ' '.join(summary_df[mask]['plot_synopsis'].to_list()).lower()
    # This next line takes out stopwords
    film_summary_string = [word for word in re.split(r"[-;,.?!/()\s]\s*",film_summary_string) if word not in stopwords]
    film_summary_string = [stemmer.stem(i) for i in film_summary_string] # stems words
    film_summary_string = ' '.join(film_summary_string)
    movie_corpus.append(film_summary_string)
    if (movie_id in sad_movie_ids) or (movie_id in happy_movie_ids):
        labeled_movies_corpus.append(film_summary_string)

sdg_params = dict(alpha=1e-5, penalty="l2", loss="log_loss")
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
mlp_params = dict(hidden_layer_sizes=(100,50), max_iter=100,activation = 'relu',solver='adam',random_state=1,learning_rate_init=0.01,
                  learning_rate='adaptive')

model_svc = SVC(kernel='rbf', 
                probability=True, 
                C=1.0,
                gamma='scale',
                random_state=0
               )
#SGDClassifier(**sdg_params)

class CustomSentenceTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Tokenize each text into sentences
        sentence_lists = [sent_tokenize(text) for text in X]
        result = [' '.join(sentences) for sentences in sentence_lists]
        return result

class CustomTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, sublinear_tf=True):
        self.tfidf_vectorizer = TfidfVectorizer(sublinear_tf=sublinear_tf)

    def fit(self, X, y=None):
        return self.tfidf_vectorizer.fit(X, y)

    def transform(self, X):
        # Tokenize and transform each sentence using custom weighting
        sentence_weights = np.arange(1, len(X) + 1)
        weighted_sentences = []
        for sentence in X:
            sentence_vector = self.tfidf_vectorizer.transform([sentence])
            weighted_sentence_vector = sentence_vector.multiply(sentence_weights[:, np.newaxis])
            weighted_sentences.append(weighted_sentence_vector)

        combined_vector = sum(weighted_sentences)
        normalized_vector = normalize(combined_vector)
        print(normalized_vector)
        return normalized_vector

st_pipeline = Pipeline(
    [
        ("sent_tokenizer", CustomSentenceTokenizer()),  # Add your custom sentence tokenizer
        ("tfidf", CustomTfidfVectorizer()),  # Use the custom TF-IDF vectorizer
        ("clf", SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True)),
    ]
)

lg_pipeline = Pipeline(
    [
        ("vect", CountVectorizer(**vectorizer_params)),
        ("tfidf", TfidfTransformer()),
        ("clf", LogisticRegression())
    ]
)

sp_pipeline = Pipeline(
    [
        ("vect", CountVectorizer(**vectorizer_params)),
        ("tfidf", TfidfTransformer()),
        ("clf", SVC(kernel='rbf', probability=True))
    ]
)

ls_pipeline = Pipeline(
    [
        ("vect", CountVectorizer(**vectorizer_params)),
        ("tfidf", TfidfTransformer()),
        # LabelSpreading does not support dense matrices
        ("toarray", FunctionTransformer(lambda x: x.toarray())),
        ("clf", LabelSpreading()),
    ]
)


def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    print("Number of training samples:", len(X_train))
    print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(
        "Micro-averaged F1 score on test set: %0.3f"
        % f1_score(y_test, y_pred, average="micro")
    )
    
    print("-" * 10)
    print()





# Define hyperparameters
batch_size = 32
learning_rate = 2e-5
epochs = 4

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load and preprocess the IMDb dataset (you can replace this with your own dataset)
# For simplicity, we'll load a small subset here. In practice, you would load your dataset.
texts = labeled_movies_corpus
labels = summary_df.loc[summary_df['label']!=-1, 'label'].tolist()

# Tokenize and encode the text data
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)

# Create a DataLoader for training
dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], torch.tensor(labels))
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

# Training loop
for epoch in range(epochs):
    model.train()
total_loss = 0
for batch in train_dataloader:
    input_ids, attention_mask, label = batch
    optimizer.zero_grad()
    output = model(input_ids, attention_mask=attention_mask, labels=label)
    loss = output.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Epoch {epoch + 1}: Average Training Loss = {total_loss / len(train_dataloader)}")

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, label = batch
        output = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(output.logits, dim=1)
        correct += (predictions == label).sum().item()
        total += len(label)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy}")

# You can now use the fine-tuned BERT model for sentiment analysis on new text data

# Define your new movie plot synopses
new_synopses = movie_corpus

# Tokenize and encode the new synopses
inputs = tokenizer(new_synopses, padding=True, truncation=True, return_tensors="pt", max_length=256)

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits

# Determine the predicted sentiment
predicted_labels = torch.argmax(logits, dim=1)

# Map the predicted labels to sentiment classes
sentiment_classes = ["Happy", "Not Happy"]
predicted_sentiments = [sentiment_classes[label] for label in predicted_labels]

# Print the predicted sentiments for the new synopses
for synopsis, sentiment in zip(new_synopses, predicted_sentiments):
    print(f"Synopsis: {synopsis}")
    print(f"Predicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    # X = labeled_movies_corpus
    # y = summary_df.loc[summary_df['label']!=-1, 'label']
    #X = movie_corpus
    #y = summary_df['label']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #eval_and_print_metrics(st_pipeline, X_train, y_train, X_test, y_test)

    #predictions = st_pipeline.predict(movie_corpus)
    """summary_df['label'] = predictions
    sad_movies = summary_df.loc[summary_df['label']== 1, 'movie_id']
    print(sad_movies)
    print(len(sad_movies))
    print(summary_df.loc[summary_df['movie_id'].isin(sad_movie_ids), 'label'])"""
    #happy_movies = summary_df.loc[summary_df['label']== 0, 'movie_id']
    #print(len(happy_movies))
    #print(summary_df.loc[summary_df['movie_id'].isin(happy_movie_ids), 'label'])
    # print(summary_df.loc[summary_df['movie_id'] == 'tt0163651', 'label'])

    
    #model.score(X_test, y_test)




    """eval_and_print_metrics(st_pipeline, X, y)
    predictions = st_pipeline.predict(X)
    summary_df['label'] = predictions
    happy_movies = summary_df.loc[summary_df['label']== 0, 'movie_id']
    print(summary_df.loc[summary_df['movie_id'] == 'tt0163651', 'label'])
    print(len(happy_movies))"""
    #print(summary_df.loc[summary_df['movie_id'].isin(happy_movie_ids), 'label'])
    #print(summary_df)