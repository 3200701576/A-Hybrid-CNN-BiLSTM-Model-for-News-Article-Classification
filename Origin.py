import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import os
import re
import time
import tracemalloc
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, precision_score, recall_score, f1_score
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential

# Load and prepare data
def load_data():
    df = pd.read_csv("./data.csv").dropna()
    np.random.seed(38)
    df = df.sample(frac=1)
    df = df.drop_duplicates()
    return df

# Text cleaning functions
def initialize_stopwords():
    return set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
                "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", 
                "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
                "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", 
                "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", 
                "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", 
                "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", 
                "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", 
                "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
                "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", 
                "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", 
                "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])

def clean_text(text, stop_words):
    text = re.sub(r'[^\w\s]','', text, re.UNICODE)
    text = text.lower()
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)

def preprocess_data(df, stop_words):
    df['Title'] = df['Title'].apply(lambda x: clean_text(x, stop_words))
    df['Description'] = df['Description'].apply(lambda x: clean_text(x, stop_words))
    df['Content'] = df['Title'] + ' ' + df['Description']
    return df

# Model building functions
def build_model(max_features, embedding_size, num_labels):
    model = Sequential([
        Embedding(max_features, embedding_size),
        Conv1D(256, 4, activation='relu'),
        MaxPooling1D(5),
        Dropout(0.25),
        Conv1D(64, 10, activation='relu'),
        MaxPooling1D(5),
        LSTM(32, return_sequences=True),
        LSTM(64),
        Dense(350, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(num_labels, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                 loss=tf.keras.losses.CategoricalCrossentropy(), 
                 metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test_onehot):
    loss, accuracy = model.evaluate(X_test, y_test_onehot)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_onehot, axis=1)
    
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return precision, recall, f1

def plot_results(history, precision, recall, f1):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(['Precision', 'Recall', 'F1-score'], [precision, recall, f1], color=['blue', 'green', 'red'])
    plt.title('Model Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for i, v in enumerate([precision, recall, f1]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.show()

# Main execution
def main():
    # Initialize and load data
    stop_words = initialize_stopwords()
    df = load_data()
    df = preprocess_data(df, stop_words)
    
    # Prepare features and labels
    X = df.drop(['Class Index', 'Title', 'Description'], axis=1)
    y = df['Class Index']
    
    # Undersample data
    rus = RandomUnderSampler(random_state=38)
    X, y = rus.fit_resample(X, y)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_labels = len(np.unique(y))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    
    # Text vectorization
    max_features = 1500
    embedding_size = 8
    maxlen = 250
    
    tokenizer = Tokenizer(num_words=max_features, filters='–!"#$”%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    X_train_text = X_train['Content'].values
    tokenizer.fit_on_texts(X_train_text)
    
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=maxlen)
    X_val = pad_sequences(tokenizer.texts_to_sequences(X_val['Content'].values), maxlen=maxlen)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test['Content'].values), maxlen=maxlen)
    
    # One-hot encode labels
    y_train_onehot = to_categorical(y_train, num_classes=num_labels)
    y_val_onehot = to_categorical(y_val, num_classes=num_labels)
    y_test_onehot = to_categorical(y_test, num_classes=num_labels)
    
    # Build and train model
    model = build_model(max_features, embedding_size, num_labels)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    
    # Track resources
    start_time = time.time()
    tracemalloc.start()
    
    history = model.fit(X_train, y_train_onehot, 
                       batch_size=32, 
                       epochs=20, 
                       validation_data=(X_val, y_val_onehot), 
                       callbacks=[early_stopping])
    
    # Get resource usage
    end_time = time.time()
    training_time = end_time - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
    
    # Evaluate and plot results
    precision, recall, f1 = evaluate_model(model, X_test, y_test_onehot)
    plot_results(history, precision, recall, f1)

if __name__ == "__main__":
    main()
