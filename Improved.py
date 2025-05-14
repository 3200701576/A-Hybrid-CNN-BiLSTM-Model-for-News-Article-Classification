import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, GlobalMaxPool1D
from tensorflow.keras.models import Sequential

class TextClassifier:
    def __init__(self):
        self.stop_words = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
                             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
                             "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", 
                             "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", 
                             "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
                             "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", 
                             "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", 
                             "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", 
                             "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", 
                             "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
                             "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", 
                             "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", 
                             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
                             "yourself", "yourselves"])
        self.tokenizer = None
        self.model = None
        self.le = LabelEncoder()
        self.max_features = 1500
        self.embedding_size = 8
        self.maxlen = None

    def load_and_preprocess_data(self, filepath):
        df = pd.read_csv(filepath).dropna()
        np.random.seed(38)
        df = df.sample(frac=1).drop_duplicates()
        return df

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text, re.UNICODE).lower()
        text = ' '.join([word for word in text.split() if word not in self.stop_words])
        return text

    def prepare_text_data(self, df):
        df['Title'] = df['Title'].apply(self.clean_text)
        df['Description'] = df['Description'].apply(self.clean_text)
        df['Content'] = df['Title'] + ' ' + df['Description']
        return df

    def prepare_features_labels(self, df):
        X = df.drop(['Class Index', 'Title', 'Description'], axis=1)
        y = df['Class Index']
        return X, y

    def balance_data(self, X, y):
        rus = RandomUnderSampler(random_state=38)
        return rus.fit_resample(X, y)

    def encode_labels(self, y):
        return self.le.fit_transform(y)

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def initialize_tokenizer(self):
        self.tokenizer = Tokenizer(
            num_words=self.max_features, 
            filters='–!"#$”%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
            lower=True
        )

    def preprocess_text(self, texts):
        return pad_sequences(
            self.tokenizer.texts_to_sequences(texts),
            maxlen=self.maxlen
        )

    def calculate_maxlen(self, texts):
        text_lengths = texts.apply(lambda x: len(x.split()))
        self.maxlen = int(np.percentile(text_lengths, 95))
        print(f"Auto-calculated sequence length: {self.maxlen}")

    def build_model(self, num_labels):
        model = Sequential([
            Embedding(self.max_features, self.embedding_size, input_length=self.maxlen),
            Conv1D(256, 4, activation='relu', padding='same'),
            MaxPooling1D(2),
            Dropout(0.35),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPool1D(),
            Dense(350, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.5),
            Dense(num_labels, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        
        start_time = time.time()
        tracemalloc.start()
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=15,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return history, training_time, peak

    def evaluate_model(self, model, X_test, y_test):
        loss, accuracy = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        
        print(f"\nEvaluation Results:")
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        return precision, recall, f1

    def plot_results(self, history, precision, recall, f1):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        metrics = ['Precision', 'Recall', 'F1-score']
        values = [precision, recall, f1]
        colors = ['blue', 'green', 'red']
        
        bars = plt.bar(metrics, values, color=colors)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.show()

    def run(self, filepath):
        # Load and preprocess data
        df = self.load_and_preprocess_data(filepath)
        df = self.prepare_text_data(df)
        
        # Prepare features and labels
        X, y = self.prepare_features_labels(df)
        X, y = self.balance_data(X, y)
        y = self.encode_labels(y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Text preprocessing
        self.calculate_maxlen(df['Content'])
        self.initialize_tokenizer()
        self.tokenizer.fit_on_texts(X_train['Content'].values)
        
        X_train = self.preprocess_text(X_train['Content'].values)
        X_val = self.preprocess_text(X_val['Content'].values)
        X_test = self.preprocess_text(X_test['Content'].values)
        
        # Prepare labels
        num_labels = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes=num_labels)
        y_val = to_categorical(y_val, num_classes=num_labels)
        y_test = to_categorical(y_test, num_classes=num_labels)
        
        # Build and train model
        model = self.build_model(num_labels)
        history, training_time, peak_memory = self.train_model(
            model, X_train, y_train, X_val, y_val)
        
        print(f"\nTraining Details:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Peak memory usage: {peak_memory / 1024**2:.2f} MB")
        
        # Evaluate and plot results
        precision, recall, f1 = self.evaluate_model(model, X_test, y_test)
        self.plot_results(history, precision, recall, f1)

if __name__ == "__main__":
    classifier = TextClassifier()
    classifier.run("./data/data.csv")