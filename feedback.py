import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('feedback.csv')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
data['cleaned_feedback'] = data['feedback'].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_feedback'], data['sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
def analyze_sentiment(feedback):
    cleaned_feedback = preprocess_text(feedback)
    feedback_tfidf = vectorizer.transform([cleaned_feedback])
    sentiment = model.predict(feedback_tfidf)
    return sentiment[0]
#example
new_feedback = "The counseling session was extremely helpful!"
print(f"Sentiment: {analyze_sentiment(new_feedback)}")
