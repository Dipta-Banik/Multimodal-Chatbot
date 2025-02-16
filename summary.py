import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import heapq

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(sentence):
    words = word_tokenize(re.sub('[^a-zA-Z]', ' ', sentence.lower()))
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tag(words)
    ]
    return " ".join(lemmatized_words)

def extract_important_entities(paragraph):
    doc = nlp(paragraph)
    entities = [ent.text for ent in doc.ents]
    return entities

def summarize_text(paragraph):
    sentences = sent_tokenize(paragraph)
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    entities = extract_important_entities(paragraph)
    tfidf = TfidfVectorizer()
    tfidf.fit(processed_sentences)
    word_scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        score = sum(word_scores.get(word, 0) for word in words)
        for entity in entities:
            if entity.lower() in sentence.lower():
                score += 1
        sentence_scores[sentence] = score
    top_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(top_sentences)
    return summary
