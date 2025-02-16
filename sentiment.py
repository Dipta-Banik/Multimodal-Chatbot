from dotenv import load_dotenv
import os
import pickle

load_dotenv()
pickle_model_path = os.getenv("PICKLE_FILE_PATH")
pickle_vectorizer_path = os.getenv("PICKLE_VECTORIZER_PATH")
pickle_encoder_path = os.getenv("PICKLE_ENCODER_PATH")

if not pickle_model_path or not pickle_vectorizer_path or not pickle_encoder_path:
    raise ValueError("Error: One or more file paths are not set in the .env file.")

with open(pickle_model_path, "rb") as model_file:
    loaded_best_model = pickle.load(model_file)

with open(pickle_vectorizer_path, "rb") as vectorizer_file:
    cv = pickle.load(vectorizer_file)

with open(pickle_encoder_path, "rb") as encoder_file:
    le = pickle.load(encoder_file)


def get_sentiment(text):
    text_cv = cv.transform([text]).toarray()
    prediction_encoded = loaded_best_model.predict(text_cv)[0] 
    predictions = le.inverse_transform([prediction_encoded])[0]
    return predictions
