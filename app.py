import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_text

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis for Reviews",
    page_icon="😊",
    layout="centered"
)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform the text using the vectorizer
    text_tfidf = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability

def main():
    # Title
    st.title("Sentiment Analysis for Reviews")
    
    # Warning message about English-only support
    st.warning("⚠️ This model only supports English text. Please enter your review in English.")
    
    # Load model
    try:
        model, vectorizer = load_model()
    except:
        st.error("Model not found. Please make sure you've run model.py first")
        return
    
    # Text input
    text_input = st.text_area("Enter your review here:", height=150)
    
    if st.button("Analyze Sentiment"):
        if text_input:
            # Make prediction
            prediction, probability = predict_sentiment(text_input, model, vectorizer)
            
            # Display results
            st.write("---")
            st.subheader("Result:")
            
            if prediction == 1:
                st.success("Positive 😊")
                st.write(f"Confidence: {probability[1]:.2%}")
            else:
                st.error("Negative 😞")
                st.write(f"Confidence: {probability[0]:.2%}")
            
            # Display preprocessed text
            with st.expander("Preprocessed Text"):
                st.write(preprocess_text(text_input))
        else:
            st.warning("Please enter a review text")

if __name__ == "__main__":
    main() 