import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.chunk import tree2conlltags
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def load_data():
    # Load the raw data
    df = pd.read_csv('SmallReviews.csv')
    return df

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    # 1. Tokenization
    tokens = word_tokenize(str(text).lower())
    
    # 2. Remove Punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # 3. Remove Stop Words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # 4. Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # 6. POS Tagging
    pos_tags = pos_tag(tokens)
    
    # 7. Named Entity Recognition (NER)
    ner_tags = ne_chunk(pos_tags)
    
    # 8. Chunking (Simple Noun Phrase Chunking)
    grammar = r"NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(grammar)
    chunked = chunk_parser.parse(pos_tags)
    
    # Return the processed tokens (using lemmatized tokens as the base)
    return ' '.join(lemmatized_tokens)

def process_chunk(chunk):
    """Process a chunk of data in parallel"""
    return chunk['Text'].apply(preprocess_text)

def remove_rare_common_words(texts, rare_threshold=2, common_threshold_percent=0.8):
    # 9. Remove Rare and Most Common Words
    all_words = [word for text in texts for word in text.split()]
    word_freq = Counter(all_words)
    common_threshold = int(common_threshold_percent * len(texts))
    
    filtered_words = {word for word, freq in word_freq.items() 
                     if rare_threshold <= freq <= common_threshold}
    
    return [' '.join(word for word in text.split() if word in filtered_words) 
            for text in texts]

def create_bow_tfidf(texts):
    # 10. Bag of Words
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    bow_feature_names = vectorizer.get_feature_names_out()
    
    # 11. TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    return bow_matrix, tfidf_matrix, bow_feature_names, tfidf_feature_names

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Preprocess text using parallel processing
    print("Preprocessing text using parallel processing...")
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(df) // num_cores
    
    # Split data into chunks
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    processed_texts = pd.concat(results).tolist()
    
    # Remove rare and common words
    print("Removing rare and common words...")
    filtered_texts = remove_rare_common_words(processed_texts)
    
    # Create BoW and TF-IDF representations
    print("Creating BoW and TF-IDF representations...")
    bow_matrix, tfidf_matrix, bow_features, tfidf_features = create_bow_tfidf(filtered_texts)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    df['processed_text'] = filtered_texts
    df.to_csv('preprocessed_data.csv', index=False)
    
    # Save vectorizers and matrices
    import joblib
    joblib.dump(bow_matrix, 'bow_matrix.joblib')
    joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')
    joblib.dump(bow_features, 'bow_features.joblib')
    joblib.dump(tfidf_features, 'tfidf_features.joblib')
    
    print("Preprocessing completed!")
    print(f"Number of BoW features: {len(bow_features)}")
    print(f"Number of TF-IDF features: {len(tfidf_features)}")

if __name__ == "__main__":
    main() 