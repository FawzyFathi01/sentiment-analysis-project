import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Negation words
negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing', 'nowhere', 'hardly', 'barely', 'scarcely', 'doesnt', 'isnt', 'wasnt', 'shouldnt', 'wouldnt', 'couldnt', 'wont', 'cant', 'dont'}

# Intensifiers
intensifiers = {
    'very': 2.0,
    'extremely': 2.0,
    'really': 1.5,
    'so': 1.5,
    'too': 1.5,
    'absolutely': 2.0,
    'completely': 2.0,
    'totally': 2.0,
    'incredibly': 2.0,
    'amazingly': 2.0,
    'exceptionally': 2.0,
    'particularly': 1.5,
    'especially': 1.5,
    'slightly': 0.5,
    'somewhat': 0.5,
    'rather': 0.5,
    'quite': 0.5,
    'pretty': 0.5,
    'fairly': 0.5,
    'a bit': 0.5,
    'a little': 0.5
}

def load_data():
    # Load the raw data
    df = pd.read_csv('SmallReviews.csv')
    return df

def handle_negation(tokens):
    negated = False
    result = []
    
    for token in tokens:
        if token in negation_words:
            negated = not negated
        elif negated:
            # Add NEG_ prefix to negated words
            result.append('NEG_' + token)
        else:
            result.append(token)
    
    return result

def handle_intensifiers(tokens):
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in intensifiers:
            # Add intensity marker
            result.append('INT_' + tokens[i])
            if i + 1 < len(tokens):
                # Mark the next word as intensified
                result.append('INTENSIFIED_' + tokens[i + 1])
                i += 2
                continue
        result.append(tokens[i])
        i += 1
    return result

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep negation markers
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove English stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Handle negation
        tokens = handle_negation(tokens)
        
        # Handle intensifiers
        tokens = handle_intensifiers(tokens)
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        return processed_text
    return ''

def process_chunk(chunk):
    """Process a chunk of data in parallel"""
    return chunk['Text'].apply(preprocess_text)

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
    df['processed_text'] = pd.concat(results)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    df.to_csv('preprocessed_data.csv', index=False)
    print("Preprocessing completed!")

if __name__ == "__main__":
    main() 