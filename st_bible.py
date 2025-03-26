import streamlit as st
import numpy as np
import pandas as pd
import string
import nltk
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords 
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import gdown
import pickle
url = "https://drive.google.com/file/d/1-z5RDrWKn13t65PmsWb4FhOGyRcJbOpB/view?usp=sharing"
output = "bible_embeddings.npy"
gdown.download(url, output, quiet=False)

url = "https://drive.google.com/file/d/1I7sqgWmMjFcjqDVic73IMPXK8tehcX-A/view?usp=sharing"
output = "bible_faiss.index"
gdown.download(url, output, quiet=False)

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('stopwords')

# Streamlit Page Configuration
st.set_page_config(page_title="Bible Verse Recommender", layout="wide")
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_data():
    data = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    book_names = {
        1: 'Genesis', 2: 'Exodus', 3: 'Leviticus', 4: 'Numbers', 5: 'Deuteronomy',
        6: 'Joshua', 7: 'Judges', 8: 'Ruth', 9: '1 Samuel', 10: '2 Samuel',
        11: '1 Kings', 12: '2 Kings', 13: '1 Chronicles', 14: '2 Chronicles',
        15: 'Ezra', 16: 'Nehemiah', 17: 'Esther', 18: 'Job', 19: 'Psalms',
        20: 'Proverbs', 21: 'Ecclesiastes', 22: 'Song of Solomon', 23: 'Isaiah',
        24: 'Jeremiah', 25: 'Lamentations', 26: 'Ezekiel', 27: 'Daniel',
        28: 'Hosea', 29: 'Joel', 30: 'Amos', 31: 'Obadiah', 32: 'Jonah',
        33: 'Micah', 34: 'Nahum', 35: 'Habakkuk', 36: 'Zephaniah', 37: 'Haggai',
        38: 'Zechariah', 39: 'Malachi', 40: 'Matthew', 41: 'Mark', 42: 'Luke',
        43: 'John', 44: 'Acts', 45: 'Romans', 46: '1 Corinthians',
        47: '2 Corinthians', 48: 'Galatians', 49: 'Ephesians', 50: 'Philippians',
        51: 'Colossians', 52: '1 Thessalonians', 53: '2 Thessalonians',
        54: '1 Timothy', 55: '2 Timothy', 56: 'Titus', 57: 'Philemon',
        58: 'Hebrews', 59: 'James', 60: '1 Peter', 61: '2 Peter',
        62: '1 John', 63: '2 John', 64: '3 John', 65: 'Jude', 66: 'Revelation'
    }

    # Map book numbers to names
    data['Book Name'] = data['b'].map(book_names)

    # Process text: Remove stopwords
    stop_words = set(stopwords.words('english'))
    data['corpus'] = data['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )

    return data, book_names

# Load data
data, book_names = load_data()
##################################################################################################################
# Compute TF-IDF & Cosine Similarity
@st.cache_resource
def compute_similarity():
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(data['corpus'])
    return cosine_similarity(tf_idf_matrix)

similarity_matrix = compute_similarity()

# Reverse book name lookup
book_numbers = {v: k for k, v in book_names.items()}

# **Find Similar Verses Function**
def top_verse(input_book, input_chapter, input_verse, top_n=10):
    try:
        book_num = str(book_numbers.get(input_book, ""))
        locator = data.loc[
            (data['b'].astype(str) == book_num) &
            (data['c'].astype(str) == str(input_chapter)) &
            (data['v'].astype(str) == str(input_verse))
        ]
        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])
        idx = locator.index[0]
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
        sim_values = [i[1] for i in similarity_scores[1:top_n + 1]]
        recommended = data.iloc[sim_indices].copy()
        recommended['Similarity Score'] = sim_values
        recommended = recommended[['Book Name', 'c', 'v', 't', 'Similarity Score']]
        recommended.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        return recommended
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])

##################################################################################################################

# Load Data for BERT model
df = pd.read_csv("bible_verses_processed.csv")
embeddings = np.load("bible_embeddings.npy")

# Load FAISS index
index = faiss.read_index("bible_faiss.index")

# Load Sentence-BERT model for query embedding
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Function to get embedding for a new query
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy()

# Find similar verses
def find_similar_verses(query, top_n=5):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n)

    # Get top matching verses, along with book, chapter, and verse
    results = df.iloc[indices[0]][["Book Name", "c", "v", "t"]].copy()
    results["similarity"] = 1 - distances[0]  # Convert distance to similarity score
    return results
##################################################################################################################

# **Streamlit UI**
tab1, tab2 = st.tabs(["Verse Recommender", "Semantic Search Recommender"])

with tab1:
    st.title("📖 Bible Verse Recommender")
    st.write("Find verses similar to your selection from the Old and New Testament.")

    with st.form("user_input"):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_book = st.selectbox("Select Book", book_names.values())
        with col2:
            input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
        with col3:
            input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1, step=1)

        top_n = st.slider("Number of Similar Verses", min_value=1, max_value=50, value=10, step=5)

        submitted = st.form_submit_button("Find Similar Verses")

    if submitted:
        results = top_verse(input_book, input_chapter, input_verse, top_n)
        searched_verse = data.loc[
            (data['Book Name'] == input_book) &
            (data['c'].astype(str) == str(input_chapter)) &
            (data['v'].astype(str) == str(input_verse))
        ]

        if not searched_verse.empty:
            st.write(f"**Input Verse:** {searched_verse.iloc[0]['t']}")
            st.write("### 🔍 Similar Verses:")
        else:
            st.write("Verse not found.")

        st.table(results)
##################################################################################################################
with tab2:

    # Streamlit UI
    st.title("📖 Bible Verse Similarity Finder")
    query = st.text_input("Enter a phrase or verse:", "Love your neighbor as yourself")
    top_n = st.slider("Number of similar verses:", min_value=1, max_value=10, value=5)

    if st.button("Find Similar Verses"):
        results = find_similar_verses(query, top_n)
        st.write("### 🔍 Similar Verses:")
        for i, row in results.iterrows():
            st.write(f"**Book:** {row['Book Name']} | **Chapter:** {row['c']} | **Verse:** {row['v']}")
            st.write(f"**Text:** {row['t']} (Similarity: {row['similarity']:.2f})")



