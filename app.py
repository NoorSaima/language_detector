import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# CSV file path (Ensure it has two columns: 'text' and 'language')
csv_file = "language_data.csv"

# Set Streamlit page configuration
st.set_page_config(page_title="🌐 Language Detector (KNN)", page_icon="🌍", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    .logo-container { display: flex; justify-content: center; }
    body { background-color: #f0f8ff !important; }
    div[data-testid="stMarkdownContainer"] p { color: #2c3e50 !important; }
    h1, h2, h3 { color: #1d3557 !important; }
    section[data-testid="stSidebar"] { background-color: #d6eaf8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌐 Enhanced Language Detector (KNN)")
st.write("Accurately detect languages using K-Nearest Neighbors with balanced data and advanced visualization.")

# Sidebar input
st.sidebar.header("📝 Input Text")
user_input = st.sidebar.text_area("Enter text for language detection:")

# Load and clean dataset
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
        
        # Ensure only two columns are present
        if df.shape[1] != 2:
            st.error("❗ CSV must contain exactly two columns: 'text' and 'language'.")
            st.stop()

        # Drop missing and duplicate rows
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # Ensure correct column names
        df.columns = ["text", "language"]

        return df
    except Exception as e:
        st.error(f"❗ Error loading CSV file: {e}")
        st.stop()

# Balance dataset (Oversample minority classes for better model performance)
def balance_dataset(df):
    balanced_df = pd.DataFrame()
    max_samples = df["language"].value_counts().max()
    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang]
        oversampled = resample(lang_df, replace=True, n_samples=max_samples, random_state=42)
        balanced_df = pd.concat([balanced_df, oversampled])
    return balanced_df.sample(frac=1).reset_index(drop=True)

# Train KNN model with cached resources
@st.cache_resource
def train_knn(df, k=7):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    le = LabelEncoder()
    y = le.fit_transform(df["language"])
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    model.fit(X, y)
    return model, vectorizer, le

# Detect language function
def detect_language_knn(text, model, vectorizer, label_encoder):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return label_encoder.inverse_transform(prediction)[0]

# Main logic
dataset = load_dataset()
balanced_dataset = balance_dataset(dataset)
knn_model, tfidf_vectorizer, label_encoder = train_knn(balanced_dataset)

# Detect language on button click
if st.sidebar.button("🌍 Detect Language"):
    if user_input:
        with st.spinner("Detecting language..."):
            time.sleep(1)  # Simulate processing time
            result = detect_language_knn(user_input, knn_model, tfidf_vectorizer, label_encoder)
            st.success(f"🗣️ Detected Language: **{result}**")
            
            # Store detection history in session state
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"Input": user_input, "Detected Language": result})
    else:
        st.warning("❗ Please enter some text.")

# Tabs for additional insights
tab1, tab2, tab3 = st.tabs(["📜 Detection History", "📊 Data Insights", "📂 View Dataset"])

# Tab 1: Detection History
with tab1:
    st.subheader("📜 Detection History")
    if "history" in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
    else:
        st.write("No detection history yet.")

# Tab 2: Data Insights
with tab2:
    st.subheader("📈 Data Insights")
    st.write("✅ **Total Records:**", dataset.shape[0])
    st.write("🌍 **Unique Languages:**", dataset["language"].nunique())
    st.write(f"🏆 **Most Common Language:** {dataset['language'].mode()[0]}")

    # Language frequency distribution
    st.bar_chart(dataset["language"].value_counts())

    # Scatter Plot with User Input Highlight
    st.subheader("📊 Language Scatter Plot")

    fig, ax = plt.subplots()

    # Map each language to a unique position
    lang_map = {lang: i for i, lang in enumerate(dataset["language"].unique())}
    lang_positions = dataset["language"].map(lang_map)

    # Plot dataset points
    ax.scatter(range(len(dataset)), lang_positions, color='blue', label="Dataset")

    # Highlight user input in scatter plot
    if user_input:
        detected_input_lang = detect_language_knn(user_input, knn_model, tfidf_vectorizer, label_encoder)
        if detected_input_lang in lang_map:
            ax.scatter(len(dataset), lang_map.get(detected_input_lang, -1), color='red', label="User Input")

    ax.set_xlabel("Index")
    ax.set_ylabel("Languages")
    ax.set_title("Language Scatter Plot")
    ax.set_yticks(list(lang_map.values()))
    ax.set_yticklabels(list(lang_map.keys()))
    ax.legend()
    st.pyplot(fig)

# Tab 3: Dataset
with tab3:
    st.subheader("📂 Current Dataset")
    st.dataframe(balanced_dataset)
