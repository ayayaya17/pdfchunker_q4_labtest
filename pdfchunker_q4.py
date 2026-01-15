import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

# -----------------------------
# NLTK downloads (fix punkt_tab issue)
# -----------------------------
def ensure_nltk():
    # download punkt + punkt_tab (some versions require both)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

ensure_nltk()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Q4 PDF Sentence Chunking (NLTK)", layout="wide")
st.title("Q4: PDF Sentence Chunking using NLTK")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Step 1: Read PDF
        reader = PdfReader(uploaded_file)

        # Step 2: Extract text
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        full_text = full_text.strip()

        st.subheader("Basic Info")
        st.write("Pages:", len(reader.pages))
        st.write("Characters extracted:", len(full_text))

        if not full_text:
            st.warning("No text extracted. This PDF might be scanned (image-based).")
        else:
            # Step 3: Split into sentences
            sentences = sent_tokenize(full_text)
            st.success(f"Total sentences detected: {len(sentences)}")

            # Display sample indices 58 to 68
            st.subheader("Step 3: Display sentences index 58 to 68")
            start_i = 58
            end_i = 68

            if len(sentences) <= start_i:
                st.warning(f"This PDF only has {len(sentences)} sentences. Cannot show indices 58–68.")
            else:
                end_i = min(end_i, len(sentences) - 1)

                for i in range(start_i, end_i + 1):
                    st.markdown(f"**[{i}]** {sentences[i]}")

                # Step 4: Semantic chunking (each sentence = 1 chunk)
                st.subheader("Step 4: Semantic Sentence Chunking (NLTK)")
                st.write("Each sentence below is treated as one semantic chunk:")

                for i in range(start_i, end_i + 1):
                    st.info(f"Chunk [{i}]: {sentences[i]}")

            with st.expander("Show extracted text (first 2000 chars)"):
                st.text(full_text[:2000])

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
else:
    st.info("Please upload a PDF to begin.")
