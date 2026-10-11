import os
import fitz  # PyMuPDF for reading PDFs
from dotenv import load_dotenv
import streamlit as st
from google import genai

# Load environment variables from a .env file if present
load_dotenv()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Research Paper Summarizer", page_icon="📄", layout="wide"
)

st.title("📄 Research Paper Summarizer")
st.write(
    "Upload a research paper in PDF format to instantly extract authors, dates,"
    " main topics, abstract highlights, experiment results, and conclusions."
)

# --- API Key Management (Sidebar) ---
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help=(
        "Enter your API key here, or leave blank to use a .env file or system"
        " environment variable."
    ),
)

# Determine the API key to use (Sidebar input takes precedence, then environment/dotenv)
api_key = api_key_input or os.getenv("GEMINI_API_KEY")

if not api_key:
  st.warning(
      "⚠️ Please enter your Gemini API Key in the sidebar or configure your"
      " `.env` file to proceed."
  )
  st.stop()

# Initialize the Gemini client dynamically with the provided key
client = genai.Client(api_key=api_key)


def extract_text_from_pdf(uploaded_file):
  """Extracts text from an uploaded PDF file."""
  doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
  text = ""
  for page in doc:
    text += page.get_text()
  return text


def summarize_research_paper(paper_text):
  """Sends the paper text to Gemini to extract structured summary components."""
  prompt = f"""
    You are an expert research assistant. Analyze the following research paper text and provide a structured summary containing these exact sections:
    
    - **Main Topic**: The primary research focus or domain.
    - **Authors**: Names of the authors (if available in the text).
    - **Date**: Publication date or year.
    - **Abstract Summary**: Key takeaways from the abstract.
    - **Experiment Results**: The primary findings, metrics, or experimental outcomes.
    - **Conclusion**: The ultimate conclusion and future work proposed by the authors.

    Keep the summary clear, professional, and well-structured using markdown.

    Paper Text (truncated if too long):
    {paper_text[:35000]}
    """

  response = client.models.generate_content(
      model="gemini-3.8-flash",
      contents=prompt,
  )
  return response.text


# File uploader widget
uploaded_file = st.file_uploader(
    "Upload your research paper (PDF)", type=["pdf"]
)

if uploaded_file is not None:
  with st.spinner("Extracting text and analyzing the paper with Gemini..."):
    try:
      # Step 1: Extract text from PDF
      paper_text = extract_text_from_pdf(uploaded_file)

      if not paper_text.strip():
        st.error(
            "Could not extract any text from this PDF. It might be scanned or"
            " image-based."
        )
      else:
        # Step 2: Generate Summary
        summary = summarize_research_paper(paper_text)

        # Step 3: Display Results
        st.success("Analysis Complete!")
        st.markdown("---")
        st.markdown(summary)

    except Exception as e:
      st.error(f"An error occurred during processing: {e}")