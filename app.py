import streamlit as st
import pandas as pd
import os
from pypdf import PdfReader
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="AI Chip Assistant", layout="wide")
st.title("🤖 AI Chip: Local Data & Conversational Assistant")

STORAGE_DIR = "stored_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Sidebar for Storage View
with st.sidebar:
    st.header("📁 Local Data Cabinet")
    stored_files = os.listdir(STORAGE_DIR)
    if stored_files:
        for file in stored_files:
            st.write(f"📄 {file}")
    else:
        st.write("No files saved yet.")

# 2. Download/Cache Local Conversational Model
@st.cache_resource
def load_conversational_brain():
    # Loads a free, lightweight conversational model that runs locally on the server
    return pipeline("text-generation", model="microsoft/DialoGPT-medium", pad_token_id=50256)

st.info("🔄 Waking up local conversational core... (This may take a moment on first boot)")
chat_brain = load_conversational_brain()

# 3. File Processing Component
st.subheader("📥 Upload Data or Books")
uploaded_file = st.file_uploader("Drop a CSV, Excel, or Book PDF here", type=["csv", "xlsx", "pdf"])

file_context = ""
df_context = None

if uploaded_file is not None:
    file_path = os.path.join(STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Stored securely: {uploaded_file.name}")

    if uploaded_file.name.endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            # Gather text from first 5 pages to keep processing swift
            pages_to_read = min(5, len(reader.pages))
            text_slices = [reader.pages[i].extract_text() for i in range(pages_to_read)]
            file_context = " ".join([t for t in text_slices if t])
            st.info(f"📖 Loaded book context ({pages_to_read} pages extracted). Ready to analyze.")
        except Exception as e:
            st.error(f"Error processing book text: {e}")
            
    elif uploaded_file.name.endswith(('.csv', '.xlsx')):
        try:
            df_context = pd.read_csv(file_path) if uploaded_file.name.endswith('.csv') else pd.read_excel(file_path)
            st.info(f"📊 Dataset loaded successfully. Target columns: {list(df_context.columns)}")
        except Exception as e:
            st.error(f"Error loading matrix: {e}")

# 4. Human-like Chat Interface
st.subheader("💬 Chat with AI Chip")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display current chat stream
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Talk to me, ask me to sort data, or summarize a topic..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    query = user_input.lower()
    ai_reply = ""

    # Sort operation override
    if df_context is not None and "sort by" in query:
        col_target = user_input.split("sort by")[-1].strip().strip('`').strip()
        matched = [c for c in df_context.columns if c.lower() == col_target.lower()]
        if matched:
            sorted_df = df_context.sort_values(by=matched[0])
            new_f = f"sorted_{uploaded_file.name}"
            sorted_df.to_csv(os.path.join(STORAGE_DIR, new_f), index=False)
            ai_reply = f"I've gone ahead and sorted that dataset by `{matched[0]}` for you! I saved the result as a new file called `{new_f}` in our storage cabinet."
        else:
            ai_reply = f"I took a look, but I couldn't find a column matching '{col_target}'. The columns I can see are: {list(df_context.columns)}"
            
    # Document reading focus
    elif file_context and ("explain" in query or "about" in query or "summarize" in query):
        keyword = query.replace("explain", "").replace("about", "").replace("summarize", "").strip()
        sentences = file_context.split(". ")
        matches = [s for s in sentences if keyword in s.lower()]
        
        if matches:
            context_snippet = ". ".join(matches[:2])
            ai_reply = f"Looking at your uploaded file for details on '{keyword}': {context_snippet}."
        else:
            ai_reply = f"I scanned the document for '{keyword}' but didn't see an exact sentence match. Could you rephrase your question or specify another topic?"

    # Fallback to natural chat conversational intelligence
    else:
        try:
            # Build conversation prompt string from history
            recent_chat = " ".join([m["content"] for m in st.session_state.messages[-3:]])
            generated_outputs = chat_brain(recent_chat, max_length=100, num_return_sequences=1)
            ai_reply = generated_outputs[0]['generated_text'].replace(recent_chat, "").strip()
            
            # Fallback block if output generation returns empty
            if not ai_reply:
                ai_reply = "I hear you! I am ready to process your files or just chat. Let me know what you'd like to dive into next."
        except:
            ai_reply = "I'm right here listening! Upload a data sheet or a book PDF above, and I can extract explanations or sort rows for you."

    with st.chat_message("assistant"):
        st.write(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
