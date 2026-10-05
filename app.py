import streamlit as st
import pandas as pd
import os
from pypdf import PdfReader

# 1. Setup Layout
st.set_page_config(page_title="Local AI Data & Book Assistant", layout="wide")
st.title("📟 Local Data Storage & Book Analyzer (No Internet Required)")

STORAGE_DIR = "stored_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Sidebar view for saved items
with st.sidebar:
    st.header("📂 Local Storage Cabinet")
    stored_files = os.listdir(STORAGE_DIR)
    if stored_files:
        for file in stored_files:
            st.write(f"📁 {file}")
    else:
        st.write("Storage is empty.")

# 2. File Uploader supporting Spreadsheets & PDFs
st.subheader("📥 Drop your Data Sheets or Book PDFs here")
uploaded_file = st.file_uploader("Upload CSV, Excel, or PDF", type=["csv", "xlsx", "pdf"])

file_text_content = ""
df_context = None

if uploaded_file is not None:
    file_path = os.path.join(STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Successfully secured in storage: {uploaded_file.name}")

    # Process based on file type
    if uploaded_file.name.endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            # Extract text from the first few pages for local processing
            full_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text.append(text)
            file_text_content = "\n".join(full_text)
            st.info(f"✨ Read completed! Extracted {len(reader.pages)} pages from the book.")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            
    elif uploaded_file.name.endswith(('.csv', '.xlsx')):
        try:
            if uploaded_file.name.endswith('.csv'):
                df_context = pd.read_csv(file_path)
            else:
                df_context = pd.read_excel(file_path)
            st.info(f"📊 Dataset loaded. Found columns: {list(df_context.columns)}")
        except Exception as e:
            st.error(f"Error loading table: {e}")

# 3. Local Interaction Box (Simulated Chat Interface)
st.subheader("💬 Command Console")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

if user_input := st.chat_input("Ask me to look for a topic, sort a sheet, or summarize..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    response = ""
    query = user_input.lower()

    # Scenario A: Processing a Data Spreadsheet
    if df_context is not None:
        if "sort by" in query:
            col_target = user_input.split("sort by")[-1].strip().strip('`').strip()
            matched_cols = [c for c in df_context.columns if c.lower() == col_target.lower()]
            
            if matched_cols:
                sorted_df = df_context.sort_values(by=matched_cols[0])
                new_file = f"sorted_{uploaded_file.name}"
                sorted_df.to_csv(os.path.join(STORAGE_DIR, new_file), index=False)
                response = f"✅ I have sorted your rows by `{matched_cols[0]}` and automatically generated a new file for you: `{new_file}` inside your storage dashboard."
            else:
                response = f"I couldn't find a exact match for column '{col_target}'. Available choices are: {list(df_context.columns)}"
        else:
            response = f"📊 Data Summary Request:\nThis dataset holds {len(df_context)} rows across columns: {list(df_context.columns)}. Try telling me to 'sort by [column name]'!"

    # Scenario B: Analyzing a Book PDF locally
    elif file_text_content:
        # Local rule-based search engine to look inside the book text safely
        if "know about" in query or "find" in query or "explain" in query:
            # Extract topic term
            topic = query.replace("know about", "").replace("find", "").replace("explain", "").strip()
            
            # Simple offline string extraction matching your exact paragraphs
            paragraphs = file_text_content.split("\n")
            matches = [p for p in paragraphs if topic in p.lower()]
            
            if matches:
                summary_snippet = "\n\n• ".join(matches[:4]) # Give up to 4 matched contextual sentences
                response = f"📖 Here is what I found regarding **'{topic}'** inside your uploaded text:\n\n• {summary_snippet}"
            else:
                response = f"🔍 I scanned the entire text context but couldn't find explicit sentences containing '{topic}'. Try checking your spelling or look for another keyword!"
        else:
            response = f"📚 Your file is uploaded and safely stored offline. Type: 'explain [topic name]' or 'know about [topic]' to filter details from the book pages instantly."
    
    else:
        response = "Please drag/drop a spreadsheet or book PDF above first so I have data to store and process for you!"

    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
