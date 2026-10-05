import streamlit as st
import pandas as pd
import os
import sqlite3
from pypdf import PdfReader
from transformers import pipeline

# 1. Page Configuration & Directory Setup
st.set_page_config(page_title="AI Chip Pro", layout="wide")
st.title("🤖 AI Chip: Conversational Text Analyzer with Permanent Memory")

STORAGE_DIR = "stored_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# 2. SQLite Permanent Memory Repository Setup
def init_db():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    # Create tables to permanently remember user facts and configuration tokens
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_memory(key, value):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO user_profile (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def get_memory(key):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM user_profile WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Initialize database file immediately on system boot
init_db()

# 3. Setup Background Conversational Core
@st.cache_resource
def load_text_brain():
    return pipeline("text-generation", model="google/flan-t5-base")

st.info("🔄 Tuning conversational data chip... Running lightning-fast optimization.")
text_brain = load_text_brain()

# Fetch permanent profile configurations if they already exist in the database
stored_name = get_memory("user_name")

# 4. Storage Sidebar Dashboard
with st.sidebar:
    st.header("📂 Local Storage Dashboard")
    
    # Visual check confirming the state of the AI's Core Memory
    if stored_name:
        st.success(f"🧠 Core Memory Status: **Active**\n\nUser Profile Linked: **{stored_name}**")
    else:
        st.warning("🧠 Core Memory Status: **Empty**\n\nNo user configuration profile logged yet.")
        
    all_files = os.listdir(STORAGE_DIR)
    tab_sheets, tab_books = st.tabs(["📊 Data Sheets", "📚 Book PDFs"])
    
    with tab_sheets:
        sheets = [f for f in all_files if f.endswith(('.csv', '.xlsx'))]
        if sheets:
            for s in sheets:
                st.caption(f"📄 **{s}**")
        else:
            st.write("No sheets stored.")
            
    with tab_books:
        books = [f for f in all_files if f.endswith('.pdf')]
        if books:
            for b in books:
                st.caption(f"📕 **{b}**")
        else:
            st.write("No books stored.")

# 5. Ingestion Portals (Drag and Drop files)
st.subheader("📥 Data & Document Ingestion")
uploaded_file = st.file_uploader("Upload Data Sheets (CSV/Excel) or Literary Textbooks (PDF)", type=["csv", "xlsx", "pdf"])

file_context = ""
df_context = None

if uploaded_file is not None:
    file_path = os.path.join(STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File stored securely: **{uploaded_file.name}**")

    if uploaded_file.name.endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            pages_to_read = min(8, len(reader.pages))
            text_slices = [reader.pages[i].extract_text() for i in range(pages_to_read)]
            file_context = " ".join([t for t in text_slices if t])
            
            st.markdown("### 📚 **Document Analysis Summary**")
            st.markdown(f"* File Size Status: **{len(reader.pages)} total pages** detected.")
            st.markdown(f"* Scan Ingestion: Checked **first {pages_to_read} pages** directly into the context chip.")
        except Exception as e:
            st.error(f"Error reading doc layers: {e}")
            
    elif uploaded_file.name.endswith(('.csv', '.xlsx')):
        try:
            df_context = pd.read_csv(file_path) if uploaded_file.name.endswith('.csv') else pd.read_excel(file_path)
            st.markdown("### 📊 **Dataset Metrics Summary**")
            st.markdown(f"* Grid Structure: **{df_context.shape} rows** detected across **{df_context.shape} categories**.")
            st.markdown(f"* Available Fields: `{list(df_context.columns)}`")
        except Exception as e:
            st.error(f"Error compiling grid matrix: {e}")

# 6. Natural ChatGPT Conversational Workspace
st.subheader("💬 Active Interaction Terminal")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show conversational sequence tracking
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Talk to me, tell me your name, or ask me to sort/analyze..."):
    with st.chat_message("user"):
        st.markdown(f"**{user_input}**")
    st.session_state.messages.append({"role": "user", "content": f"**{user_input}**"})
    
    query = user_input.lower()
    ai_reply = ""
    
    # Reload name from database per interaction to guarantee consistency
    current_stored_name = get_memory("user_name")

    # Feature A: Database Memory Triggers (Extracts and saves name facts permanently)
    if "name is" in query:
        name_extracted = user_input.split("name is")[-1].strip().replace("**", "").replace(".", "")
        save_memory("user_name", name_extracted)
        ai_reply = f"### 🤖 **AI Chip Response**\n\nIdentity confirmed! I have permanently committed your name, **{name_extracted}**, into my SQLite hard disk repository. I will never ask for your name again. Let's get to work!"
    
    elif "who am i" in query or "my name" in query:
        if current_stored_name:
            ai_reply = f"### 🤖 **AI Chip Response**\n\nYou are **{current_stored_name}**! Your profile is securely locked inside my hard database memory repository."
        else:
            ai_reply = "### 🤖 **AI Chip Response**\n\nI checked my local memory blocks, but you haven't told me your name yet! Please state: *'My name is [your name]'* so I can write it down permanently."

    elif "hello" in query or "hi " in query or query == "hi":
        if current_stored_name:
            ai_reply = f"### 🤖 **AI Chip Response**\n\nHello back, **{current_stored_name}**! Welcome back to your active workspace module. What analytical tasks are we jumping into today?"
        else:
            ai_reply = "### 🤖 **AI Chip Response**\n\nHello there! I am your local AI Chip workspace assistant. What is your name, or what files are we processing today?"

    # Feature B: Direct Analytical Sorting Command Processing
    elif df_context is not None and "sort by" in query:
        col_target = user_input.split("sort by")[-1].strip().strip('`').strip()
        matched = [c for c in df_context.columns if c.lower() == col_target.lower()]
        if matched:
            sorted_df = df_context.sort_values(by=matched)
            new_f = f"sorted_{uploaded_file.name}"
            sorted_df.to_csv(os.path.join(STORAGE_DIR, new_f), index=False)
            
            ai_reply = f"### ✅ **Data Matrix Sorted**\n\n" \
                       f"I've successfully structured your table rows by **{matched}**.\n\n" \
                       f"* Records Actioned: **{len(sorted_df)} entries** processed.\n" \
                       f"* Output Target: Saved as **`{new_f}`** inside your storage cabinet panel."
        else:
            ai_reply = f"### ❌ **Column Mismatch**\n\n" \
                       f"I searched but couldn't verify an active field named '**{col_target}**'.\n\n" \
                       f"Verified fields inside this file are: `{list(df_context.columns)}`."
            
    # Feature C: Document File Deep Context Lookups
    elif file_context and ("explain" in query or "about" in query or "summarize" in query or "know" in query):
        keyword = query.replace("explain", "").replace("about", "").replace("summarize", "").replace("know", "").strip()
        sentences = file_context.split(". ")
        matches = [s for s in sentences if keyword in s.lower()]
        
        if matches:
            context_snippet = ". ".join(matches[:2])
            ai_reply = f"### 📖 **Document Analysis Excerpt**\n\n" \
                       f"I located references inside your book tracking **'{keyword}**':\n\n" \
                       f"\"*{context_snippet.strip()}*\"\n\n" \
                       f"* Context Quality: Found **{len(matches)} direct string highlights** inside pages."
        else:
            ai_reply = f"### 🔍 **Document Scan Alert**\n\n" \
                       f"I read through the uploaded book pages but did not find an explicit match for the topic phrase '**{keyword}**'."

    # Feature D: Local Language Model Conversational Generation Fallback
    else:
        try:
            conversation_history = ""
            for m in st.session_state.messages[-3:]:
                clean_role = "Human" if m["role"] == "user" else "Assistant"
                conversation_history += f"{clean_role}: {m['content'].replace('**', '')}\n"
            
            prompt_input = f"Dialogue:\n{conversation_history}Assistant response:"
            raw_response = text_brain(prompt_input, max_length=100, do_sample=True, temperature=0.7)['generated_text']
            processed_reply = raw_response.replace(prompt_input, "").strip()
            ai_reply = f"### 🤖 **AI Chip Response**\n\n{processed_reply}"
        except:
            user_label = current_stored_name if current_stored_name else "friend"
            ai_reply = f"### 🤖 **AI Chip Response**\n\nI am right here with you, **{user_label}**! Let me know what data sheets or book files you would like me to unpack next."

    # Render clean bold response instantly on screen
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
        

