import streamlit as st
import pandas as pd
import os
import base64
from pypdf import PdfReader
from transformers import pipeline
import plotly.express as px
from gtts import gTTS

# 1. Page Configuration & Setup
st.set_page_config(page_title="AI Chip Pro", layout="wide")
st.title("🤖 AI Chip: Advanced Local Data & Voice Assistant")

STORAGE_DIR = "stored_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# 2. Setup Background Local AI Brain
@st.cache_resource
def load_conversational_brain():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium", pad_token_id=50256)

st.info("🔄 Optimizing localized speech and text brains... Please wait a moment.")
# Helper function to generate and play audio text-to-speech safely online
def speak_text(text_to_speak):
    try:
        # CLEANED FIX: Corrected text cleaning syntax
        clean_text = text_to_speak.replace("📊", "").replace("📖", "").strip() 
        if clean_text:
            tts = gTTS(text=clean_text, lang='en')
            tts.save("speech.mp3")
            with open("speech.mp3", "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f"""
                    <audio autoplay="true" controls style="width: 100%; margin-top: 10px;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Voice engine notification: {e}")

    except Exception as e:
        st.error(f"Voice engine notification: {e}")

# 3. Advanced Tab Storage Manager (Suggestion 3)
with st.sidebar:
    st.header("📂 Local Storage Dashboard")
    all_files = os.listdir(STORAGE_DIR)
    
    tab_sheets, tab_books = st.tabs(["📊 Data Sheets", "📚 Book PDFs"])
    
    with tab_sheets:
        sheets = [f for f in all_files if f.endswith(('.csv', '.xlsx'))]
        if sheets:
            for s in sheets:
                st.caption(f"📄 {s}")
        else:
            st.write("No sheets stored.")
            
    with tab_books:
        books = [f for f in all_files if f.endswith('.pdf')]
        if books:
            for b in books:
                st.caption(f"📕 {b}")
        else:
            st.write("No books stored.")

# 4. File Drag and Drop Interface
st.subheader("📥 Data & Document Ingestion")
uploaded_file = st.file_uploader("Upload Data Sheets (CSV/Excel) or Literary Textbooks (PDF)", type=["csv", "xlsx", "pdf"])

file_context = ""
df_context = None

if uploaded_file is not None:
    file_path = os.path.join(STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File stored securely in dashboard memory: {uploaded_file.name}")

    if uploaded_file.name.endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            pages_to_read = min(5, len(reader.pages))
            text_slices = [reader.pages[i].extract_text() for i in range(pages_to_read)]
            file_context = " ".join([t for t in text_slices if t])
            st.info(f"📚 Book Parsed: {len(reader.pages)} pages detected. Context loaded successfully.")
        except Exception as e:
            st.error(f"Error parsing document: {e}")
            
    elif uploaded_file.name.endswith(('.csv', '.xlsx')):
        try:
            df_context = pd.read_csv(file_path) if uploaded_file.name.endswith('.csv') else pd.read_excel(file_path)
            st.info(f"📊 Dataset Active: Matrix shapes parsed to {df_context.shape[0]} rows across columns: {list(df_context.columns)}")
            
            # Interactive Graphics Engine (Suggestion 2)
            st.subheader("📊 Instant Visual Data Explorer")
            numeric_cols = df_context.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 1:
                x_axis = st.selectbox("Select X-Axis data label:", df_context.columns.tolist())
                y_axis = st.selectbox("Select Y-Axis numeric value:", numeric_cols)
                fig = px.bar(df_context, x=x_axis, y=y_axis, title=f"{y_axis} distribution grouped by {x_axis}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available to map visual analytical tracking metrics.")
        except Exception as e:
            st.error(f"Error compiling visual graphics: {e}")

# 5. Natural ChatGPT Conversational Workspace
st.subheader("💬 Chat with AI Chip Pro")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display current chat stream
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask me to analyze your text context, sort columns, or let's just chat..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    query = user_input.lower()
    ai_reply = ""

    # Sort logic framework
    if df_context is not None and "sort by" in query:
        col_target = user_input.split("sort by")[-1].strip().strip('`').strip()
        matched = [c for c in df_context.columns if c.lower() == col_target.lower()]
        if matched:
            sorted_df = df_context.sort_values(by=matched)
            new_f = f"sorted_{uploaded_file.name}"
            sorted_df.to_csv(os.path.join(STORAGE_DIR, new_f), index=False)
            ai_reply = f"Process Completed! I have sorted that spreadsheet matrix by column row `{matched}`. The output has been safely written as `{new_f}` inside our sidebar storage container."
        else:
            ai_reply = f"I scanned the dataframe but couldn't verify an exact key match for '{col_target}'. Available fields are: {list(df_context.columns)}"
            
    # Deep analytical text search 
    elif file_context and ("explain" in query or "about" in query or "summarize" in query):
        keyword = query.replace("explain", "").replace("about", "").replace("summarize", "").strip()
        sentences = file_context.split(". ")
        matches = [s for s in sentences if keyword in s.lower()]
        
        if matches:
            context_snippet = ". ".join(matches[:2])
            ai_reply = f"📖 Searching inside the document context for '{keyword}': '{context_snippet}'."
        else:
            ai_reply = f"🔍 I evaluated the active text layers for '{keyword}' but could not verify an explicit string snippet. Try typing another focus subject phrase."

    # Free conversational thinking fallback
    else:
        try:
            recent_chat = " ".join([m["content"] for m in st.session_state.messages[-2:]])
            generated_outputs = chat_brain(recent_chat, max_length=80, num_return_sequences=1)
            ai_reply = generated_outputs['generated_text'].replace(recent_chat, "").strip()
            
            if not ai_reply:
                ai_reply = "I'm logged in and operational! Upload a file or tell me what analytical operations you'd like to perform."
        except:
            ai_reply = "I'm listening closely right here! Drop in your book or dataset above, and we can explore it together."

    # Post processing output stream (Prints text response and pushes Voice engine audio simultaneously)
    with st.chat_message("assistant"):
        st.write(ai_reply)
        speak_text(ai_reply) # Text-to-Speech Engine (Suggestion 1)
        
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
