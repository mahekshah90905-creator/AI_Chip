import streamlit as st
import pandas as pd
import os
import base64
from pypdf import PdfReader
from transformers import pipeline
import plotly.express as px
from gtts import gTTS
import soundfile as sf
from scipy.signal import resample

# 1. Page Configuration & Setup
st.set_page_config(page_title="AI Chip Pro", layout="wide")
st.title("🤖 AI Chip: Deep Voice & Advanced Text Analyzer")

STORAGE_DIR = "stored_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# 2. Setup Background Local AI Brain
@st.cache_resource
def load_conversational_brain():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium", pad_token_id=50256)

st.info("🔄 Optimizing localized speech and deep text brains... Please wait a moment.")
chat_brain = load_conversational_brain()

# Helper function to generate deep male text-to-speech audio locally using scipy
def speak_text_deep_male(text_to_speak):
    try:
        clean_text = text_to_speak.replace("📊", "").replace("📖", "").replace("**", "").replace("🔍", "").strip() 
        if clean_text:
            # 1. Generate base voice audio
            tts = gTTS(text=clean_text, lang='en', tld='co.uk')
            raw_file = "raw_speech.mp3"
            deep_file = "deep_speech.wav"
            tts.save(raw_file)
            
            # 2. Read data and shift the audio pitch down locally to create a deep male effect
            data, sample_rate = sf.read(raw_file)
            
            # Lowering the sample rate changes the playback speed and deepens the register
            pitch_factor = 0.78
            new_num_samples = int(len(data) * (1.0 / pitch_factor))
            deep_data = resample(data, new_num_samples)
            
            # Save the processed deep audio wave data file
            sf.write(deep_file, deep_data, sample_rate)
            
            # 3. Embed the deep voice audio player directly below the text block
            with open(deep_file, "rb") as f:
                audio_bytes = f.read()
                b64 = base64.b64encode(audio_bytes).decode()
                md = f"""
                    <audio autoplay="true" controls style="width: 100%; margin-top: 10px;">
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Voice engine notification: {e}")

# 3. Advanced Tab Storage Manager
with st.sidebar:
    st.header("📂 Local Storage Dashboard")
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

# 4. File Drag and Drop Interface
st.subheader("📥 Data & Document Ingestion")
uploaded_file = st.file_uploader("Upload Data Sheets (CSV/Excel) or Literary Textbooks (PDF)", type=["csv", "xlsx", "pdf"])

file_context = ""
df_context = None

if uploaded_file is not None:
    file_path = os.path.join(STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File stored securely in dashboard memory: **{uploaded_file.name}**")

    if uploaded_file.name.endswith('.pdf'):
        try:
            reader = PdfReader(file_path)
            pages_to_read = min(8, len(reader.pages))
            text_slices = [reader.pages[i].extract_text() for i in range(pages_to_read)]
            file_context = " ".join([t for t in text_slices if t])
            
            st.markdown("### 📚 **Document Analysis Summary**")
            st.markdown(f"* Total Length: **{len(reader.pages)} pages** detected inside file.")
            st.markdown(f"* Scan Status: **First {pages_to_read} pages** successfully processed into text layers.")
            st.markdown("💡 *You can now use the Chat workspace below to search for specific topics or request deep text breakdowns.*")
        except Exception as e:
            st.error(f"Error parsing document: {e}")
            
    elif uploaded_file.name.endswith(('.csv', '.xlsx')):
        try:
            df_context = pd.read_csv(file_path) if uploaded_file.name.endswith('.csv') else pd.read_excel(file_path)
            
            st.markdown("### 📊 **Dataset Metrics Summary**")
            st.markdown(f"* Dataset Structure: Found a total of **{df_context.shape[0]} rows** across **{df_context.shape[1]} data fields**.")
            st.markdown(f"* Columns List: `{list(df_context.columns)}`")
            
            # Interactive Graphics Engine
            st.subheader("📊 Instant Visual Data Explorer")
            numeric_cols = df_context.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 1:
                x_axis = st.selectbox("Select X-Axis data label:", df_context.columns.tolist())
                y_axis = st.selectbox("Select Y-Axis numeric value:", numeric_cols)
                fig = px.bar(df_context, x=x_axis, y=y_axis, title=f"{y_axis} distribution grouped by {x_axis}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error compiling visual analytics: {e}")

# 5. Natural Conversational Workspace with Enhanced Visual Layouts
st.subheader("💬 Chat with AI Chip Pro")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display current chat stream with clear formatting anchors
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me to analyze your text context, sort columns, or let's just chat..."):
    with st.chat_message("user"):
        st.markdown(f"**{user_input}**")
    st.session_state.messages.append({"role": "user", "content": f"**{user_input}**"})
    
    query = user_input.lower()
    ai_reply = ""

    # Sort logic framework with bold text structures
    if df_context is not None and "sort by" in query:
        col_target = user_input.split("sort by")[-1].strip().strip('`').strip()
        matched = [c for c in df_context.columns if c.lower() == col_target.lower()]
        if matched:
            sorted_df = df_context.sort_values(by=matched)
            new_f = f"sorted_{uploaded_file.name}"
            sorted_df.to_csv(os.path.join(STORAGE_DIR, new_f), index=False)
            
            ai_reply = f"### ✅ **Data Sorting Operation Complete**\n\n" \
                       f"I have thoroughly processed the spreadsheet and sorted all tracking profiles by the requested column: **{matched}**.\n\n" \
                       f"* Target Rows Sorted: **{len(sorted_df)} entries** processed.\n" \
                       f"* Output Saved As: **`{new_f}`**\n\n" \
                       f"You can access the updated file directly inside your sidebar storage tab."
        else:
            ai_reply = f"### ❌ **Column Matching Error**\n\n" \
                       f"I scanned your dataset fields but could not locate an exact match for '**{col_target}**'.\n\n" \
                       f"Please select from the following verified fields: `{list(df_context.columns)}`."
            
    # Deep analytical document text search with punchy summary layouts
    elif file_context and ("explain" in query or "about" in query or "summarize" in query or "know" in query):
        keyword = query.replace("explain", "").replace("about", "").replace("summarize", "").replace("know", "").strip()
        sentences = file_context.split(". ")
        matches = [s for s in sentences if keyword in s.lower()]
        
        if matches:
            context_snippet = ". ".join(matches[:3])
            ai_reply = f"### 📖 **Document Analysis Report**\n\n" \
                       f"Here are the specific findings extracted from the active text layer regarding your inquiry on **'{keyword}'**:\n\n" \
                       f"\"*{context_snippet.strip()}*\"\n\n" \
                       f"* Extraction Metrics: Compiled **{len(matches)} contextual reference blocks** from the text."
        else:
            ai_reply = f"### 🔍 **Topic Scan Notification**\n\n" \
                       f"I performed a full scan of the active document layers for key terms matching '**{keyword}**' but came up empty.\n\n" \
                       f"💡 *Try rephrasing your question or typing a broader keyword phrase.*"

    # Free conversational thinking fallback
    else:
        try:
            recent_chat = " ".join([m["content"] for m in st.session_state.messages[-2:]])
            generated_outputs = chat_brain(recent_chat, max_length=80, num_return_sequences=1)
            raw_reply = generated_outputs['generated_text'].replace(recent_chat, "").strip()
            
            if raw_reply:
                ai_reply = f"### 🤖 **AI Response**\n\n{raw_reply}"
            else:
                ai_reply = "### 🤖 **System Ready**\n\nI am logged in and completely operational! Drop in a book PDF or data sheet above, and let me know what analytical task you would like to run."
        except:
            ai_reply = "### 🤖 **System Ready**\n\nI am listening closely right here! Go ahead and drop in your data sheets or book files above, and we can explore them together."

    # Post processing output stream (Prints text response and pushes Deep Voice audio simultaneously)
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
        speak_text_deep_male(ai_reply)
        
