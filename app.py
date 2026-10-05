import streamlit as st
import pandas as pd
import openai
import os
import json

# 1. Setup App Layout and Directory
st.set_page_config(page_title="AI Data Analyst App", layout="wide")
st.title("🤖 AI Data Analysis & Storage Assistant")

STORAGE_DIR = "stored_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# 2. API Key Management 
# Securely prompt for the API key in the sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.header("📂 Stored Files")
    stored_files = os.listdir(STORAGE_DIR)
    if stored_files:
        for file in stored_files:
            st.write(f"📄 {file}")
    else:
        st.write("No files saved yet.")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to start!")
    st.stop()

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# 3. File Uploading Component
st.subheader("📤 Upload Data File")
uploaded_file = st.file_uploader("Choose a CSV or Excel file to analyze and store", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Save file locally to storage directory
    file_path = os.path.join(STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Successfully stored and loaded: {uploaded_file.name}")

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a data analysis assistant. You can read, sort, and analyze datasets, and help users save new files."}
    ]

# Display older chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# 5. Chat Interface and Logic
if user_command := st.chat_input("Ask me to sort, analyze, or process your data..."):
    # Display user query
    with st.chat_message("user"):
        st.write(user_command)
    st.session_state.messages.append({"role": "user", "content": user_command})

    # Read the data context if a file exists
    data_context = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Send a summary snippet of the data structure to the AI
            data_context = f"\n\nActive File: {uploaded_file.name}\nColumns: {list(df.columns)}\nData Preview (First 3 rows):\n{df.head(3).to_string()}"
        except Exception as e:
            data_context = f"\n\n(Error reading file: {str(e)})"

    # Append context dynamically to the prompt for the AI
    ai_prompt = st.session_state.messages + [{"role": "system", "content": f"Use this data context if needed: {data_context}"}]

    # Generate response from OpenAI
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # Check for explicitly requested operations like sorting
            if "sort by" in user_command.lower() and uploaded_file is not None:
                # Basic automated sorting capability example
                col_to_sort = user_command.lower().split("sort by")[-1].strip()
                # Clean up punctuation
                col_to_sort = ''.join(e for e in col_to_sort if e.isalnum() or e == '_')
                
                # Match column regardless of case
                matched_col = [c for c in df.columns if c.lower() == col_to_sort.lower()]
                if matched_col:
                    sorted_df = df.sort_values(by=matched_col[0])
                    new_filename = f"sorted_{uploaded_file.name}"
                    new_filepath = os.path.join(STORAGE_DIR, new_filename)
                    
                    if new_filename.endswith('.csv'):
                        sorted_df.to_csv(new_filepath, index=False)
                    else:
                        sorted_df.to_excel(new_filepath, index=False)
                        
                    ai_response = f"I have successfully **sorted** the data by `{matched_col[0]}` and created a new file for you: `{new_filename}` inside your storage cabinet!"
                else:
                    ai_response = f"I tried to sort the data, but I couldn't find a column named '{col_to_sort}'. Available columns are: {list(df.columns)}"
            
            else:
                # Regular ChatGPT response logic for general analysis and answering questions
                completion = client.chat.completions.create(
                    model="gpt-4o-mini", # Lightweight, cost-efficient model
                    messages=ai_prompt
                )
                ai_response = completion.choices[0].message.content

            response_placeholder.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
