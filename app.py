import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain
from utils import get_timestamp, load_config, get_avatar
from image_handler import handle_image
from pdf_handler import add_documents_to_db, url_documents_to_db

from html_templates import css
from database_operations import load_last_k_text_messages, save_text_message, save_image_message, load_messages, get_all_chat_history_ids, delete_chat_history, init_db
import sqlite3
from datetime import datetime

config = load_config()

@st.cache_resource
def load_chain():
    if st.session_state.pdf_chat:
        print("loading pdf chat chain....")
        return load_pdf_chat_chain()
    return load_normal_chain()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def parse_datetime(session_id):
    try:
        return datetime.strptime(session_id, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.min
      
def main():
    init_db()
    st.title("Chat with multiple PDFs and Images")
    st.write(css, unsafe_allow_html=True)
    
    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.uploaded_file_key = 0
        st.session_state.file_processed = False
        st.session_state.url_processed = False 
        st.session_state.pdf_chat = False  # Initialize pdf_chat state
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + sorted(get_all_chat_history_ids(), key=parse_datetime, reverse=True)

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    file_chat_toggle_col, _ = st.sidebar.columns([2, 1])
    file_chat_toggle_col.checkbox("Chat with File", key="pdf_chat", value=False, on_change=toggle_pdf_chat)
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)
    clear_cache_col.button("Clear Cache", on_click=clear_cache)

    chat_container = st.container()
    user_input = st.chat_input("Type your message here", key="user_input")
    uploaded_files = st.sidebar.file_uploader("Upload files (PDFs, images, docs)", type=["pdf", "jpg", "jpeg", "png", "doc", "docx"], accept_multiple_files=True, key="uploaded_files")
    input_urls = st.sidebar.text_input("Enter URLs ")
    url_list = input_urls.split('\n') if input_urls else []

    if uploaded_files and not st.session_state.file_processed: 
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing file {uploaded_file.name}..."):
                if uploaded_file.type.startswith('image/'):
                    if user_input:
                        answer = handle_image(uploaded_file.getvalue(), user_input)
                        st.session_state.file_processed = True
                        save_image_message(get_session_key(), "human", uploaded_file.getvalue())
                        save_text_message(get_session_key(), "human", user_input)
                        save_text_message(get_session_key(), "ai", answer) 
                    else:
                        handle_image(uploaded_file.getvalue(), user_input)
                        st.session_state.file_processed = True
                        save_image_message(get_session_key(), "human", uploaded_file.getvalue())
                elif st.session_state.pdf_chat and uploaded_file.type == "application/pdf":
                    add_documents_to_db([uploaded_file])
                    st.session_state.file_processed = True 
                    st.sidebar.success("Done processing")
                else:
                    st.error("Unsupported file format.")

                st.session_state.uploaded_file_key += 1

    if input_urls and not st.session_state.url_processed: 
        for url in url_list:
            if st.session_state.pdf_chat:
                with st.spinner(f"Processing given url {url}..."):
                    url_documents_to_db(url)
                    st.session_state.url_processed = True 
                    st.sidebar.success("Done processing")
                st.session_state.uploaded_file_key += 1

    if user_input:
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input=user_input, 
                                   chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
        save_text_message(get_session_key(), "human", user_input)
        save_text_message(get_session_key(), "ai", llm_answer)
        user_input = None

    if st.session_state.session_key != "new_session" or st.session_state.new_session_key is not None:
        with chat_container:
            chat_history_messages = load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    elif message["message_type"] == "image":
                        st.image(message["content"])

        if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
            st.rerun()

if __name__ == "__main__":
    main()
