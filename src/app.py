import streamlit as st
from rag_pipelline import (
    extract_text_from_pdf,
    split_text_into_chunks,
    create_vector_store,
    create_rag_agent,
    get_answer
    )


# Page Config-----
st.set_page_config(
    page_title = "PDF Chatbot- using RAG",
    page_icon = "📄",
    layout = "wide"
)

# Header-----
st.markdown("### 📄 PDF Chatbot - RAG + Gemini")
st.markdown("Powered by Langchain and Gemini 2.5 Flash")
st.divider()

# Session State -----
if "agent" not in st.session_state:
    st.session_state.agent = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# Sidebar ----

with st.sidebar:
    st.header("⚙️ Stack Info")
    st.markdown("**Framework:** Langchain 1.2.10")
    st.markdown("**LLM:** Gemini 2.5 Flash")
    st.markdown("**Embeddings:** Google embedding-001")
    st.markdown("**Vector Store:** FAISS")
    st.divider()

    st.header("📁 Upload or Select PDF")
    # Upload a PDF
    uploaded_file = st.file_uploader(
        "Upload a PDF", 
        type = ["pdf"],
        help = "Max 10 MB · Max 50 pages · Must have selectable text (not scanned)"
        )
    
    # Select a sample PDF
    sample_pdf = st.selectbox(
        "Or pick a sample PDF:",
        ["None", "Attention is All You Need", "2025 ICC Champions Trophy-Wikipedia.pdf"]
    )

    # Ensure only one PDF is uploaded at a time
    chosen_file , chosen_name = None,""
    if uploaded_file is not None:
        chosen_file = uploaded_file
        chosen_name = uploaded_file.name
    elif sample_pdf != "None":
        sample_map = {
            "Attention is All You Need": "src/sample_pdf/Attention_is_all_you_need.pdf",
            "2025 ICC Champions Trophy-Wikipedia.pdf":"src/sample_pdf/2025_ICC_Champions_Trophy-Wikipedia.pdf",
        }
        # Using a variable and closing after use
        sample_path = sample_map.get(sample_pdf)
        if sample_path:
            try:
                chosen_file = open(sample_path, "rb")
                chosen_name = sample_pdf
                st.info(f"📄 Using sample file: {chosen_name}")
            except FileNotFoundError:
                st.error(f"❌ Sample file not found: {sample_path}")
                chosen_file = None


    if chosen_file is not None:
        if st.button("Process PDF", type = "primary", use_container_width = True):
            with st.spinner("Step 1/4 - Extracting raw text"):
                raw_text = extract_text_from_pdf(chosen_file)
            
            # Close sample file after reading to avoid resource leak
            if sample_pdf != "None" and hasattr(chosen_file, "close"):
                chosen_file.close()

            
            if not raw_text.strip():
                st.error("❌ No text found, please check your PDF and confirm its text selectable")
            
            else:
                with st.spinner("Step 2/4 - Splitting text into chunks"):
                    chunks = split_text_into_chunks(raw_text)

                with st.spinner("Step 3/4 - Creating embedding and vector store"):
                    vector_store = create_vector_store(chunks)
                
                with st.spinner("Step 4/4 - Creating RAG Agent"):
                    st.session_state.agent = create_rag_agent(vector_store)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = chosen_name
                    st.session_state.chat_history = []
                    st.session_state.display_messages = []
                
                st.success(f"✅ Ready! {len(chunks)} chunks indexed")
    
    if st.session_state.pdf_processed:
        st.divider()
        st.success(f" Active :\n{st.session_state.pdf_name}")
        st.caption(f"Messages so far:{len(st.session_state.display_messages)}")

        if st.button("Clear & Reset", use_container_width= True):
            st.session_state.agent = None
            st.session_state.chat_history = []
            st.session_state.display_messages = []
            st.session_state.pdf_processed = False
            st.session_state.pdf_name = ""
            st.rerun()

# Main Area -----
if not st.session_state.pdf_processed:
    st.markdown("### How to use")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("Step 1 - Upload or select the PDF from sidebar")

    with col2:
        st.markdown("Step 2 - Click Process PDF")
    
    with col3:
        st.markdown("Step 3 - Ask your questions in the chat box")
    
    st.divider()

else:
    st.markdown(f"### Chatting with {st.session_state.pdf_name}")

    # Display all previous messages
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # Show source chunks for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(" PDF Chunks used to generate this answer"):
                    for i, doc in enumerate(msg["sources"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(f"> {doc.page_content[:400]}...")
                        st.divider()


#Chat Input

if st.session_state.pdf_processed:
    user_question = st.chat_input(f"Ask Something about {st.session_state.pdf_name}...")

    if user_question:

        # Show user message
        with st.chat_message("user"):
            st.write(user_question)

        # Store in both histories
        st.session_state.chat_history.append({
            "role":"user",
            "content":user_question
        })
        st.session_state.display_messages.append({
            "role": "user",
            "content": user_question
        })

        # Get answer from agent
        with st.chat_message("assistant"):
            with st.spinner("Agent is searching PDF and thinking"):
                answer, source_docs = get_answer(
                    st.session_state.agent,
                    user_question,
                    st.session_state.chat_history[:-1] # history without current question

                )
            st.write(answer)

            if source_docs:
                with st.expander(" PDF chunks used to generate this answer"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(f"> {doc.page_content[:400]}...")
        
        #Store assistant response
        st.session_state.chat_history.append({
            "role":"assistant",
            "content" : answer
        })
        st.session_state.display_messages.append({
            "role": "assistant",
            "content": answer,
            "sources":source_docs
        })

