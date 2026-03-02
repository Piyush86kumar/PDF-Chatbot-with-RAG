PDF Chatbot with RAG
====================

Overview
--------
PDF Chatbot with RAG is a Streamlit-powered demo that lets you ask natural-language questions about any selectable-text PDF. The app takes a document, breaks it into overlapping chunks, embeds those chunks with Google Generative AI, and serves a LangChain agent that always consults retrieved context before responding. The goal is to keep answers concise, grounded, and easy for reviewers to follow without diving into the underlying code.

Demo
-----
🚀 Live Demo: View the interactive PDF Chatbot on [Hugging Face Spaces](https://huggingface.co/spaces/Piyush86/PDF-Chatbot-RAG)

How it works
------------

<p align="center">
  <img src="assets\System_architecture.png" alt="System Architecture" width="600">
</p>


1. A sidebar workflow handles file selection: upload your own PDF or choose one of the curated samples that live in `sample_pdf/`.
2. Once the document is confirmed, `rag_pipelline.py` extracts text with `PyPDF2`, splits it into 2,500-character chunks, embeds each chunk with Gemini embeddings, and stores the vectors in FAISS in memory.
3. A LangChain agent built around the Gemini 2.5 Flash chat model uses a retrieval tool to fetch the most relevant chunks and streams answers back to the Streamlit chat interface.

Components
----------
- `app.py`: Streamlit UI, session-state management, and chat orchestration. The sidebar coordinates uploads, sample selection, and processing states while the main area renders the dialog and chunk-level sources.
- `rag_pipelline.py`: Text extraction, chunking, embedding, FAISS creation, agent building, and helper utilities for rate-limit handling and retries.
- `sample_pdf/`: A handful of ready-to-use PDFs (e.g., GPT-4 technical report) so you can explore the experience without providing your own document.
- `requirements.txt`: Pinned dependencies for Streamlit, LangChain, FAISS, Google Generative AI, and related helpers.
- `.env`: Holds `GOOGLE_API_KEY` (or other Google credentials) needed to call the embedding service.

Setup
-----
Clone the repository and configure the environment before launching the app.

1. **Prerequisites**
   - Install Python 3.12+ and keep it up to date.
   - Have a Google Cloud project with the Generative AI API enabled and a valid API key (or service account credentials).

2. **Environment**
   - Create a `.env` file at the project root.
   - Add your key:

     ```
     GOOGLE_API_KEY=your-generated-key
     ```

   - If you prefer service account credentials, set `GOOGLE_APPLICATION_CREDENTIALS` instead of `GOOGLE_API_KEY`.

3. **Dependencies**
   - Create and activate a virtual environment:

     ```
     python -m venv .venv
     .venv\\Scripts\\Activate.ps1   # PowerShell
     .venv\\Scripts\\activate.bat   # cmd.exe
     source .venv/bin/activate      # Bash
     ```

   - Install the pinned packages:

     ```
     pip install -r requirements.txt
     ```

4. **Launch**
   - Start the Streamlit app:

     ```
     streamlit run app.py
     ```

   - Upload a text-based PDF or select a sample from the sidebar, click **Process PDF**, and wait for the four spinner steps (extract → chunk → embed → agent).
   - Ask questions using the chat box once the processing completes.

Tips
----
- Keep questions focused so Gemini can stay concise and reuse the retrieved chunks that are shown in the expanders.
- Use the **Clear & Reset** button before switching documents to avoid leftover state.
- If embeddings hit rate limits, wait a minute—`rag_pipelline.py` already throttles calls, but the console also logs retries.

Next steps
----------
1. Persist FAISS to disk or a managed vector database if you need to reuse vector stores across sessions.
2. Add tests that cover chunk creation, embedding retries, and agent responses so you can refactor with confidence.
