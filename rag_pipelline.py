import os
import re
import time
import random
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.chat_models import init_chat_model                     # new universal model initializer
from langchain.agents import create_agent                             # replaces AgentExecutor
from langchain.tools import tool                                      # tool decorator
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

# Constant values
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300
EMBED_DELAY_SECONDS = 1.5
TPM_SAFE_THRESHOLD = 27000
MAX_RETRIES = 3

# Extract Text from the PDF
def extract_text_from_pdf(pdf_file):
    """
    Reads each page of the PDF and extracts raw text
    Some page smay return None so we use ''or empty string' as safety
    """

    pdf_reader = PdfReader(pdf_file) 
    text  = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Split the text into Chunks
def split_text_into_chunks(raw_text):

    """
    split the text into chunks of 1000 characters to avoid hiting the token limits
    chunk_overlap = 200 
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Create Vector Store

def create_vector_store(text_chunks):
    """
    Embed chunks with Gemini and store in FAISS.
    Respects all three free tier limits:
        RPM = 100 → actual ~40 RPM (1.5s delay)         
        TPM = 30K → pauses at 27K tokens/min threshold   
        RPD = 1K  → chunk_size=2500 minimises daily calls 

    On 429 : reads Google's retry delay from error, waits + jitter
    On RPD exhausted : raises clear message immediately
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/gemini-embedding-001"
    )

    total        = len(text_chunks)
    est_min      = (total * EMBED_DELAY_SECONDS) / 60
    print(f"\nEmbedding {total} chunks — est. {est_min:.1f} min")
    print(f"RPD usage: {total}/1,000 daily quota\n")

    vector_store       = None
    tokens_this_minute = 0
    minute_start       = time.time()

    for idx, chunk in enumerate(text_chunks):
        chunk_tokens = max(1, len(chunk) // 4)   # 1 token ≈ 4 chars

        # --- TPM Guard: pause if approaching 30K tokens/min ---
        elapsed = time.time() - minute_start
        if elapsed < 60 and (tokens_this_minute + chunk_tokens) > TPM_SAFE_THRESHOLD:
            wait = 60 - elapsed + 2
            print(f"  TPM guard — {tokens_this_minute:,} tokens sent. Waiting {wait:.0f}s...")
            time.sleep(wait)
            tokens_this_minute = 0
            minute_start = time.time()

        if time.time() - minute_start >= 60:
            tokens_this_minute = 0
            minute_start = time.time()

        # --- Embed with retry ---
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(texts=[chunk], embedding=embeddings)
                else:
                    vector_store.add_texts(texts=[chunk])

                tokens_this_minute += chunk_tokens

                if (idx + 1) % 10 == 0 or (idx + 1) == total:
                    print(f"  ✓ {idx+1}/{total} chunks | ~{tokens_this_minute:,} tokens this min")
                break

            except Exception as e:
                err = str(e)

                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    match  = re.search(r"retry[^\d]*(\d+\.?\d*)\s*s", err, re.IGNORECASE)
                    g_wait = float(match.group(1)) if match else 30
                    wait   = g_wait + random.uniform(1, 3)
                    print(f"  429 on chunk {idx+1} (attempt {attempt}/{MAX_RETRIES}). Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    tokens_this_minute = 0
                    minute_start = time.time()

                    if attempt == MAX_RETRIES:
                        raise Exception(
                            f"Chunk {idx+1} failed after {MAX_RETRIES} retries. "
                            f"Daily quota may be exhausted — try again tomorrow."
                        ) from e

                elif "per day" in err.lower():
                    raise Exception(
                        "Daily RPD quota (1,000) exhausted. "
                        "Resets at midnight Pacific Time. Try again tomorrow."
                    ) from e

                else:
                    raise

        time.sleep(EMBED_DELAY_SECONDS)

    print(f"\n✅ Done — {total} chunks stored in FAISS.\n")
    return vector_store

# Build RAG agent

def create_rag_agent(vector_store):

    model = init_chat_model(
        "google_genai:gemini-2.5-flash",
        temperature = 0
    )

    @tool(response_format = "content_and_artifact")
    def retrieve_context(query:str):
        """
        searches the PDF for context relevant to query and return relevant chunks from the PDF
        """
        retrieved_docs = vector_store.similarity_search(query, k = 3)

        # Format docs as readable text for the LLM
        serialized = "\n\n".join(
            f"[Chunk {i+1}] :\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        )
        return serialized, retrieved_docs  # content for LLM, raw docs for UI
    
    tools  = [retrieve_context]

    system_prompt = (
        "You are a helpful assistant that answers questions about an uploaded PDF document "
        "You have access to a retrieval tool that searches the PDF content for context relevant to the question"
        "Always use the retrieval tool to find relevant information before answering the question"
        "If the document does not contain the answer, say so clearly"
        "Keep your answers concise, accurate and grounded in the document content"
    )

    agent = create_agent(model, tools, system_prompt = system_prompt)
    return agent

# Get the answer with conversation history
def get_answer(agent, user_question, chat_history):

    # convert history dicts to langchain message objects
    messages = []
    for msg in chat_history:
        if msg['role'] == "user":
            messages.append(HumanMessage(content = msg["content"]))
        elif msg['role'] =="assistant":
            messages.append(AIMessage(content = msg["content"]))

    # Append the current question
    messages.append(HumanMessage(content = user_question))

    source_docs = []
    final_answer = ""

    # Stream through all agent steps

    for step in agent.stream({"messages" : messages}, stream_mode = "values"):
        last_message = step["messages"][-1]

        #Collect source docs from tool message
        if isinstance(last_message, ToolMessage):
            if hasattr(last_message, "artifact") and isinstance(last_message.artifact, list):
                source_docs = last_message.artifact
        
        # Extract final answer from the AIMessage only
        if isinstance(last_message, AIMessage):
            content = last_message.content

            # Handle string content (common case)
            if isinstance(content, str) and content.strip():
                final_answer = content
            
            # Handle list of content blocks
            elif isinstance(content, list):
                text_parts = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ]
                assembled = " ".join(part for part in text_parts if part.strip())
                if assembled:
                    final_answer = assembled
    
    # Fallback if no answer was captured
    if not final_answer:
        final_answer = "I was unable to generate response. Please try rephrasing your question"
    
    return final_answer, source_docs