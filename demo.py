"""Streamlit Demo Application for PulmoRAG"""

import os
import time
import threading
import streamlit as st
from pathlib import Path
from typing import Generator
from src.logger import logger
from src.config import get_config
from src.generation.llm import LocalLLM, LLMFactory
from src.rag import RAGPipeline

# Global lock to prevent multiple LLM loads
_llm_lock = threading.Lock()
_llm_instance = None

# Enable streaming responses by default
ENABLE_STREAMING = True

st.set_page_config(
    page_title="PulmoRAG - COPD Assistant",
    page_icon="🫁",
    layout="centered",
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        padding: 0.5rem;
        border-left: 3px solid #ff9800;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🫁 Pulmonary Disease Assistant")
st.markdown("Get evidence-based information about pulmonary diseases including COPD, asthma, pneumonia, tuberculosis, and more.")

# Disclaimer
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Medical Disclaimer:</strong> This assistant provides general information only and should not replace professional medical advice. Always consult a healthcare provider for medical decisions.
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


@st.cache_resource(show_spinner="Loading AI model...")
def load_local_llm():
    """Load the local LLM model (cached across all sessions) with optimizations."""
    global _llm_instance
    
    # Use lock to prevent race conditions during concurrent loads
    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance
            
        config = get_config()
        model_path = Path(config.LOCAL_MODEL_PATH)
        if model_path.exists():
            try:
                start_time = time.time()
                logger.info(f"Loading LLM from {model_path}...")
                
                _llm_instance = LocalLLM(
                    model_path=str(model_path),
                    n_ctx=config.LOCAL_MODEL_CTX,
                    n_gpu_layers=config.LOCAL_MODEL_GPU_LAYERS,
                    n_batch=config.LOCAL_MODEL_BATCH_SIZE,
                    n_threads=config.LOCAL_MODEL_THREADS if config.LOCAL_MODEL_THREADS > 0 else None,
                )
                
                # Warm-up inference (reduces first-query latency)
                logger.info("Warming up model...")
                _llm_instance.generate("Hello", max_tokens=5, temperature=0.1)
                
                load_time = time.time() - start_time
                logger.info(f"LLM loaded and warmed up in {load_time:.2f}s")
                
                return _llm_instance
            except Exception as e:
                logger.error(f"Failed to load local LLM: {e}")
                st.error(f"Failed to load local model: {e}")
                return None
    return None


@st.cache_data
def get_available_backends():
    """Get available LLM backends (cached to avoid repeated checks)."""
    return LLMFactory.get_available_backends()


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_rag_pipeline():
    """Load the RAG pipeline (cached across sessions)."""
    try:
        pipeline = RAGPipeline(collection_name="copd_documents")
        logger.info("RAG pipeline loaded")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load RAG pipeline: {e}")
        return None


def get_system_prompt() -> str:
    """Get the system prompt for COPD assistant."""
    return """You are PulmoRAG, a helpful and empathetic medical information assistant specializing in COPD (Chronic Obstructive Pulmonary Disease).

Your role is to:
- Provide accurate, evidence-based information about COPD
- Explain medical concepts in clear, understandable language
- Be empathetic and supportive in your responses
- Always recommend consulting healthcare professionals for medical decisions

Important guidelines:
- Be concise but thorough
- Cite sources when possible
- Acknowledge uncertainty when appropriate
- Never provide specific medical advice or diagnoses"""


def get_response_stream(user_prompt: str) -> Generator[str, None, None]:
    """Generate streaming response using RAG pipeline."""
    # Check available backends (cached)
    backends = get_available_backends()
    
    if not backends:
        yield """⚠️ **No LLM backend available.**

To enable AI responses, you have two options:

**Option 1: Use Local Model (No API key needed)**
```bash
# Download the MistralLite model
python scripts/download_model.py

# Restart the app
docker compose restart rag-app
```

**Option 2: Use API**
1. Copy `.env.example` to `.env`
2. Add your `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
3. Restart: `docker compose restart rag-app`"""
        return
    
    # Load RAG pipeline
    rag_pipeline = load_rag_pipeline()
    if not rag_pipeline:
        yield "⚠️ Failed to load RAG pipeline. Check logs for details."
        return
    
    # Try local LLM first if available
    if "local" in backends:
        try:
            llm = load_local_llm()
            if llm:
                # Set LLM for RAG pipeline
                rag_pipeline.set_llm(llm)
                
                # Retrieve relevant context
                results = rag_pipeline.retrieve(user_prompt)
                context, sources = rag_pipeline.build_context(results)
                
                # Store sources for display
                st.session_state.last_sources = sources
                
                # Assess relevance level
                relevance_level = rag_pipeline._assess_relevance(sources)
                st.session_state.last_relevance = relevance_level
                
                # Build RAG prompt with relevance-aware instructions
                prompt = rag_pipeline.build_prompt(user_prompt, context, relevance_level)
                
                logger.info(f"RAG: {len(sources)} sources, context ~{len(context)//4} tokens, relevance={relevance_level}")
                
                # Show relevance warning before streaming if low relevance
                if relevance_level == "low":
                    yield "⚠️ **Note:** This query may be outside my knowledge base. The sources below have low relevance to your question.\n\n"
                elif relevance_level == "medium":
                    yield "ℹ️ **Note:** Found partially relevant sources. Some information may be tangentially related.\n\n"
                
                # Stream the response
                for chunk in llm.generate(
                    prompt,
                    max_tokens=512,
                    temperature=0.7,
                    stream=True,
                ):
                    yield chunk
                
                # Append source citations if sources found
                if sources:
                    yield "\n\n---\n**Sources:**\n"
                    for src in sources:
                        relevance_indicator = "🟢" if src['score'] >= 0 else ("🟡" if src['score'] >= -5 else "🔴")
                        yield f"- [{src['index']}] {src['source']} {relevance_indicator} (score: {src['score']:.2f})\n"
                return
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            yield f"⚠️ Error with RAG pipeline: {str(e)}"
            return
    
    # API-based response (placeholder for now)
    if "openai" in backends or "anthropic" in backends:
        yield """I'm connected to an API backend but the full RAG pipeline isn't implemented yet.

Once fully configured, I'll be able to:
- Provide evidence-based treatment information
- Cite medical sources for all answers
- Answer questions about symptoms, diagnosis, and management

Please check back soon!"""
        return
    
    yield "No LLM backend available."


def get_response(user_prompt: str) -> str:
    """Generate non-streaming response (fallback)."""
    return "".join(get_response_stream(user_prompt))


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask about pulmonary diseases (COPD, asthma, pneumonia, TB, etc.)...")

# Handle pending prompt from sidebar buttons
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    logger.info(f"Query submitted: {prompt}")
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        if ENABLE_STREAMING:
            # Streaming response for better UX
            response_placeholder = st.empty()
            full_response = ""
            start_time = time.time()
            
            for chunk in get_response_stream(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")  # Show cursor
            
            # Final render without cursor
            response_placeholder.markdown(full_response)
            
            # Log performance metrics
            elapsed = time.time() - start_time
            tokens_estimate = len(full_response.split())
            logger.info(f"Response generated in {elapsed:.2f}s (~{tokens_estimate} tokens, {tokens_estimate/elapsed:.1f} tok/s)")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Non-streaming fallback
            with st.spinner("Thinking..."):
                response = get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with example questions
with st.sidebar:
    st.subheader("💡 Example Questions")
    
    examples = [
        "What are the main symptoms of COPD?",
        "What treatment options are available for COPD?",
        "What inhalers are used for COPD?",
        "How can I improve my breathing with COPD?",
        "What lifestyle changes help manage COPD?",
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.pending_prompt = example
            st.rerun()
    
    st.divider()
    
    # Show LLM backend status
    st.subheader("🤖 LLM Status")
    backends = get_available_backends()
    
    if "local" in backends:
        st.success("✅ Local Model (MistralLite)")
    elif "openai" in backends:
        st.success("✅ OpenAI API")
    elif "anthropic" in backends:
        st.success("✅ Anthropic API")
    else:
        st.warning("⚠️ No LLM configured")
        st.caption("Run `python scripts/download_model.py` for local model")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
