"""LLM Interface with support for both API and local models"""

import os
from pathlib import Path
from typing import Optional, Generator
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)


class LocalLLM:
    """Local LLM using llama-cpp-python with GGUF models"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        n_batch: int = 512,
    ):
        """
        Initialize the local LLM with performance optimizations.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads (None = auto-detect)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            n_batch: Batch size for prompt processing (higher = faster but more memory)
        """
        from llama_cpp import Llama
        
        config = get_config()
        
        if model_path is None:
            model_path = config.LOCAL_MODEL_PATH
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'python scripts/download_model.py' to download the model."
            )
        
        # Auto-detect optimal thread count (leave some for system)
        if n_threads is None:
            cpu_count = os.cpu_count() or 4
            n_threads = max(1, cpu_count - 1)
        
        logger.info(f"Loading local LLM from {model_path}...")
        logger.info(f"Config: ctx={n_ctx}, threads={n_threads}, gpu_layers={n_gpu_layers}, batch={n_batch}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=False,
            use_mlock=True,  # Lock model in RAM to prevent swapping (better performance)
            use_mmap=True,   # Use memory-mapped file for faster loading
        )
        
        logger.info("Local LLM loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[list] = None,
        stream: bool = False,
    ) -> str | Generator:
        """
        Generate a response from the local LLM with optimized parameters.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused, higher = more creative)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter (limits vocab considered)
            repeat_penalty: Penalty for repeating tokens (>1 discourages repetition)
            stop: List of stop sequences
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        if stream:
            return self._stream_generate(
                prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, stop
            )
        
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
        )
        
        return response["choices"][0]["text"].strip()
    
    def _stream_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[list] = None,
    ) -> Generator:
        """Stream response tokens with optimized parameters."""
        for chunk in self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text:  # Only yield non-empty chunks
                yield text
    
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Generator:
        """
        Chat-style interface for the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Generated response
        """
        # Format messages into a prompt
        prompt = self._format_chat_prompt(messages)
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
    
    def _format_chat_prompt(self, messages: list[dict]) -> str:
        """
        Format chat messages into a prompt for MistralLite.
        
        MistralLite uses the following format:
        <|prompter|>User message</s><|assistant|>
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"<|prompter|>{content}</s>")
            elif role == "user":
                prompt_parts.append(f"<|prompter|>{content}</s>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>{content}</s>")
        
        # Add the assistant prefix for generation
        prompt_parts.append("<|assistant|>")
        
        return "".join(prompt_parts)


class LLMFactory:
    """Factory for creating LLM instances based on configuration"""
    
    _local_llm_instance: Optional[LocalLLM] = None
    
    @classmethod
    def get_llm(cls, prefer_local: bool = False) -> LocalLLM | None:
        """
        Get an LLM instance based on available configuration.
        
        Args:
            prefer_local: If True, prefer local model over API
            
        Returns:
            LLM instance or None if no LLM is available
        """
        config = get_config()
        
        # Check for local model first if preferred
        if prefer_local or (not config.OPENAI_API_KEY and not config.ANTHROPIC_API_KEY):
            local_model_path = config.LOCAL_MODEL_PATH
            if Path(local_model_path).exists():
                if cls._local_llm_instance is None:
                    logger.info("Using local LLM (MistralLite)")
                    cls._local_llm_instance = LocalLLM(model_path=local_model_path)
                return cls._local_llm_instance
            else:
                logger.warning(
                    f"Local model not found at {local_model_path}. "
                    "Run 'python scripts/download_model.py' to download."
                )
        
        # Fall back to API-based LLM info
        if config.OPENAI_API_KEY:
            logger.info("OpenAI API key found - API mode available")
            return None  # Will use API directly in demo
        
        if config.ANTHROPIC_API_KEY:
            logger.info("Anthropic API key found - API mode available")
            return None  # Will use API directly in demo
        
        return None
    
    @classmethod
    def is_local_model_available(cls) -> bool:
        """Check if local model is available."""
        config = get_config()
        return Path(config.LOCAL_MODEL_PATH).exists()
    
    @classmethod
    def get_available_backends(cls) -> list[str]:
        """Get list of available LLM backends."""
        backends = []
        config = get_config()
        
        if Path(config.LOCAL_MODEL_PATH).exists():
            backends.append("local")
        if config.OPENAI_API_KEY:
            backends.append("openai")
        if config.ANTHROPIC_API_KEY:
            backends.append("anthropic")
        
        return backends
