"""Generation module"""

from src.logger import setup_logger
from src.generation.llm import LocalLLM, LLMFactory

logger = setup_logger(__name__)

__all__ = ["LLMGenerator", "LocalLLM", "LLMFactory"]


class LLMGenerator:
    """Handles LLM-based generation"""

    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        raise NotImplementedError
