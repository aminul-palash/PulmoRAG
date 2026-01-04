"""RAG Evaluation Metrics

Implements RAGAS-style metrics for evaluating RAG systems:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the query?
- Context Relevancy: Is the retrieved context relevant to the query?
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import re

from src.logger import setup_logger

logger = setup_logger(__name__)


class BaseLLMJudge(ABC):
    """Abstract base class for LLM-based evaluation"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response from the LLM"""
        pass


class LocalLLMJudge(BaseLLMJudge):
    """Use local MistralLite as evaluation judge"""
    
    def __init__(self, llm=None):
        """
        Initialize with optional pre-loaded LLM.
        
        Args:
            llm: Pre-loaded LocalLLM instance (loads new if None)
        """
        if llm is None:
            from src.generation.llm import LLMFactory
            llm = LLMFactory.get_llm(prefer_local=True)
            if llm is None:
                raise RuntimeError("No local LLM available. Run scripts/download_model.py first.")
        self.llm = llm
        logger.info("LocalLLMJudge initialized")
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return self.llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,  # Low temperature for consistent evaluation
            stream=False,
        )


class OpenAIJudge(BaseLLMJudge):
    """Use OpenAI API as evaluation judge"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI judge.
        
        Args:
            model: OpenAI model to use
        """
        import openai
        from src.config import get_config
        
        config = get_config()
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model
        logger.info(f"OpenAIJudge initialized with {model}")
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content


def get_judge(use_local: bool = True, llm=None) -> BaseLLMJudge:
    """
    Get an LLM judge for evaluation.
    
    Args:
        use_local: Use local LLM if True, OpenAI if False
        llm: Pre-loaded LLM instance (for local)
        
    Returns:
        LLM judge instance
    """
    if use_local:
        return LocalLLMJudge(llm=llm)
    else:
        return OpenAIJudge()


class FaithfulnessMetric:
    """
    Measures if the answer is grounded in the provided context.
    
    Score: 0.0 (hallucinated) to 1.0 (fully grounded)
    """
    
    PROMPT_TEMPLATE = """You are evaluating whether an answer is faithful to the given context.

Context:
{context}

Answer:
{answer}

Task: Determine what fraction of the claims in the answer can be verified from the context.

Instructions:
1. List the key claims/statements in the answer
2. For each claim, check if it's supported by the context
3. Calculate: (supported claims) / (total claims)

Respond with ONLY a JSON object:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}"""

    def __init__(self, judge: BaseLLMJudge):
        self.judge = judge
    
    def evaluate(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Evaluate faithfulness of answer to context.
        
        Args:
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Dict with 'score' and 'reasoning'
        """
        prompt = self.PROMPT_TEMPLATE.format(context=context, answer=answer)
        response = self.judge.generate(prompt, max_tokens=256)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract score"""
        try:
            # Try to extract JSON from response
            import json
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "score": float(result.get("score", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                }
        except Exception as e:
            logger.warning(f"Failed to parse faithfulness response: {e}")
        
        # Fallback: try to extract a number
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
        if numbers:
            return {"score": float(numbers[0]), "reasoning": response}
        
        return {"score": 0.5, "reasoning": "Failed to parse response"}


class AnswerRelevancyMetric:
    """
    Measures if the answer addresses the query.
    
    Score: 0.0 (irrelevant) to 1.0 (highly relevant)
    """
    
    PROMPT_TEMPLATE = """You are evaluating whether an answer is relevant to a question.

Question:
{query}

Answer:
{answer}

Task: Rate how well the answer addresses the question.

Consider:
- Does it directly answer what was asked?
- Is it complete and informative?
- Does it stay on topic?

Respond with ONLY a JSON object:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}"""

    def __init__(self, judge: BaseLLMJudge):
        self.judge = judge
    
    def evaluate(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Evaluate relevancy of answer to query.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            Dict with 'score' and 'reasoning'
        """
        prompt = self.PROMPT_TEMPLATE.format(query=query, answer=answer)
        response = self.judge.generate(prompt, max_tokens=256)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract score"""
        try:
            import json
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "score": float(result.get("score", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                }
        except Exception as e:
            logger.warning(f"Failed to parse relevancy response: {e}")
        
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
        if numbers:
            return {"score": float(numbers[0]), "reasoning": response}
        
        return {"score": 0.5, "reasoning": "Failed to parse response"}


class ContextRelevancyMetric:
    """
    Measures if the retrieved context is relevant to the query.
    
    Score: 0.0 (irrelevant) to 1.0 (highly relevant)
    """
    
    PROMPT_TEMPLATE = """You are evaluating whether retrieved context is relevant to a question.

Question:
{query}

Retrieved Context:
{context}

Task: Rate how relevant the context is for answering the question.

Consider:
- Does the context contain information to answer the question?
- Is the context focused on the topic?
- Would this context help generate a good answer?

Respond with ONLY a JSON object:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}"""

    def __init__(self, judge: BaseLLMJudge):
        self.judge = judge
    
    def evaluate(self, query: str, context: str) -> Dict[str, Any]:
        """
        Evaluate relevancy of context to query.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Dict with 'score' and 'reasoning'
        """
        prompt = self.PROMPT_TEMPLATE.format(query=query, context=context)
        response = self.judge.generate(prompt, max_tokens=256)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract score"""
        try:
            import json
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "score": float(result.get("score", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                }
        except Exception as e:
            logger.warning(f"Failed to parse context relevancy response: {e}")
        
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
        if numbers:
            return {"score": float(numbers[0]), "reasoning": response}
        
        return {"score": 0.5, "reasoning": "Failed to parse response"}
