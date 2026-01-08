"""RAG Evaluation Metrics

Implements RAGAS-style metrics for evaluating RAG systems:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the query?
- Context Relevancy: Is the retrieved context relevant to the query?
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import re
import json

from src.logger import setup_logger

logger = setup_logger(__name__)


def _parse_score_response(response: str, metric_name: str = "metric") -> Dict[str, Any]:
    """
    Parse LLM response to extract a score (0-10 scale, normalized to 0-1).
    
    Handles multiple formats:
    - "8" or "8/10" 
    - "Score: 8"
    - "8. The answer is good..."
    - JSON {"score": 0.8}
    
    Args:
        response: Raw LLM response
        metric_name: Name of metric for logging
        
    Returns:
        Dict with 'score' (0.0-1.0) and 'reasoning'
    """
    response = response.strip()
    
    # Try JSON first
    try:
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get("score", 0.5))
            # Normalize if 0-10 scale
            if score > 1.0:
                score = score / 10.0
            return {
                "score": min(1.0, max(0.0, score)),
                "reasoning": result.get("reasoning", response),
            }
    except:
        pass
    
    # Try to extract number from start of response or after common patterns
    patterns = [
        r'^(\d+(?:\.\d+)?)',           # Number at start: "8" or "8.5"
        r'^(\d+)/10',                   # "8/10" format
        r'[Ss]core[:\s]+(\d+(?:\.\d+)?)',  # "Score: 8"
        r'^(\d+)\.',                    # "8. The answer..."
        r'(\d+(?:\.\d+)?)/10',          # "8/10" anywhere
        r'(\d+(?:\.\d+)?)\s*out of\s*10', # "8 out of 10"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                # Normalize 0-10 to 0-1
                if score > 1.0:
                    score = score / 10.0
                score = min(1.0, max(0.0, score))
                return {"score": score, "reasoning": response}
            except:
                continue
    
    # Try to find any decimal between 0 and 1
    decimal_match = re.search(r'0\.\d+', response)
    if decimal_match:
        try:
            score = float(decimal_match.group())
            return {"score": min(1.0, max(0.0, score)), "reasoning": response}
        except:
            pass
    
    # Look for word-based scores
    response_lower = response.lower()
    if any(word in response_lower for word in ['excellent', 'perfect', 'fully', 'completely']):
        return {"score": 0.9, "reasoning": response}
    elif any(word in response_lower for word in ['good', 'well', 'mostly', 'largely']):
        return {"score": 0.75, "reasoning": response}
    elif any(word in response_lower for word in ['moderate', 'partial', 'somewhat']):
        return {"score": 0.5, "reasoning": response}
    elif any(word in response_lower for word in ['poor', 'weak', 'little', 'barely']):
        return {"score": 0.25, "reasoning": response}
    elif any(word in response_lower for word in ['none', 'not', 'irrelevant', 'no']):
        return {"score": 0.1, "reasoning": response}
    
    logger.warning(f"Failed to parse {metric_name} response: {response[:100]}")
    return {"score": 0.5, "reasoning": f"Failed to parse: {response[:200]}"}


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
    
    PROMPT_TEMPLATE = """Rate if this answer is supported by the context.

Context: {context}

Answer: {answer}

Is the answer faithful to the context? Give a rating 0-10 where:
- 10 = every claim in the answer is supported by context
- 5 = about half the claims are supported  
- 0 = none of the claims are supported

Just write the number (0-10):"""

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
        # Truncate context to avoid token limits
        context_truncated = context[:1500] if len(context) > 1500 else context
        answer_truncated = answer[:500] if len(answer) > 500 else answer
        
        prompt = self.PROMPT_TEMPLATE.format(context=context_truncated, answer=answer_truncated)
        response = self.judge.generate(prompt, max_tokens=100)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract score"""
        result = _parse_score_response(response, "faithfulness")
        logger.debug(f"Faithfulness parse: {response[:80]} -> {result['score']}")
        return result


class AnswerRelevancyMetric:
    """
    Measures if the answer addresses the query.
    
    Score: 0.0 (irrelevant) to 1.0 (highly relevant)
    """
    
    PROMPT_TEMPLATE = """Does this answer address the question?

Question: {query}

Answer: {answer}

Rate the answer quality from 0 to 10:
- 10 = excellent, directly answers question
- 7 = good, mostly answers question
- 5 = okay, partially answers question
- 3 = poor, barely addresses question
- 0 = doesn't answer the question

Write just the number (0-10):"""

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
        answer_truncated = answer[:500] if len(answer) > 500 else answer
        prompt = self.PROMPT_TEMPLATE.format(query=query, answer=answer_truncated)
        response = self.judge.generate(prompt, max_tokens=100)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract score"""
        result = _parse_score_response(response, "answer_relevancy")
        logger.info(f"AnswerRel parse: '{response[:100]}' -> score={result['score']}")
        return result


class ContextRelevancyMetric:
    """
    Measures if the retrieved context is relevant to the query.
    
    Score: 0.0 (irrelevant) to 1.0 (highly relevant)
    """
    
    PROMPT_TEMPLATE = """Rate if this context is useful for answering the question.

Question: {query}

Context: {context}

Is the context relevant to the question? Rate from 0 to 10 where:
- 10 = context directly answers the question
- 5 = context is somewhat related
- 0 = context is completely irrelevant

Just write the number (0-10):"""

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
        context_truncated = context[:1500] if len(context) > 1500 else context
        prompt = self.PROMPT_TEMPLATE.format(query=query, context=context_truncated)
        response = self.judge.generate(prompt, max_tokens=100)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract score"""
        result = _parse_score_response(response, "context_relevancy")
        logger.debug(f"ContextRel parse: {response[:80]} -> {result['score']}")
        return result
