"""
Base classes for content generation agents.
"""
from typing import Dict, Any, TypeVar, Generic
from ..core.base import Agent
from ..core.llm import llm_generator

T = TypeVar('T')
U = TypeVar('U')

class ContentGenerationAgent(Agent[T, U], Generic[T, U]):
    """Base class for content generation agents.
    
    This class provides common functionality for agents that generate
    content using LLMs.
    """
    def __init__(self, name: str, model: str):
        super().__init__(name, model)
        self.llm = llm_generator

    async def generate_content(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate content using the LLM.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
            
        Raises:
            Exception: If text generation fails
        """
        return await self.llm.generate(
            prompt=prompt,
            model_name=self.model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

class ResearchAgent(ContentGenerationAgent[Dict[str, Any], str]):
    """Base class for research-focused agents.
    
    These agents gather and analyze information from various sources.
    """
    pass

class GenerationAgent(ContentGenerationAgent[str, str]):
    """Base class for pure generation agents.
    
    These agents focus on generating new content from input text.
    """
    pass

class ProcessingAgent(ContentGenerationAgent[Dict[str, Any], Dict[str, Any]]):
    """Base class for processing agents.
    
    These agents transform input data into new formats or structures.
    """
    pass 