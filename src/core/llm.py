"""
LLM generator singleton for text generation tasks.
"""
import logging
import torch
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..constants import (
    DEFAULT_TEXT_GENERATION_MODEL,
    DEFAULT_LONG_TEXT_GENERATION_MODEL
)

logger = logging.getLogger(__name__)

class LLMGenerator:
    """Singleton class for managing LLM text generation.
    
    This class ensures we only load each model once and reuse it
    across different agents.
    """
    _instance: Optional['LLMGenerator'] = None
    _models: Dict[str, AutoModelForCausalLM] = {}
    _tokenizers: Dict[str, AutoTokenizer] = {}

    def __new__(cls) -> 'LLMGenerator':
        if cls._instance is None:
            cls._instance = super(LLMGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize if this is the first instantiation
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

    def _load_model(self, model_name: str) -> None:
        """Load a model and its tokenizer if not already loaded.
        
        Args:
            model_name: The name/path of the model to load
            
        Raises:
            Exception: If model loading fails
        """
        if model_name not in self._models:
            try:
                logger.info(f"Loading model {model_name}")
                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                self._models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info(f"Successfully loaded model {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                raise

    async def generate(
        self,
        prompt: str,
        model_name: str = DEFAULT_TEXT_GENERATION_MODEL,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate text using the specified model.
        
        Args:
            prompt: The input prompt for text generation
            model_name: Name/path of the model to use
            system_prompt: Optional system prompt for instruction tuning
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
            
        Raises:
            Exception: If text generation fails
        """
        try:
            # Load model if not already loaded
            if model_name not in self._models:
                self._load_model(model_name)

            # Format prompt based on model type
            if "llama" in model_name.lower() or "mistral" in model_name.lower():
                full_prompt = f"<s>[INST] {system_prompt or ''}\n\n{prompt} [/INST]"
            else:
                full_prompt = f"System: {system_prompt or 'You are a helpful assistant.'}\n\nUser: {prompt}\n\nAssistant:"

            # Tokenize input
            tokens = self._tokenizers[model_name](
                full_prompt, 
                return_tensors="pt"
            ).to(self.device)

            # Generate text
            with torch.no_grad():
                generation_output = self._models[model_name].generate(
                    **tokens,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )

            # Decode output
            output = self._tokenizers[model_name].decode(
                generation_output[0], 
                skip_special_tokens=True
            )

            # Extract response
            if "Assistant:" in output:
                response = output.split("Assistant:")[-1].strip()
            else:
                response = output.split("[/INST]")[-1].strip()

            logger.info(f"Generated text with {len(response)} characters")
            return response

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

# Global instance
llm_generator = LLMGenerator() 