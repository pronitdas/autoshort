"""
Text-to-speech module using open source engines.
"""
import logging
from typing import Optional, Literal
import pyttsx3

from ..core.base import VoiceModule

logger = logging.getLogger(__name__)

class OpenSourceTTS(VoiceModule):
    """Text-to-speech module using open source engines.
    
    Currently supports pyttsx3 and Mozilla TTS (when available).
    Falls back to pyttsx3 if Mozilla TTS is not available.
    
    Args:
        engine_type: Type of TTS engine to use ('pyttsx3' or 'mozilla')
        
    Attributes:
        engine_type: The type of TTS engine being used
        engine: The TTS engine instance
    """
    def __init__(self, engine_type: Literal["pyttsx3", "mozilla"] = "pyttsx3"):
        super().__init__()
        self.engine_type = engine_type
        self.engine: Optional[pyttsx3.Engine] = None
        
        if engine_type == "pyttsx3":
            self._init_pyttsx3()
        elif engine_type == "mozilla":
            try:
                self._init_mozilla()
            except ImportError:
                logger.warning("Mozilla TTS not available, falling back to pyttsx3")
                self.engine_type = "pyttsx3"
                self._init_pyttsx3()
        else:
            raise ValueError(f"Unsupported TTS engine type: {engine_type}")

    def _init_pyttsx3(self) -> None:
        """Initialize the pyttsx3 engine with default settings."""
        try:
            self.engine = pyttsx3.init()
            # Set properties
            self.engine.setProperty('rate', 175)  # Speed of speech
            self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            
            # Try to find a good female voice
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if "female" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
                    
            logger.info("Successfully initialized pyttsx3 engine")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            raise

    def _init_mozilla(self) -> None:
        """Initialize the Mozilla TTS engine.
        
        Currently a placeholder for future implementation.
        
        Raises:
            ImportError: If Mozilla TTS is not available
        """
        try:
            # Mozilla TTS setup would go here
            # For now, we just raise ImportError to trigger fallback
            raise ImportError("Mozilla TTS not implemented yet")
        except ImportError:
            raise

    def generate_voice(self, text: str, output_file: str) -> str:
        """Generate speech from text and save to file.
        
        Args:
            text: The text to convert to speech
            output_file: Path where the audio file should be saved
            
        Returns:
            Path to the generated audio file
            
        Raises:
            Exception: If speech generation fails
        """
        try:
            if self.engine_type == "pyttsx3":
                if not self.engine:
                    self._init_pyttsx3()
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
            elif self.engine_type == "mozilla":
                # Mozilla TTS implementation would go here
                raise NotImplementedError("Mozilla TTS not implemented yet")
            
            logger.info(f"Generated speech saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise 