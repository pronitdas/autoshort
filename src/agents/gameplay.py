"""
Gameplay video agent for processing and selecting video segments.
"""
import logging
from typing import Dict, Any, List, Optional

from ..core.base import Agent
from ..video.gameplay import GameplayVideoProcessor

logger = logging.getLogger(__name__)

class GameplayVideoAgent(Agent[Dict[str, Any], List[Optional[Dict[str, Any]]]]):
    """Agent for processing gameplay videos for each scene.
    
    This agent manages the selection and extraction of gameplay video
    segments based on scene requirements.
    """
    def __init__(self):
        super().__init__("Gameplay Video Agent", "local")
        self.processor: Optional[GameplayVideoProcessor] = None
        
    def set_gameplay_folder(self, gameplay_folder: str) -> None:
        """Set the folder containing gameplay videos.
        
        Args:
            gameplay_folder: Path to the gameplay videos folder
            
        Raises:
            FileNotFoundError: If gameplay folder doesn't exist
        """
        self.processor = GameplayVideoProcessor(gameplay_folder)
        
    async def execute(self, input_data: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
        """Process gameplay videos for scenes.
        
        Args:
            input_data: Dictionary containing:
                - scenes: List of scene dictionaries
                
        Returns:
            List of processed scene results
            
        Raises:
            ValueError: If gameplay folder not set or no videos found
            Exception: If video processing fails
        """
        try:
            scenes = input_data.get('scenes', [])
            
            if not self.processor:
                logger.error("Gameplay folder not set. Call set_gameplay_folder() first.")
                raise ValueError("Gameplay folder not set")
                
            if not self.processor.video_files:
                logger.error("No gameplay videos found. Please check the gameplay folder.")
                raise ValueError("No gameplay videos found")
            
            # Process all scenes
            results = await self.processor.process_scenes(scenes)
            
            # Log processing summary
            successful = len([r for r in results if r is not None])
            logger.info(f"Processed {successful}/{len(scenes)} scenes successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in GameplayVideoAgent: {e}")
            raise 