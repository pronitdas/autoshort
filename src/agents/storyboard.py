"""
Storyboard generation agent for creating video scene breakdowns.
"""
import logging
import spacy
from typing import Dict, Any, List

from ..constants import DEFAULT_LONG_TEXT_GENERATION_MODEL
from .base import ProcessingAgent

logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    logger.warning("Downloading spaCy model 'en_core_web_md'...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

class StoryboardGenerationAgent(ProcessingAgent):
    """Agent for generating video storyboards.
    
    This agent breaks down scripts into scenes with visual descriptions
    and keywords for video segment selection.
    """
    def __init__(self):
        super().__init__(
            "Storyboard Generation Agent",
            DEFAULT_LONG_TEXT_GENERATION_MODEL
        )
        self.nlp = nlp

    async def execute(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate storyboard from script.
        
        Args:
            input_data: Dictionary containing:
                - script: Video script
                
        Returns:
            List of scene dictionaries
            
        Raises:
            Exception: If storyboard generation fails
        """
        try:
            script = input_data.get('script', '')
            
            if not script:
                logger.error("No script provided for storyboard generation")
                raise ValueError("No script provided for storyboard generation")

            system_prompt = (
                "You are an AI assistant specializing in creating engaging "
                "and viral storyboards for faceless YouTube Shorts videos "
                "using the provided script."
            )
            
            prompt = f"""Create a storyboard for a three minute faceless YouTube Shorts video based on the following script:

{script}

For each major scene (aim for 15-20 scenes), provide:

1. Visual: A brief description of the visual elements (1 sentence). Ensure each scene has unique and engaging visuals suitable for a faceless video.

2. Text: The exact text/dialogue for voiceover and subtitles, written in lowercase with minimal punctuation, only when absolutely necessary.

3. Video Keyword: A specific keyword or phrase for searching stock video footage. Be precise and avoid repeating keywords across scenes.

4. Image Keyword: A backup keyword for searching stock images. Be specific and avoid repeating keywords.

Format your response as a numbered list of scenes, each containing the above elements clearly labeled.

Example:

1. Visual: A time-lapse of clouds moving rapidly over a city skyline

   Text: time flies when we're lost in the hustle

   Video Keyword: time-lapse city skyline

   Image Keyword: fast-moving clouds over city

2. Visual: ...

Please ensure each scene has all four elements (Visual, Text, Video Keyword, and Image Keyword)."""

            response = await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.7
            )
            
            logger.info(f"Raw storyboard response: {response}")
            scenes = self._parse_scenes(response)
            
            if not scenes:
                logger.error("Failed to generate valid storyboard scenes")
                raise ValueError("Failed to generate valid storyboard scenes")
            
            return scenes
            
        except Exception as e:
            logger.error(f"Storyboard generation failed: {e}")
            raise
    
    def _parse_scenes(self, response: str) -> List[Dict[str, Any]]:
        """Parse raw response into scene dictionaries.
        
        Args:
            response: Raw storyboard text
            
        Returns:
            List of scene dictionaries
        """
        scenes = []
        current_scene = {}
        current_scene_number = None

        for line in response.split('\n'):
            line = line.strip()
            logger.debug(f"Processing line: {line}")

            if line.startswith(tuple(f"{i}." for i in range(1, 51))):  # Up to 50 scenes
                if current_scene:
                    # Append the completed current_scene
                    current_scene['number'] = current_scene_number
                    current_scene = self._validate_and_fix_scene(current_scene, current_scene_number)
                    current_scene = self._enhance_scene_keywords(current_scene)
                    scenes.append(current_scene)
                    logger.debug(f"Scene {current_scene_number} appended to scenes list")
                    current_scene = {}

                try:
                    current_scene_number = int(line.split('.', 1)[0])
                    logger.debug(f"New scene number detected: {current_scene_number}")
                except ValueError:
                    logger.warning(f"Invalid scene number format: {line}")
                    continue
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                current_scene[key] = value
                logger.debug(f"Key-value pair added to current scene: {key}:{value}")
            else:
                logger.warning(f"Line format not recognized: {line}")

        # Process the last scene if exists
        if current_scene:
            current_scene['number'] = current_scene_number
            current_scene = self._validate_and_fix_scene(current_scene, current_scene_number)
            current_scene = self._enhance_scene_keywords(current_scene)
            scenes.append(current_scene)
            logger.debug(f"Final scene {current_scene_number} appended to scenes list")

        logger.info(f"Parsed and enhanced scenes: {scenes}")
        return scenes
    
    def _enhance_scene_keywords(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance scene keywords using NLP.
        
        Args:
            scene: Scene dictionary
            
        Returns:
            Enhanced scene dictionary
        """
        # Extract keywords from narration_text and visual descriptions
        narration_doc = self.nlp(scene.get('narration_text', ''))
        visual_doc = self.nlp(scene.get('visual', ''))

        # Extract nouns and named entities
        def extract_keywords(doc):
            return [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN') or token.ent_type_]

        narration_keywords = extract_keywords(narration_doc)
        visual_keywords = extract_keywords(visual_doc)

        # Combine and deduplicate keywords
        combined_keywords = list(set(narration_keywords + visual_keywords))

        # Generate enhanced video and image keywords
        scene['video_keyword'] = ' '.join(combined_keywords[:5])  # Use top 5 keywords
        scene['image_keyword'] = scene['video_keyword']

        return scene

    def _validate_and_fix_scene(self, scene: Dict[str, Any], scene_number: int) -> Dict[str, Any]:
        """Validate and fix scene data.
        
        Args:
            scene: Scene dictionary
            scene_number: Scene number
            
        Returns:
            Validated and fixed scene dictionary
        """
        # Ensure 'number' key is present
        scene['number'] = scene_number

        # Required keys with default values
        required_keys = {
            'visual': f"Visual representation of scene {scene_number}",
            'text': "",
            'video_keyword': f"video scene {scene_number}",
            'image_keyword': f"image scene {scene_number}"
        }

        # Add missing keys with default values
        for key, default_value in required_keys.items():
            if key not in scene:
                scene[key] = default_value
                logger.warning(f"Added missing {key} for scene {scene_number}")

        # Clean the 'text' field
        text = scene.get('text', '')
        text = text.strip('"').strip("'")
        scene['text'] = text

        # Copy the cleaned text into 'narration_text'
        scene['narration_text'] = text

        return scene 