"""
Video processor for gameplay footage.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)

class GameplayVideoProcessor:
    """Processes gameplay videos to extract relevant segments."""
    
    def __init__(self, gameplay_folder: str):
        """Initialize the processor.
        
        Args:
            gameplay_folder: Path to folder containing gameplay videos
            
        Raises:
            FileNotFoundError: If gameplay folder doesn't exist
        """
        self.gameplay_folder = Path(gameplay_folder)
        if not self.gameplay_folder.exists():
            raise FileNotFoundError(f"Gameplay folder not found: {gameplay_folder}")
            
        self.video_files = self._get_video_files()
        logger.info(f"Found {len(self.video_files)} gameplay videos")
        
    def _get_video_files(self) -> List[Path]:
        """Get list of video files in gameplay folder."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        return [
            f for f in self.gameplay_folder.glob('**/*') 
            if f.suffix.lower() in video_extensions
        ]
        
    async def process_scenes(self, scenes: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Process gameplay videos for each scene.
        
        Args:
            scenes: List of scene dictionaries containing:
                - visual_description: Description of required visuals
                - keywords: List of keywords to match
                - duration: Required duration in seconds
                
        Returns:
            List of processed scene results, with None for failed scenes
        """
        results = []
        
        for scene in scenes:
            try:
                # Find best matching video segment
                result = await self._process_scene(scene)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process scene: {e}")
                results.append(None)
                
        return results
        
    async def _process_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single scene.
        
        Args:
            scene: Scene dictionary
            
        Returns:
            Dictionary containing:
                - video_path: Path to source video
                - start_time: Start time in seconds
                - end_time: End time in seconds
                - score: Match score
                
        Raises:
            ValueError: If no suitable segment found
        """
        best_match = None
        best_score = -1
        
        for video_file in self.video_files:
            try:
                # Score video segments based on visual features and keywords
                segments = await self._analyze_video(video_file, scene)
                
                for segment in segments:
                    score = self._score_segment(segment, scene)
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'video_path': str(video_file),
                            'start_time': segment['start_time'],
                            'end_time': segment['end_time'],
                            'score': score
                        }
                        
            except Exception as e:
                logger.warning(f"Error analyzing video {video_file}: {e}")
                continue
                
        if not best_match:
            raise ValueError("No suitable video segment found")
            
        return best_match
        
    async def _analyze_video(self, video_path: Path, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze video to find potential segments.
        
        Args:
            video_path: Path to video file
            scene: Scene requirements
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        duration = scene.get('duration', 10)  # Default 10s
        
        try:
            video = VideoFileClip(str(video_path))
            
            # Sample frames at 1fps for analysis
            for t in range(0, int(video.duration - duration), 1):
                segment = {
                    'start_time': t,
                    'end_time': t + duration,
                    'features': await self._extract_features(video, t)
                }
                segments.append(segment)
                
            video.close()
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            raise
            
        return segments
        
    async def _extract_features(self, video: VideoFileClip, time: float) -> Dict[str, Any]:
        """Extract visual features from video frame.
        
        Args:
            video: VideoFileClip object
            time: Time in seconds
            
        Returns:
            Dictionary of visual features
        """
        frame = video.get_frame(time)
        
        # Convert to grayscale for feature extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Extract basic features
        features = {
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'edges': cv2.Canny(gray, 100, 200).sum() / (gray.shape[0] * gray.shape[1])
        }
        
        return features
        
    def _score_segment(self, segment: Dict[str, Any], scene: Dict[str, Any]) -> float:
        """Score how well a segment matches scene requirements.
        
        Args:
            segment: Segment dictionary with features
            scene: Scene requirements
            
        Returns:
            Score between 0 and 1
        """
        # Implement scoring based on:
        # - Visual feature matching
        # - Keyword relevance
        # - Scene requirements
        
        # Simple scoring example
        features = segment['features']
        score = 0.0
        
        # Prefer segments with good brightness and contrast
        score += 0.3 * (0.5 - abs(features['brightness'] - 128) / 256)
        score += 0.3 * (features['contrast'] / 128)
        
        # Prefer segments with moderate edge content
        score += 0.4 * (1.0 - abs(features['edges'] - 0.1) / 0.2)
        
        return max(0.0, min(1.0, score)) 