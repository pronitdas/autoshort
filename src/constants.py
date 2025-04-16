"""
Constants and configuration values for the AutoShort pipeline.
"""
from enum import Enum
from typing import Final

# Video Configuration
YOUTUBE_SHORT_RESOLUTION: Final = (1080, 1920)
MAX_SCENE_DURATION: Final = 5
DEFAULT_SCENE_DURATION: Final = 1

# Subtitle Configuration
SUBTITLE_FONT_SIZE: Final = 13
SUBTITLE_FONT_COLOR: Final = "yellow@0.5"
SUBTITLE_ALIGNMENT: Final = 2  # Centered horizontally and vertically
SUBTITLE_BOLD: Final = True
SUBTITLE_OUTLINE_COLOR: Final = "&H40000000"  # Black with 50% transparency
SUBTITLE_BORDER_STYLE: Final = 3

# Fallback Scene Configuration
FALLBACK_SCENE_COLOR: Final = "red"
FALLBACK_SCENE_TEXT_COLOR: Final = "yellow@0.5"
FALLBACK_SCENE_BOX_COLOR: Final = "black@0.5"
FALLBACK_SCENE_BOX_BORDER_WIDTH: Final = 5
FALLBACK_SCENE_FONT_SIZE: Final = 30
FALLBACK_SCENE_FONT_FILE: Final = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Model Configuration
DEFAULT_TEXT_GENERATION_MODEL: Final = "meta-llama/Llama-3-8b-chat-hf"
DEFAULT_LONG_TEXT_GENERATION_MODEL: Final = "mistralai/Mistral-7B-Instruct-v0.2"
STABLE_DIFFUSION_MODEL: Final = "runwayml/stable-diffusion-v1-5"

class PixelFormat(Enum):
    """Supported pixel formats for video processing."""
    YUVJ420P = 'yuvj420p'
    YUVJ422P = 'yuvj422p'
    YUVJ444P = 'yuvj444p'
    YUVJ440P = 'yuvj440p'
    YUV420P = 'yuv420p'
    YUV422P = 'yuv422p'
    YUV444P = 'yuv444p'
    YUV440P = 'yuv440p' 