"""
Helper functions for the AutoShort pipeline.
"""
import logging
import requests
from typing import Dict, Optional, Any

from ..constants import PixelFormat

logger = logging.getLogger(__name__)

def get_compatible_pixel_format(pix_fmt: str) -> str:
    """Convert deprecated pixel formats to their compatible alternatives.
    
    Args:
        pix_fmt: Input pixel format
        
    Returns:
        Compatible pixel format
    """
    if pix_fmt == PixelFormat.YUVJ420P.value:
        return PixelFormat.YUV420P.value
    elif pix_fmt == PixelFormat.YUVJ422P.value:
        return PixelFormat.YUV422P.value
    elif pix_fmt == PixelFormat.YUVJ444P.value:
        return PixelFormat.YUV444P.value
    elif pix_fmt == PixelFormat.YUVJ440P.value:
        return PixelFormat.YUV440P.value
    else:
        return pix_fmt

def align_with_gentle(audio_file: str, transcript_file: str) -> Optional[Dict[str, Any]]:
    """Aligns audio and text using Gentle.
    
    Args:
        audio_file: Path to the audio file
        transcript_file: Path to the transcript file
        
    Returns:
        Alignment result dictionary or None if failed
    """
    url = 'http://localhost:8765/transcriptions?async=false'
    try:
        with open(audio_file, 'rb') as audio, open(transcript_file, 'r') as transcript:
            files = {
                'audio': audio,
                'transcript': transcript
            }
            response = requests.post(url, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error communicating with Gentle: {e}")
        return None

def format_time(seconds: float) -> str:
    """Format time in seconds to HH:MM:SS,mmm format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    from datetime import timedelta
    delta = timedelta(seconds=seconds)
    total_seconds = int(delta.total_seconds())
    millis = int((delta.total_seconds() - total_seconds) * 1000)
    time_str = str(delta)
    if '.' in time_str:
        time_str, _ = time_str.split('.')
    time_str = time_str.zfill(8)  # Ensure at least HH:MM:SS
    return f"{time_str},{millis:03d}"

def format_ass_time(seconds: float) -> str:
    """Format time in seconds to ASS subtitle format (h:mm:ss.cc).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int((secs - int(secs)) * 100)
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"

def wrap_text(text: str, max_width: int) -> str:
    """Wrap text to multiple lines with a maximum width.
    
    Args:
        text: Text to wrap
        max_width: Maximum width in characters
        
    Returns:
        Wrapped text with ASS line breaks
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '\\N'.join(lines)  # ASS format line break 