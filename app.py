"""
Simple Streamlit app for video text overlay and dictation.
"""
import os
import tempfile
import logging
from datetime import datetime
import streamlit as st
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pyttsx3
import requests

# Setup logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"video_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log system info
logger.info("Starting Video Text Overlay & Dictation App")
logger.info(f"Python version: {os.sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

def init_tts_engine():
    """Initialize the TTS engine."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
        logger.info("TTS engine initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {str(e)}")
        raise

def generate_audio(text: str, engine) -> str:
    """Generate audio from text using TTS."""
    try:
        temp_audio = tempfile.mktemp(suffix='.wav')
        logger.info(f"Generating audio for text: {text[:50]}...")
        engine.save_to_file(text, temp_audio)
        engine.runAndWait()
        logger.info(f"Audio generated successfully: {temp_audio}")
        return temp_audio
    except Exception as e:
        logger.error(f"Failed to generate audio: {str(e)}")
        raise

def process_video(video_path: str, text: str, audio_path: str) -> str:
    """Process video with text overlay and audio."""
    try:
        logger.info(f"Starting video processing: {video_path}")
        logger.info(f"Text overlay: {text[:50]}...")
        
        # Get video info before processing
        video_info = VideoFileClip(video_path)
        logger.info(f"Input video details - Duration: {video_info.duration}s, Size: {video_info.size}")
        video_info.close()
        
        # Load video with lower quality for faster processing
        video = VideoFileClip(video_path, target_resolution=(720, None))
        logger.info(f"Video loaded with resolution: {video.size}")
        
        # Create text overlay
        text_clip = TextClip(
            text, 
            fontsize=70, 
            color='white',
            stroke_color='black',
            stroke_width=2,
            size=(video.w * 0.8, None),  # Width is 80% of video width
            method='caption'
        ).set_duration(video.duration)
        logger.info("Text overlay created")
        
        # Position text at the center
        text_clip = text_clip.set_position(('center', 'center'))
        
        # Load audio
        logger.info("Loading audio tracks")
        audio = VideoFileClip(video_path).audio
        tts_audio = VideoFileClip(audio_path).audio
        
        # Combine video, text and audio
        logger.info("Compositing video with text and audio")
        final_clip = CompositeVideoClip([video, text_clip])
        final_clip = final_clip.set_audio(tts_audio)
        
        # Save result with optimized settings
        output_path = tempfile.mktemp(suffix='.mp4')
        logger.info(f"Writing final video to: {output_path}")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='faster',  # Faster encoding
            threads=4,  # Use multiple threads
            bitrate='2000k',  # Lower bitrate for smaller file size
            logger=None  # Disable moviepy's internal logging
        )
        
        # Log output video info
        output_info = VideoFileClip(output_path)
        logger.info(f"Output video details - Duration: {output_info.duration}s, Size: {output_info.size}")
        output_info.close()
        
        # Clean up
        logger.info("Cleaning up resources")
        video.close()
        text_clip.close()
        final_clip.close()
        
        logger.info("Video processing completed successfully")
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise

def query_llm(prompt: str, endpoint_type: str, endpoint: str = "http://localhost:11434") -> str:
    """Query local LLM for text generation."""
    try:
        logger.info(f"Querying {endpoint_type} LLM with prompt: {prompt[:50]}...")
        
        if endpoint_type == "Ollama":
            # Query Ollama
            response = requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False
                }
            )
            result = response.json()['response']
            
        elif endpoint_type == "LM Studio":
            # Query LM Studio
            response = requests.post(
                endpoint,
                json={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
            result = response.json()['choices'][0]['text']
            
        logger.info(f"LLM response received: {result[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}", exc_info=True)
        raise

def main():
    try:
        st.title("Video Text Overlay & Dictation")
        logger.info("Application started")
        
        # Sidebar for LLM settings
        st.sidebar.header("LLM Settings")
        endpoint_type = st.sidebar.selectbox(
            "LLM Type",
            ["None", "Ollama", "LM Studio"]
        )
        logger.info(f"Selected LLM type: {endpoint_type}")
        
        if endpoint_type == "LM Studio":
            endpoint = st.sidebar.text_input(
                "LM Studio Endpoint",
                "http://localhost:1234/v1/completions"
            )
        elif endpoint_type == "Ollama":
            endpoint = st.sidebar.text_input(
                "Ollama Endpoint",
                "http://localhost:11434"
            )
        
        # Video settings in sidebar
        st.sidebar.header("Video Settings")
        target_resolution = st.sidebar.selectbox(
            "Output Resolution",
            ["Original", "720p", "480p"],
            index=1
        )
        logger.info(f"Selected output resolution: {target_resolution}")
        
        # Main interface
        st.info("Supports video files up to 2GB")
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
        
        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Create temp directory for processing
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")
            temp_video = os.path.join(temp_dir, "input_video.mp4")
            
            # Save uploaded file with progress bar
            with st.spinner("Uploading video..."):
                with open(temp_video, 'wb') as f:
                    # Write in chunks to handle large files
                    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
                    file_size = 0
                    progress_bar = st.progress(0)
                    chunks_written = 0
                    
                    while True:
                        chunk = uploaded_file.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        file_size += len(chunk)
                        chunks_written += 1
                        # Update progress bar
                        progress = min(file_size / uploaded_file.size, 1.0)
                        progress_bar.progress(progress)
                        
                        if chunks_written % 10 == 0:  # Log every 10MB
                            logger.info(f"Upload progress: {progress*100:.1f}% ({file_size/(1024*1024):.1f}MB)")
                    
                    logger.info(f"File saved to temporary location: {temp_video}")
            
            # Show video preview
            st.video(temp_video)
            
            # Text input
            if endpoint_type != "None":
                prompt = st.text_input("Enter prompt for text generation")
                if st.button("Generate Text") and prompt:
                    logger.info("Text generation requested")
                    with st.spinner("Generating text..."):
                        text = query_llm(prompt, endpoint_type, endpoint)
                        st.session_state['generated_text'] = text
                        st.text_area("Generated Text", text)
            
            text = st.text_area(
                "Enter text to overlay",
                value=st.session_state.get('generated_text', ''),
                height=100
            )
            
            if st.button("Process Video") and text:
                logger.info("Video processing requested")
                with st.spinner("Processing video... This may take a while for large files"):
                    try:
                        # Initialize TTS
                        engine = init_tts_engine()
                        
                        # Generate audio
                        audio_path = generate_audio(text, engine)
                        
                        # Process video
                        output_path = process_video(temp_video, text, audio_path)
                        
                        # Show result
                        st.success("Processing complete!")
                        st.video(output_path)
                        
                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                "Download Video",
                                f,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                        logger.info("Video processing and download setup completed")
                        
                    except Exception as e:
                        logger.error(f"Error in video processing pipeline: {str(e)}", exc_info=True)
                        st.error(f"Error processing video: {str(e)}")
                    finally:
                        # Cleanup temp files
                        try:
                            if os.path.exists(temp_dir):
                                for file in os.listdir(temp_dir):
                                    os.remove(os.path.join(temp_dir, file))
                                os.rmdir(temp_dir)
                                logger.info(f"Cleaned up temporary directory: {temp_dir}")
                        except Exception as e:
                            logger.warning(f"Error cleaning up temporary files: {str(e)}")
                            st.warning(f"Error cleaning up temporary files: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main() 