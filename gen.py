async def youtube_shorts_workflow(topic: str, time_frame: str, video_length: int, user_script: str = None, gameplay_folder: str = "./gameplay") -> Dict[str, Any]:
    # Create graph instance
    graph = Graph()  # Create an instance of the Graph class
    video_length = video_length * 1000  # Convert to milliseconds
    results = {}

    # Create nodes
    recent_events_node = Node(agent=RecentEventsResearchAgent())
    title_gen_node = Node(agent=TitleGenerationAgent())
    title_select_node = Node(agent=TitleSelectionAgent())
    desc_gen_node = Node(agent=DescriptionGenerationAgent())
    hashtag_tag_node = Node(agent=HashtagAndTagGenerationAgent())
    script_gen_node = Node(agent=VideoScriptGenerationAgent())
    gameplay_node = Node(agent=GameplayVideoAgent())  # Use gameplay videos instead of generated images
    storyboard_gen_node = Node(agent=StoryboardGenerationAgent())

    # Set up gameplay processor
    gameplay_agent = gameplay_node.agent
    gameplay_agent.set_gameplay_folder(gameplay_folder)

    # Add nodes to graph
    graph.add_node(recent_events_node)
    graph.add_node(title_gen_node)
    graph.add_node(title_select_node)
    graph.add_node(desc_gen_node)
    graph.add_node(hashtag_tag_node)
    graph.add_node(script_gen_node)
    graph.add_node(gameplay_node)
    graph.add_node(storyboard_gen_node)

    # Create and add edges
    graph.add_edge(Edge(recent_events_node, title_gen_node))
    graph.add_edge(Edge(title_gen_node, title_select_node))
    graph.add_edge(Edge(title_select_node, desc_gen_node))
    graph.add_edge(Edge(desc_gen_node, hashtag_tag_node))
    graph.add_edge(Edge(hashtag_tag_node, script_gen_node))
    graph.add_edge(Edge(script_gen_node, storyboard_gen_node))
    graph.add_edge(Edge(storyboard_gen_node, gameplay_node))

    logger.info(f"Running workflow for topic {topic} and time frame {time_frame}")
    logger.info(f"Using gameplay videos from {gameplay_folder}")
    
    if user_script and user_script.strip():
        # Skip RecentEventsResearchAgent and use the provided script as research_result
        research_result = user_script.strip()
        results["User Script Provided"] = user_script
        logger.info("User provided a script. Skipping RecentEventsResearchAgent.")
    else:
        # Step 1: Recent Events Research Agent
        input_data = {"topic": topic, "time_frame": time_frame}
        try:
            research_result = await recent_events_node.process(input_data)
            results[recent_events_node.agent.name] = research_result
        except Exception as e:
            logger.error(f"Error in RecentEventsResearchAgent: {str(e)}")
            results["Error"] = f"RecentEventsResearchAgent failed: {str(e)}"
            return results

    # Step 2: Title Generation Agent
    try:
        title_gen_result = await title_gen_node.process(research_result)
        results[title_gen_node.agent.name] = title_gen_result
    except Exception as e:
        logger.error(f"Error in TitleGenerationAgent: {str(e)}")
        results["Error"] = f"TitleGenerationAgent failed: {str(e)}"
        return results

    # Step 3: Title Selection Agent
    try:
        title_select_result = await title_select_node.process(title_gen_result)
        results[title_select_node.agent.name] = title_select_result
    except Exception as e:
        logger.error(f"Error in TitleSelectionAgent: {str(e)}")
        results["Error"] = f"TitleSelectionAgent failed: {str(e)}"
        return results

    # Extract the selected title from the title selection result
    selected_title = extract_selected_title(title_select_result)
    results["Selected Title"] = selected_title

    # Step 4: Description Generation Agent
    try:
        desc_gen_result = await desc_gen_node.process(selected_title)
        results[desc_gen_node.agent.name] = desc_gen_result
    except Exception as e:
        logger.error(f"Error in DescriptionGenerationAgent: {str(e)}")
        results["Error"] = f"DescriptionGenerationAgent failed: {str(e)}"
        return results

    # Step 5: Hashtag and Tag Generation Agent
    try:
        hashtag_tag_result = await hashtag_tag_node.process(selected_title)
        results[hashtag_tag_node.agent.name] = hashtag_tag_result
    except Exception as e:
        logger.error(f"Error in HashtagAndTagGenerationAgent: {str(e)}")
        results["Error"] = f"HashtagAndTagGenerationAgent failed: {str(e)}"
        return results

    # Step 6: Video Script Generation Agent
    if user_script and user_script.strip():
        script_gen_result = user_script.strip()
        results[script_gen_node.agent.name] = "User provided script."
        logger.info("Using user-provided script.")
    else:
        try:
            script_gen_input = {"research": research_result}
            script_gen_result = await script_gen_node.process(script_gen_input)
            results[script_gen_node.agent.name] = script_gen_result
        except Exception as e:
            logger.error(f"Error in VideoScriptGenerationAgent: {str(e)}")
            results["Error"] = f"VideoScriptGenerationAgent failed: {str(e)}"
            return results

    # Step 7: Storyboard Generation Agent
    logger.info("Executing Storyboard Generation Agent")
    storyboard_gen_input = {
        "script": script_gen_result,
    }
    storyboard_gen_result = await storyboard_gen_node.process(storyboard_gen_input)
    if storyboard_gen_result is None:
        raise ValueError("Storyboard Generation Agent returned None")
    results[storyboard_gen_node.agent.name] = storyboard_gen_result
    
    # Step 8: Gameplay Video Agent
    logger.info("Executing Gameplay Video Agent")
    gameplay_input = {"scenes": storyboard_gen_result}
    gameplay_result = await gameplay_node.process(gameplay_input)
    if gameplay_result is None:
        raise ValueError("Gameplay Video Agent returned None")
    results[gameplay_node.agent.name] = gameplay_result

    # Update storyboard with gameplay video clips and calculate scene durations
    total_duration = 0
    for i, (scene, gameplay) in enumerate(zip(storyboard_gen_result, gameplay_result)):
        if gameplay is not None and 'video_path' in gameplay:
            scene['video_path'] = gameplay['video_path']
            # Calculate scene duration based on word count or use a default duration
            word_count = len(scene.get('script', '').split())
            scene['duration'] = max(word_count * 0.5, 3.0)  # Assume 0.5 seconds per word, minimum 3 seconds
            total_duration += scene['duration']
        else:
            logger.warning(f"No gameplay video for scene {scene.get('number', i+1)}")

    # Adjust scene durations to match target video length
    target_duration = video_length / 1000  # Convert video_length to seconds
    duration_factor = target_duration / total_duration
    for scene in storyboard_gen_result:
        scene['adjusted_duration'] = scene['duration'] * duration_factor
    
    logger.info(f"Target duration: {target_duration} seconds")
    logger.info(f"Total calculated duration: {total_duration} seconds")
    logger.info(f"Duration factor: {duration_factor}")

    # Filter out scenes without gameplay videos
    valid_scenes = [scene for scene in storyboard_gen_result if 'video_path' in scene]

    if not valid_scenes:
        raise ValueError("No valid scenes with gameplay videos remaining")

    # Log scene information
    for i, scene in enumerate(valid_scenes):
        logger.info(f"Scene {i}: Duration = {scene['duration']:.2f}s, Adjusted Duration = {scene['adjusted_duration']:.2f}s, Video = {scene['video_path']}")

    # Proceed to generate voiceover and compile video
    temp_dir = tempfile.mkdtemp()
    audio_file = os.path.join(temp_dir, "voiceover.mp3")
    
    try:
        # Generate voiceover
        if not generate_voiceover(valid_scenes, audio_file):
            raise Exception("Failed to generate voiceover")
        
        # Compile the final video
        output_path = compile_youtube_short(scenes=valid_scenes, audio_file=audio_file)
        if output_path:
            logger.info(f"YouTube Short saved as '{output_path}'")
            results["Output Video Path"] = output_path
        else:
            logger.error("Failed to compile YouTube Short")
            results["Output Video Path"] = None
    finally:
        # Clean up
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Error removing temporary directory {temp_dir}: {str(e)}")
    
    return resultsclass GameplayVideoAgent(Agent):
    """Agent that processes gameplay videos for each scene"""
    def __init__(self):
        super().__init__("Gameplay Video Agent", "local")
        self.processor = None
        
    def set_gameplay_folder(self, gameplay_folder: str):
        """Set the folder containing gameplay videos"""
        self.processor = GameplayVideoProcessor(gameplay_folder)
        
    async def execute(self, input_data: Dict[str, Any]) -> Any:
        scenes = input_data.get('scenes', [])
        results = []
        
        if not self.processor:
            logger.error("Gameplay folder not set. Call set_gameplay_folder() first.")
            return []
            
        if not self.processor.video_files:
            logger.error("No gameplay videos found. Please check the gameplay folder.")
            return []
        
        temp_dir = tempfile.mkdtemp()
        
        for i, scene in enumerate(scenes):
            try:
                # Get scene duration or use default
                duration = scene.get('adjusted_duration', DEFAULT_SCENE_DURATION)
                
                # Get a random segment from a gameplay video
                video_path, start_time, end_time = self.processor.get_random_segment(duration)
                
                if not video_path:
                    logger.error(f"Failed to get video segment for scene {i+1}")
                    results.append(None)
                    continue
                
                # Extract the segment
                output_path = os.path.join(temp_dir, f"scene_{i+1}.mp4")
                if self.processor.extract_segment(video_path, start_time, end_time, output_path):
                    # Success - add to results
                    results.append({
                        'video_path': output_path,
                        'original_video': video_path,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                    logger.info(f"Processed gameplay for scene {i+1}/{len(scenes)}")
                else:
                    logger.error(f"Failed to extract segment for scene {i+1}")
                    results.append(None)
            
            except Exception as e:
                logger.error(f"Error processing gameplay for scene {i+1}: {str(e)}")
                results.append(None)
        
        logger.info(f"Gameplay processing completed. Processed {len([r for r in results if r is not None])}/{len(scenes)} scenes.")
        return resultsimport streamlit as st
import asyncio
import aiohttp
import toml
import glob
import tempfile
import subprocess
import base64
import torch
import os
import re
import requests
import spacy
import datetime
import json
import logging
import shutil
from typing import List, Dict, Any, Tuple, Callable, Optional
from abc import ABC, abstractmethod
from enum import Enum
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from moviepy.editor import *
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load spaCy model (will download it if not already present)
nlp = spacy.load("en_core_web_md")

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
YOUTUBE_SHORT_RESOLUTION = (1080, 1920)
MAX_SCENE_DURATION = 5
DEFAULT_SCENE_DURATION = 1
SUBTITLE_FONT_SIZE = 13
SUBTITLE_FONT_COLOR = "yellow@0.5"
SUBTITLE_ALIGNMENT = 2  # Centered horizontally and vertically
SUBTITLE_BOLD = True
SUBTITLE_OUTLINE_COLOR = "&H40000000"  # Black with 50% transparency
SUBTITLE_BORDER_STYLE = 3
FALLBACK_SCENE_COLOR = "red"
FALLBACK_SCENE_TEXT_COLOR = "yellow@0.5"
FALLBACK_SCENE_BOX_COLOR = "black@0.5"
FALLBACK_SCENE_BOX_BORDER_WIDTH = 5
FALLBACK_SCENE_FONT_SIZE = 30
FALLBACK_SCENE_FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Common font on Linux systems

# Model settings - using Hugging Face open source models
DEFAULT_TEXT_GENERATION_MODEL = "meta-llama/Llama-3-8b-chat-hf"  # For small/fast generations
DEFAULT_LONG_TEXT_GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # For detailed content
STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"  # For image generation

# Helper functions
class PixelFormat(Enum):
    YUVJ420P = 'yuvj420p'
    YUVJ422P = 'yuvj422p'
    YUVJ444P = 'yuvj444p'
    YUVJ440P = 'yuvj440p'
    YUV420P = 'yuv420p'
    YUV422P = 'yuv422p'
    YUV444P = 'yuv444p'
    YUV440P = 'yuv440p'

def get_compatible_pixel_format(pix_fmt: str) -> str:
    """Convert deprecated pixel formats to their compatible alternatives."""
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

def align_with_gentle(audio_file: str, transcript_file: str) -> dict:
    """Aligns audio and text using Gentle and returns the alignment result."""
    url = 'http://localhost:8765/transcriptions?async=false'
    files = {
        'audio': open(audio_file, 'rb'),
        'transcript': open(transcript_file, 'r')
    }
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Gentle: {e}")
        return None

def gentle_alignment_to_ass(gentle_alignment: dict, ass_file: str):
    """Converts Gentle alignment JSON to ASS subtitle format with styling."""
    with open(ass_file, 'w', encoding='utf-8') as f:
        # Write ASS header
        f.write("""[Script Info]
Title: Generated by Gentle Alignment
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, 
Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, 
MarginV, Encoding
Style: Default,Verdana,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,{bold},0,0,0,100,100,0,0,1,1,0,{alignment},2,2,2,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n""".format(
    font_size=SUBTITLE_FONT_SIZE, bold=int(SUBTITLE_BOLD), alignment=SUBTITLE_ALIGNMENT))
        
        words = gentle_alignment.get('words', [])
        i = 0
        while i < len(words):
            start = words[i].get('start')
            if start is None:
                i += 1
                continue
            end = words[i].get('end')
            text_words = []
            colors = []
            for j in range(2):  # Get up to 2 words
                if i + j < len(words):
                    word_info = words[i + j]
                    word_text = word_info.get('word', '')
                    text_words.append(word_text)
                    if j == 0:
                        # First word in dark orange
                        colors.append(r'{\c&H0080FF&}')  # Dark orange color code in ASS (BGR order)
                    else:
                        colors.append(r'{\c&HFFFFFF&}')  # White color code
                else:
                    break
            dialogue_text = ''.join(f"{colors[k]}{text_words[k]} " for k in range(len(text_words))).strip()
            end = words[min(i + len(text_words) - 1, len(words) - 1)].get('end', end)
            if end is None:
                i += len(text_words)
                continue

            start_time = format_ass_time(start)
            end_time = format_ass_time(end)
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{dialogue_text}\n")
            i += len(text_words)

def wrap_text(text, max_width):
    """Wraps text to multiple lines with a maximum width."""
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

    return '\\N'.join(lines)  # Include all lines

def format_ass_time(seconds: float) -> str:
    """Formats time in seconds to ASS subtitle format (h:mm:ss.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int((secs - int(secs)) * 100)
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"

def format_time(seconds: float) -> str:
    """Formats time in seconds to HH:MM:SS,mmm format for subtitles."""
    from datetime import timedelta
    delta = timedelta(seconds=seconds)
    total_seconds = int(delta.total_seconds())
    millis = int((delta.total_seconds() - total_seconds) * 1000)
    time_str = str(delta)
    if '.' in time_str:
        time_str, _ = time_str.split('.')
    else:
        time_str = time_str
    time_str = time_str.zfill(8)  # Ensure at least HH:MM:SS
    return f"{time_str},{millis:03d}"

# Abstract classes for Agents and Tools
class Agent(ABC):
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        pass

class Tool(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def use(self, input_data: Any) -> Any:
        pass

class VoiceModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_voice(self, text: str, output_file: str):
        pass
        
# Open source TTS module using pyttsx3 (offline engine) or Mozilla TTS
class OpenSourceTTS(VoiceModule):
    def __init__(self, engine_type="pyttsx3"):
        super().__init__()
        self.engine_type = engine_type
        
        if engine_type == "pyttsx3":
            import pyttsx3
            self.engine = pyttsx3.init()
            # Set properties
            self.engine.setProperty('rate', 175)  # Speed of speech
            self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            # Get available voices and set to a more natural one if available
            voices = self.engine.getProperty('voices')
            for voice in voices:
                # Try to find a good female voice
                if "female" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        elif engine_type == "mozilla":
            # Mozilla TTS setup would go here if available
            # This requires more setup but produces better quality
            # Using pyttsx3 as fallback for now
            logger.warning("Mozilla TTS not configured, falling back to pyttsx3")
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 175)
            self.engine.setProperty('volume', 1.0)
            self.engine_type = "pyttsx3"  # Fallback
        else:
            raise ValueError(f"Unsupported TTS engine type: {engine_type}")

    def generate_voice(self, text: str, output_file: str):
        """Generate speech from text and save to file"""
        if self.engine_type == "pyttsx3":
            # pyttsx3 directly saves to file
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            return output_file
        elif self.engine_type == "mozilla":
            # Mozilla TTS implementation would go here
            pass
        
        return output_file

# Node and Edge classes for graph representation
class Node:
    def __init__(self, agent: Agent = None, tool: Tool = None):
        self.agent = agent
        self.tool = tool
        self.edges: List['Edge'] = []

    async def process(self, input_data: Any) -> Any:
        if self.agent:
            return await self.agent.execute(input_data)
        elif self.tool:
            return await self.tool.use(input_data)
        else:
            raise ValueError("Node has neither agent nor tool")

class Edge:
    def __init__(self, source: Node, target: Node, condition: Callable[[Any], bool] = None):
        self.source = source
        self.target = target
        self.condition = condition

class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        edge.source.edges.append(edge)

# Local LLM Text Generation using transformers
class LocalLLMGenerator:
    def __init__(self, model_name: str = DEFAULT_TEXT_GENERATION_MODEL):
        self.model_name = model_name
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info(f"Loaded model {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    async def generate(self, prompt: str, system_prompt: str = None, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Format prompt based on model type
            if "llama" in self.model_name.lower():
                full_prompt = f"<s>[INST] {system_prompt or ''}\n\n{prompt} [/INST]"
            elif "mistral" in self.model_name.lower():
                full_prompt = f"<s>[INST] {system_prompt or ''}\n\n{prompt} [/INST]"
            else:
                # Generic instruction format
                full_prompt = f"System: {system_prompt or 'You are a helpful assistant.'}\n\nUser: {prompt}\n\nAssistant:"
            
            tokens = self.tokenizer(full_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    **tokens,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
            
            output = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
            
            # Extract just the generated response (remove the prompt)
            response = output.split("Assistant:")[-1] if "Assistant:" in output else output.split("[/INST]")[-1]
            response = response.strip()
            
            logger.info(f"Generated text with {len(response)} characters")
            return response
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"

# Web search tool using SerpAPI (free alternative)
class WebSearchTool(Tool):
    def __init__(self):
        super().__init__("Web Search Tool")
        # We'll use a custom RSS feed parser to get news instead of a paid API
        
    async def use(self, input_data: str, time_period: str = 'all') -> Dict[str, Any]:
        try:
            # Use RSS feeds from news sites for recent content
            feeds = [
                "http://rss.cnn.com/rss/cnn_topstories.rss",
                "http://feeds.bbci.co.uk/news/rss.xml",
                "https://www.reddit.com/r/news/.rss",
                "https://news.google.com/rss",
                "https://feeds.npr.org/1001/rss.xml"
            ]
            
            all_items = []
            search_terms = input_data.lower().split()
            
            async with aiohttp.ClientSession() as session:
                for feed_url in feeds:
                    try:
                        async with session.get(feed_url) as response:
                            if response.status == 200:
                                content = await response.text()
                                # Simple RSS parsing (can be improved with dedicated XML parser)
                                item_matches = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
                                
                                for item in item_matches:
                                    title_match = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
                                    description_match = re.search(r'<description>(.*?)</description>', item, re.DOTALL)
                                    link_match = re.search(r'<link>(.*?)</link>', item, re.DOTALL)
                                    pubdate_match = re.search(r'<pubDate>(.*?)</pubDate>', item, re.DOTALL)
                                    
                                    title = title_match.group(1) if title_match else ""
                                    description = description_match.group(1) if description_match else ""
                                    link = link_match.group(1) if link_match else ""
                                    pubdate = pubdate_match.group(1) if pubdate_match else ""
                                    
                                    # Clean HTML from description
                                    description = re.sub(r'<.*?>', '', description)
                                    
                                    # Check if any search term is in title or description
                                    if any(term in title.lower() or term in description.lower() for term in search_terms):
                                        all_items.append({
                                            "title": title,
                                            "snippet": description,
                                            "link": link,
                                            "published_date": pubdate
                                        })
                    except Exception as e:
                        logger.error(f"Error fetching feed {feed_url}: {e}")
            
            # Sort items by relevance to search terms
            def calculate_relevance(item):
                score = 0
                for term in search_terms:
                    if term in item["title"].lower():
                        score += 3  # Higher weight for title matches
                    if term in item["snippet"].lower():
                        score += 1
                return score
            
            all_items.sort(key=calculate_relevance, reverse=True)
            
            # Format as expected by agents
            return {
                "organic_results": all_items[:20]  # Return top 20 results
            }
        except Exception as e:
            logger.error(f"Error in WebSearchTool: {str(e)}")
            return {"organic_results": []}

# Image generation using local Stable Diffusion
class GameplayVideoProcessor:
    def __init__(self, gameplay_folder: str):
        """
        Initializes a processor for gameplay videos
        
        Args:
            gameplay_folder: Path to the folder containing gameplay videos
        """
        self.gameplay_folder = gameplay_folder
        self.video_files = []
        self.scan_videos()
        
    def scan_videos(self):
        """Scans the gameplay folder for video files"""
        if not os.path.exists(self.gameplay_folder):
            logger.error(f"Gameplay folder does not exist: {self.gameplay_folder}")
            return
            
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        self.video_files = []
        
        for file in os.listdir(self.gameplay_folder):
            file_path = os.path.join(self.gameplay_folder, file)
            if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in video_extensions:
                self.video_files.append(file_path)
                
        logger.info(f"Found {len(self.video_files)} gameplay videos in {self.gameplay_folder}")
        
    def get_random_segment(self, duration: float) -> Tuple[str, float, float]:
        """
        Gets a random segment from a random gameplay video
        
        Args:
            duration: Desired duration of the segment in seconds
            
        Returns:
            Tuple of (video_path, start_time, end_time)
        """
        if not self.video_files:
            logger.error("No gameplay videos available")
            return None, 0, 0
            
        # Select a random video
        video_path = random.choice(self.video_files)
        
        try:
            # Get video duration
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            video_duration = float(result.stdout.strip())
            
            # Calculate random start time, ensuring we don't exceed video length
            max_start = max(0, video_duration - duration)
            if max_start <= 0:
                # Video is shorter than requested duration
                return video_path, 0, video_duration
                
            start_time = random.uniform(0, max_start)
            end_time = min(start_time + duration, video_duration)
            
            return video_path, start_time, end_time
            
        except Exception as e:
            logger.error(f"Error getting random segment from {video_path}: {e}")
            return video_path, 0, min(duration, 10)  # Default to first 10 seconds if error
            
    def extract_segment(self, video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
        """
        Extracts a segment from a video and saves it to output_path
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save the extracted segment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            duration = end_time - start_time
            command = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac',
                output_path
            ]
            
            subprocess.run(command, check=True)
            
            if os.path.exists(output_path):
                logger.info(f"Extracted {duration}s segment from {video_path} to {output_path}")
                return True
            else:
                logger.error(f"Failed to extract segment - output file does not exist: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting segment from {video_path}: {e}")
            return False

# Text-based agents using local LLM
class RecentEventsResearchAgent(Agent):
    def __init__(self):
        super().__init__("Recent Events Research Agent", DEFAULT_LONG_TEXT_GENERATION_MODEL)
        self.web_search_tool = WebSearchTool()
        self.llm_generator = LocalLLMGenerator(DEFAULT_LONG_TEXT_GENERATION_MODEL)

    async def execute(self, input_data: Dict[str, Any]) -> Any:
        topic = input_data['topic']
        time_frame = input_data['time_frame']
        video_length = input_data.get('video_length', 60)
        
        # Decide how many events to include based on video length
        max_events = min(5, video_length // 15)  # Rough estimate: 15 seconds per event
        
        search_query = f"{topic} events in the {time_frame}"
        search_results = await self.web_search_tool.use(search_query, time_frame)
        
        organic_results = search_results.get("organic_results", [])
        
        system_prompt = "You are an AI assistant embodying the expertise of a world-renowned investigative journalist specializing in creating viral and engaging content for social media platforms."
        
        prompt = f"""Your task is to analyze and summarize the most engaging and relevant {topic} events that occurred in the {time_frame}. Using the following search results, select the {max_events} most compelling cases:

Search Results:
{json.dumps(organic_results[:10], indent=2)}

For each selected event, provide a concise yet engaging summary suitable for a up to three minute faceless YouTube Shorts video script, that includes:

1. A vivid description of the event, highlighting its most unusual or attention-grabbing aspects
2. The precise date of occurrence
3. The specific location, including city and country if available
4. An expert analysis of why this event is significant, intriguing, or unexpected
5. A brief mention of the credibility of the information source (provide the URL)

Format your response as a numbered list, with each event separated by two newline characters.

Ensure your summaries are both informative and captivating, presented in a style suitable for a fast-paced, engaging faceless YouTube Shorts video narration."""

        return await self.llm_generator.generate(prompt, system_prompt, 2048, 0.7)

class TitleGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Title Generation Agent", DEFAULT_TEXT_GENERATION_MODEL)
        self.llm_generator = LocalLLMGenerator(DEFAULT_TEXT_GENERATION_MODEL)

    async def execute(self, input_data: Any) -> Any:
        research_result = input_data  # Accept research output
        
        system_prompt = "You are an expert in keyword strategy, copywriting, and a renowned YouTuber with over a decade of experience in crafting attention-grabbing titles for viral content."
        
        prompt = f"""Using the following research, generate 15 captivating, SEO-optimized YouTube Shorts titles:

Research:
{research_result}

Categorize them under appropriate headings:

- Beginning: 5 titles with the keyword at the beginning
- Middle: 5 titles with the keyword in the middle
- End: 5 titles with the keyword at the end

Ensure that the titles are:

- Attention-grabbing and suitable for faceless YouTube Shorts videos
- Optimized for SEO with high-ranking keywords relevant to the topic
- Crafted to maximize viewer engagement and encourage clicks

Present the titles clearly under each heading."""

        return await self.llm_generator.generate(prompt, system_prompt, 1024, 0.7)

class TitleSelectionAgent(Agent):
    def __init__(self):
        super().__init__("Title Selection Agent", DEFAULT_TEXT_GENERATION_MODEL)
        self.llm_generator = LocalLLMGenerator(DEFAULT_TEXT_GENERATION_MODEL)

    async def execute(self, input_data: Any) -> Any:
        generated_titles = input_data  # Accept generated titles
        
        system_prompt = "You are an AI assistant embodying the expertise of a top-tier YouTube content strategist with over 15 years of experience in video optimization, audience engagement, and title creation, particularly for YouTube Shorts."
        
        prompt = f"""You are an expert YouTube content strategist with over a decade of experience in video optimization and audience engagement, particularly specializing in YouTube Shorts. Your task is to analyze the following list of titles for a faceless YouTube Shorts video and select the most effective one:

{generated_titles}

Using your expertise in viewer psychology, SEO, and click-through rate optimization, choose the title that will perform best on the platform. Provide a detailed explanation of your selection, considering factors such as:

1. Immediate attention-grabbing potential, essential for short-form content
2. Keyword optimization for maximum discoverability
3. Emotional appeal to captivate viewers quickly
4. Clarity and conciseness appropriate for YouTube Shorts
5. Alignment with current YouTube Shorts trends and algorithms

Present your selected title and offer a comprehensive rationale for why this title stands out among the others. Ensure your explanation is clear and insightful, highlighting how the chosen title will drive engagement and views."""

        return await self.llm_generator.generate(prompt, system_prompt, 1024, 0.5)

class DescriptionGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Description Generation Agent", DEFAULT_LONG_TEXT_GENERATION_MODEL)
        self.llm_generator = LocalLLMGenerator(DEFAULT_LONG_TEXT_GENERATION_MODEL)

    async def execute(self, input_data: Any) -> Any:
        selected_title = input_data  # Accept selected title
        
        system_prompt = "You are an AI assistant taking on the role of a prodigy SEO copywriter and YouTube content creator with over 20 years of experience."
        
        prompt = f"""As a seasoned SEO copywriter and YouTube content creator with extensive experience in crafting engaging, algorithm-friendly video descriptions, your task is to compose a masterful 1000-character YouTube video description for a faceless YouTube Shorts video titled "{selected_title}". This description should:

1. Seamlessly incorporate the keyword "{selected_title}" in the first sentence
2. Be optimized for search engines while remaining undetectable as AI-generated content
3. Engage viewers and encourage them to watch the full video
4. Include relevant calls-to-action (e.g., subscribe, like, comment)
5. Utilize natural language and a conversational tone suitable for the target audience
6. Highlight how the video addresses a real-world problem or provides valuable insights to engage viewers

Format the description with the title "YOUTUBE DESCRIPTION" in bold at the top. Ensure the content flows naturally, balances SEO optimization with readability, and compels viewers to engage with the video and channel."""

        return await self.llm_generator.generate(prompt, system_prompt, 1024, 0.6)

class HashtagAndTagGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Hashtag and Tag Generation Agent", DEFAULT_TEXT_GENERATION_MODEL)
        self.llm_generator = LocalLLMGenerator(DEFAULT_TEXT_GENERATION_MODEL)

    async def execute(self, input_data: str) -> Any:
        selected_title = input_data  # Accept selected title
        
        system_prompt = "You are an AI assistant taking on the role of a leading YouTube SEO specialist and social media strategist with over 10 years of experience in optimizing video discoverability."
        
        prompt = f"""As a leading YouTube SEO specialist and social media strategist with a proven track record in optimizing video discoverability and virality, your task is to create an engaging and relevant set of hashtags and tags for the faceless YouTube Shorts video titled "{selected_title}". Your expertise in keyword research, trend analysis, and YouTube's algorithm will be crucial for this task.

Develop the following:

1. 10 trending, SEO-optimized hashtags that will maximize the video's reach and engagement on YouTube Shorts. Present the hashtags with the '#' symbol.

2. 35 high-value, low-competition SEO tags (keywords) to strategically boost the video's search ranking on YouTube.

In your selection process, prioritize:

- Relevance to the video title and content
- Potential search volume on YouTube Shorts
- Engagement potential (views, likes, comments)
- Current trends on YouTube Shorts
- Alignment with YouTube's recommendation algorithm for Shorts

Ensure all tags are separated by commas. Provide a brief explanation of your strategy for selecting these hashtags and tags, highlighting how they will contribute to the video's overall performance on YouTube Shorts."""

        return await self.llm_generator.generate(prompt, system_prompt, 1024, 0.6)

class VideoScriptGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Video Script Generation Agent", DEFAULT_LONG_TEXT_GENERATION_MODEL)
        self.llm_generator = LocalLLMGenerator(DEFAULT_LONG_TEXT_GENERATION_MODEL)

    async def execute(self, input_data: Dict[str, Any]) -> Any:
        research_result = input_data.get('research', '')
        video_length = input_data.get('video_length', 180)  # Default to 180 seconds if not specified
        
        system_prompt = "You are an AI assistant taking on the role of a leading YouTube content creator and SEO specialist with a deep understanding of audience engagement, particularly in creating faceless YouTube Shorts."
        
        prompt = f"""As a YouTube content creator specializing in faceless YouTube Shorts, craft a detailed, engaging, and enthralling script for a {video_length}-second vertical video based on the following information:

{research_result}

Your script should include:

1. An attention-grabbing opening hook that immediately captivates viewers
2. Key points from the research presented in a concise and engaging manner
3. A strong call-to-action conclusion to encourage viewer interaction (e.g., like, share, subscribe)

Ensure that the script is suitable for a faceless video, relying on voiceover narration and visual storytelling elements.

Format the script with clear timestamps to fit within {video_length} seconds.

Optimize for viewer retention and engagement, keeping in mind the fast-paced nature of YouTube Shorts."""

        return await self.llm_generator.generate(prompt, system_prompt, 2048, 0.7)

class StoryboardGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Storyboard Generation Agent", DEFAULT_LONG_TEXT_GENERATION_MODEL)
        self.llm_generator = LocalLLMGenerator(DEFAULT_LONG_TEXT_GENERATION_MODEL)
        self.nlp = nlp

    async def execute(self, input_data: Dict[str, Any]) -> Any:
        script = input_data.get('script', '')
        
        if not script:
            logger.error("No script provided for storyboard generation")
            return []

        system_prompt = "You are an AI assistant specializing in creating engaging and viral storyboards for faceless YouTube Shorts videos using the provided script."
        
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

        response = await self.llm_generator.generate(prompt, system_prompt, 2048, 0.7)
        
        logger.info(f"Raw storyboard response: {response}")
        scenes = self.parse_scenes(response)
        if not scenes:
            logger.error("Failed to generate valid storyboard scenes")
            return []
        
        return scenes
    
    def parse_scenes(self, response: str) -> List[Dict[str, Any]]:
        scenes = []
        current_scene = {}
        current_scene_number = None

        for line in response.split('\n'):
            line = line.strip()
            logger.debug(f"Processing line: {line}")

            if line.startswith(tuple(f"{i}." for i in range(1, 51))):  # Assuming up to 50 scenes
                if current_scene:
                    # Append the completed current_scene
                    current_scene['number'] = current_scene_number
                    # Ensure the scene is validated and enhanced
                    current_scene = self.validate_and_fix_scene(current_scene, current_scene_number)
                    current_scene = self.enhance_scene_keywords(current_scene)
                    scenes.append(current_scene)
                    logger.debug(f"Scene {current_scene_number} appended to scenes list")
                    current_scene = {}

                try:
                    # Start a new scene
                    current_scene_number = int(line.split('.', 1)[0])
                    logger.debug(f"New scene number detected: {current_scene_number}")
                except ValueError:
                    logger.warning(f"Invalid scene number format: {line}")
                    continue  # Skip this line and move to the next
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                current_scene[key] = value
                logger.debug(f"Key-value pair added to current scene: {key}:{value}")
            else:
                logger.warning(f"Line format not recognized: {line}")

        # After looping through all lines, check if there is an unfinished scene
        if current_scene:
            current_scene['number'] = current_scene_number
            current_scene = self.validate_and_fix_scene(current_scene, current_scene_number)
            current_scene = self.enhance_scene_keywords(current_scene)
            scenes.append(current_scene)
            logger.debug(f"Final scene {current_scene_number} appended to scenes list")

        logger.info(f"Parsed and enhanced scenes: {scenes}")
        return scenes
    
    def enhance_scene_keywords(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        # Extract keywords from narration_text and visual descriptions
        narration_doc = self.nlp(scene.get('narration_text', ''))
        visual_doc = self.nlp(scene.get('visual', ''))

        # Function to extract nouns and named entities
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

    def validate_and_fix_scene(self, scene: Dict[str, Any], scene_number: int) -> Dict[str, Any]:
        # Ensure 'number' key is present in the scene dictionary
        scene['number'] = scene_number

        required_keys = ['visual', 'text', 'video_keyword', 'image_keyword']
        for key in required_keys:
            if key not in scene:
                if key == 'visual':
                    scene[key] = f"Visual representation of scene {scene_number}"
                elif key == 'text':
                    scene[key] = ""
                elif key == 'video_keyword':
                    scene[key] = f"video scene {scene_number}"
                elif key == 'image_keyword':
                    scene[key] = f"image scene {scene_number}"
                logger.warning(f"Added missing {key} for scene {scene_number}")

        # Clean the 'text' field by removing leading/trailing quotation marks
        text = scene.get('text', '')
        text = text.strip('"').strip("'")
        scene['text'] = text

        # Copy the cleaned text into 'narration_text'
        scene['narration_text'] = text

        return scene
