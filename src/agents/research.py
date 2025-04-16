"""
Research agents for content discovery and analysis.
"""
import logging
from typing import Dict, Any

from ..constants import DEFAULT_LONG_TEXT_GENERATION_MODEL
from ..tools.web_search import WebSearchTool
from .base import ResearchAgent

logger = logging.getLogger(__name__)

class RecentEventsResearchAgent(ResearchAgent):
    """Agent for researching recent events on a given topic.
    
    This agent uses web search to find and analyze recent events,
    producing a detailed summary suitable for video content.
    """
    def __init__(self):
        super().__init__(
            "Recent Events Research Agent",
            DEFAULT_LONG_TEXT_GENERATION_MODEL
        )
        self.web_search_tool = WebSearchTool()

    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute the research task.
        
        Args:
            input_data: Dictionary containing:
                - topic: The topic to research
                - time_frame: Time frame to search within
                - video_length: Target video length in seconds
                
        Returns:
            Detailed research summary
            
        Raises:
            Exception: If research fails
        """
        try:
            topic = input_data['topic']
            time_frame = input_data['time_frame']
            video_length = input_data.get('video_length', 60)
            
            # Decide how many events to include based on video length
            max_events = min(5, video_length // 15)  # ~15 seconds per event
            
            # Search for recent events
            search_query = f"{topic} events in the {time_frame}"
            search_results = await self.web_search_tool.use(search_query, time_frame)
            
            organic_results = search_results.get("organic_results", [])
            
            if not organic_results:
                logger.warning(f"No results found for {search_query}")
                return "No relevant events found for the given topic and time frame."
            
            # Generate research summary
            system_prompt = (
                "You are an AI assistant embodying the expertise of a "
                "world-renowned investigative journalist specializing in "
                "creating viral and engaging content for social media platforms."
            )
            
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

            return await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.7
            )
            
        except KeyError as e:
            logger.error(f"Missing required input: {e}")
            raise ValueError(f"Missing required input: {e}")
        except Exception as e:
            logger.error(f"Research failed: {e}")
            raise 