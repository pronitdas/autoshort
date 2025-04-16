"""
Content generation agents for creating video metadata and scripts.
"""
import logging
from typing import Dict, Any

from ..constants import (
    DEFAULT_TEXT_GENERATION_MODEL,
    DEFAULT_LONG_TEXT_GENERATION_MODEL
)
from .base import GenerationAgent, ProcessingAgent

logger = logging.getLogger(__name__)

class TitleGenerationAgent(GenerationAgent):
    """Agent for generating engaging video titles.
    
    This agent creates SEO-optimized titles suitable for YouTube Shorts.
    """
    def __init__(self):
        super().__init__(
            "Title Generation Agent",
            DEFAULT_TEXT_GENERATION_MODEL
        )

    async def execute(self, input_data: str) -> str:
        """Generate video titles.
        
        Args:
            input_data: Research summary to base titles on
            
        Returns:
            Generated titles with explanations
            
        Raises:
            Exception: If title generation fails
        """
        try:
            system_prompt = (
                "You are an expert in keyword strategy, copywriting, and a "
                "renowned YouTuber with over a decade of experience in "
                "crafting attention-grabbing titles for viral content."
            )
            
            prompt = f"""Using the following research, generate 15 captivating, SEO-optimized YouTube Shorts titles:

Research:
{input_data}

Categorize them under appropriate headings:

- Beginning: 5 titles with the keyword at the beginning
- Middle: 5 titles with the keyword in the middle
- End: 5 titles with the keyword at the end

Ensure that the titles are:

- Attention-grabbing and suitable for faceless YouTube Shorts videos
- Optimized for SEO with high-ranking keywords relevant to the topic
- Crafted to maximize viewer engagement and encourage clicks

Present the titles clearly under each heading."""

            return await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.7
            )
            
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            raise

class TitleSelectionAgent(GenerationAgent):
    """Agent for selecting the best title from generated options.
    
    This agent analyzes and selects the most effective title based on
    SEO and engagement potential.
    """
    def __init__(self):
        super().__init__(
            "Title Selection Agent",
            DEFAULT_TEXT_GENERATION_MODEL
        )

    async def execute(self, input_data: str) -> str:
        """Select the best title.
        
        Args:
            input_data: Generated titles to choose from
            
        Returns:
            Selected title with explanation
            
        Raises:
            Exception: If title selection fails
        """
        try:
            system_prompt = (
                "You are an AI assistant embodying the expertise of a "
                "top-tier YouTube content strategist with over 15 years "
                "of experience in video optimization, audience engagement, "
                "and title creation, particularly for YouTube Shorts."
            )
            
            prompt = f"""You are an expert YouTube content strategist with over a decade of experience in video optimization and audience engagement, particularly specializing in YouTube Shorts. Your task is to analyze the following list of titles for a faceless YouTube Shorts video and select the most effective one:

{input_data}

Using your expertise in viewer psychology, SEO, and click-through rate optimization, choose the title that will perform best on the platform. Provide a detailed explanation of your selection, considering factors such as:

1. Immediate attention-grabbing potential, essential for short-form content
2. Keyword optimization for maximum discoverability
3. Emotional appeal to captivate viewers quickly
4. Clarity and conciseness appropriate for YouTube Shorts
5. Alignment with current YouTube Shorts trends and algorithms

Present your selected title and offer a comprehensive rationale for why this title stands out among the others. Ensure your explanation is clear and insightful, highlighting how the chosen title will drive engagement and views."""

            return await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.5
            )
            
        except Exception as e:
            logger.error(f"Title selection failed: {e}")
            raise

class DescriptionGenerationAgent(GenerationAgent):
    """Agent for generating video descriptions.
    
    This agent creates SEO-optimized descriptions that encourage
    viewer engagement.
    """
    def __init__(self):
        super().__init__(
            "Description Generation Agent",
            DEFAULT_LONG_TEXT_GENERATION_MODEL
        )

    async def execute(self, input_data: str) -> str:
        """Generate video description.
        
        Args:
            input_data: Selected title to base description on
            
        Returns:
            Generated description
            
        Raises:
            Exception: If description generation fails
        """
        try:
            system_prompt = (
                "You are an AI assistant taking on the role of a prodigy "
                "SEO copywriter and YouTube content creator with over 20 "
                "years of experience."
            )
            
            prompt = f"""As a seasoned SEO copywriter and YouTube content creator with extensive experience in crafting engaging, algorithm-friendly video descriptions, your task is to compose a masterful 1000-character YouTube video description for a faceless YouTube Shorts video titled "{input_data}". This description should:

1. Seamlessly incorporate the keyword "{input_data}" in the first sentence
2. Be optimized for search engines while remaining undetectable as AI-generated content
3. Engage viewers and encourage them to watch the full video
4. Include relevant calls-to-action (e.g., subscribe, like, comment)
5. Utilize natural language and a conversational tone suitable for the target audience
6. Highlight how the video addresses a real-world problem or provides valuable insights to engage viewers

Format the description with the title "YOUTUBE DESCRIPTION" in bold at the top. Ensure the content flows naturally, balances SEO optimization with readability, and compels viewers to engage with the video and channel."""

            return await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.6
            )
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            raise

class HashtagAndTagGenerationAgent(GenerationAgent):
    """Agent for generating hashtags and tags.
    
    This agent creates SEO-optimized hashtags and tags to improve
    video discoverability.
    """
    def __init__(self):
        super().__init__(
            "Hashtag and Tag Generation Agent",
            DEFAULT_TEXT_GENERATION_MODEL
        )

    async def execute(self, input_data: str) -> str:
        """Generate hashtags and tags.
        
        Args:
            input_data: Selected title to base tags on
            
        Returns:
            Generated hashtags and tags
            
        Raises:
            Exception: If tag generation fails
        """
        try:
            system_prompt = (
                "You are an AI assistant taking on the role of a leading "
                "YouTube SEO specialist and social media strategist with "
                "over 10 years of experience in optimizing video discoverability."
            )
            
            prompt = f"""As a leading YouTube SEO specialist and social media strategist with a proven track record in optimizing video discoverability and virality, your task is to create an engaging and relevant set of hashtags and tags for the faceless YouTube Shorts video titled "{input_data}". Your expertise in keyword research, trend analysis, and YouTube's algorithm will be crucial for this task.

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

            return await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.6
            )
            
        except Exception as e:
            logger.error(f"Hashtag and tag generation failed: {e}")
            raise

class VideoScriptGenerationAgent(ProcessingAgent):
    """Agent for generating video scripts.
    
    This agent creates engaging scripts optimized for YouTube Shorts.
    """
    def __init__(self):
        super().__init__(
            "Video Script Generation Agent",
            DEFAULT_LONG_TEXT_GENERATION_MODEL
        )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video script.
        
        Args:
            input_data: Dictionary containing:
                - research: Research summary
                - video_length: Target video length in seconds
                
        Returns:
            Generated script with scene breakdown
            
        Raises:
            Exception: If script generation fails
        """
        try:
            research_result = input_data.get('research', '')
            video_length = input_data.get('video_length', 180)
            
            system_prompt = (
                "You are an AI assistant taking on the role of a leading "
                "YouTube content creator and SEO specialist with a deep "
                "understanding of audience engagement, particularly in "
                "creating faceless YouTube Shorts."
            )
            
            prompt = f"""As a YouTube content creator specializing in faceless YouTube Shorts, craft a detailed, engaging, and enthralling script for a {video_length}-second vertical video based on the following information:

{research_result}

Your script should include:

1. An attention-grabbing opening hook that immediately captivates viewers
2. Key points from the research presented in a concise and engaging manner
3. A strong call-to-action conclusion to encourage viewer interaction (e.g., like, share, subscribe)

Ensure that the script is suitable for a faceless video, relying on voiceover narration and visual storytelling elements.

Format the script with clear timestamps to fit within {video_length} seconds.

Optimize for viewer retention and engagement, keeping in mind the fast-paced nature of YouTube Shorts."""

            script = await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.7
            )
            
            return {
                'script': script,
                'video_length': video_length
            }
            
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            raise 