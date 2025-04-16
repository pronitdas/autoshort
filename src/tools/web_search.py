"""
Web search tool using RSS feeds for content discovery.
"""
import logging
import re
import aiohttp
from typing import Dict, Any, List
from datetime import datetime

from ..core.base import Tool

logger = logging.getLogger(__name__)

class WebSearchTool(Tool):
    """Tool for searching web content using RSS feeds.
    
    This tool provides a free alternative to paid search APIs by
    aggregating content from various RSS feeds.
    """
    def __init__(self):
        super().__init__("Web Search Tool")
        self.feeds = [
            "http://rss.cnn.com/rss/cnn_topstories.rss",
            "http://feeds.bbci.co.uk/news/rss.xml",
            "https://www.reddit.com/r/news/.rss",
            "https://news.google.com/rss",
            "https://feeds.npr.org/1001/rss.xml"
        ]

    async def use(self, input_data: str, time_period: str = 'all') -> Dict[str, Any]:
        """Search for content across RSS feeds.
        
        Args:
            input_data: Search query
            time_period: Time period to search ('all', 'day', 'week', 'month')
            
        Returns:
            Dictionary containing search results
            
        Raises:
            Exception: If search fails
        """
        try:
            all_items = []
            search_terms = input_data.lower().split()
            
            async with aiohttp.ClientSession() as session:
                for feed_url in self.feeds:
                    try:
                        async with session.get(feed_url) as response:
                            if response.status == 200:
                                content = await response.text()
                                items = await self._parse_feed(content, search_terms)
                                all_items.extend(items)
                    except Exception as e:
                        logger.error(f"Error fetching feed {feed_url}: {e}")
                        continue

            # Filter by time period if specified
            if time_period != 'all':
                all_items = self._filter_by_time(all_items, time_period)

            # Sort by relevance
            all_items.sort(key=lambda x: self._calculate_relevance(x, search_terms), reverse=True)
            
            logger.info(f"Found {len(all_items)} relevant items")
            return {
                "organic_results": all_items[:20]  # Return top 20 results
            }

        except Exception as e:
            logger.error(f"Error in WebSearchTool: {str(e)}")
            return {"organic_results": []}

    async def _parse_feed(self, content: str, search_terms: List[str]) -> List[Dict[str, str]]:
        """Parse RSS feed content and extract relevant items.
        
        Args:
            content: RSS feed content
            search_terms: List of search terms
            
        Returns:
            List of relevant items from the feed
        """
        items = []
        item_matches = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
        
        for item in item_matches:
            title_match = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
            desc_match = re.search(r'<description>(.*?)</description>', item, re.DOTALL)
            link_match = re.search(r'<link>(.*?)</link>', item, re.DOTALL)
            date_match = re.search(r'<pubDate>(.*?)</pubDate>', item, re.DOTALL)
            
            title = title_match.group(1) if title_match else ""
            description = desc_match.group(1) if desc_match else ""
            link = link_match.group(1) if link_match else ""
            pubdate = date_match.group(1) if date_match else ""
            
            # Clean HTML from description
            description = re.sub(r'<.*?>', '', description)
            
            # Check if any search term is in title or description
            if any(term in title.lower() or term in description.lower() for term in search_terms):
                items.append({
                    "title": title,
                    "snippet": description,
                    "link": link,
                    "published_date": pubdate
                })
        
        return items

    def _calculate_relevance(self, item: Dict[str, str], search_terms: List[str]) -> int:
        """Calculate relevance score for an item.
        
        Args:
            item: Item to score
            search_terms: List of search terms
            
        Returns:
            Relevance score
        """
        score = 0
        for term in search_terms:
            if term in item["title"].lower():
                score += 3  # Higher weight for title matches
            if term in item["snippet"].lower():
                score += 1
        return score

    def _filter_by_time(self, items: List[Dict[str, str]], time_period: str) -> List[Dict[str, str]]:
        """Filter items by time period.
        
        Args:
            items: List of items to filter
            time_period: Time period to filter by ('day', 'week', 'month')
            
        Returns:
            Filtered list of items
        """
        now = datetime.now()
        filtered_items = []
        
        for item in items:
            try:
                pub_date = datetime.strptime(item['published_date'], '%a, %d %b %Y %H:%M:%S %z')
                delta = now - pub_date.replace(tzinfo=None)
                
                if (time_period == 'day' and delta.days <= 1) or \
                   (time_period == 'week' and delta.days <= 7) or \
                   (time_period == 'month' and delta.days <= 30):
                    filtered_items.append(item)
            except (ValueError, TypeError):
                # If date parsing fails, include the item
                filtered_items.append(item)
        
        return filtered_items 