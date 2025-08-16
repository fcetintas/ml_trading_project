import logging
import requests
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class NewsArticle:
    """Represents a news article"""
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    sentiment_score: Optional[float] = None
    currencies: List[str] = None
    
    def __post_init__(self):
        if self.currencies is None:
            self.currencies = []

class NewsFetcher:
    """Fetches cryptocurrency news from various sources"""
    
    def __init__(self, api_key: str):
        """
        Initialize the news fetcher
        
        Args:
            api_key: CryptoPanic API key
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/developer/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoSentimentBot/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # Cache for storing recent news to avoid duplicates
        self.news_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a request to the CryptoPanic API
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            JSON response or None if failed
        """
        self._rate_limit()
        
        # Add API key to params
        params['auth_token'] = self.api_key
        
        try:
            url = f"{self.base_url}/{endpoint}"
            self.logger.debug(f"Making request to: {url}")
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON response: {e}")
            return None
    
    def _parse_cryptopanic_article(self, article_data: Dict) -> NewsArticle:
        """
        Parse a CryptoPanic article response
        
        Args:
            article_data: Raw article data from API
            
        Returns:
            NewsArticle object
        """
        try:
            # Extract basic information
            title = article_data.get('title', '')
            url = article_data.get('url', '')
            source = article_data.get('source', {}).get('title', 'Unknown')
            
            # Parse published date
            published_str = article_data.get('published_at', '')
            try:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                from datetime import timezone
                published_at = datetime.now(timezone.utc)
            
            # Extract currencies
            currencies = []
            if 'currencies' in article_data:
                for currency in article_data['currencies']:
                    if isinstance(currency, dict):
                        currencies.append(currency.get('code', ''))
                    else:
                        currencies.append(str(currency))
            
            # Use title as content if no separate content available
            content = title
            
            return NewsArticle(
                title=title,
                content=content,
                url=url,
                published_at=published_at,
                source=source,
                currencies=currencies
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing article: {e}")
            raise
    
    def fetch_recent_news(self, 
                         currencies: Optional[List[str]] = None,
                         limit: int = 50,
                         hours_back: int = 24) -> List[NewsArticle]:
        """
        Fetch recent news articles
        
        Args:
            currencies: List of currency codes to filter by (e.g., ['BTC', 'ETH'])
            limit: Maximum number of articles to fetch
            hours_back: How many hours back to fetch news
            
        Returns:
            List of NewsArticle objects
        """
        try:
            # Prepare parameters
            params = {
                'kind': 'news',  # Only news articles
                'page': 1,
                'per_page': min(limit, 100)  # API limit
            }
            
            # Add currency filter if specified
            if currencies:
                # Convert symbols to match CryptoPanic format
                currency_mapping = {
                    'BTCUSDT': 'BTC',    # Bitcoin
                    'ETHUSDT': 'ETH',    # Ethereum
                    'BNBUSDT': 'BNB',    # Binance Coin  
                    'ADAUSDT': 'ADA',    # Cardano
                    'SOLUSDT': 'SOL',    # Solana
                    'XRPUSDT': 'XRP',    # XRP
                    'DOTUSDT': 'DOT',    # Polkadot
                    'DOGEUSDT': 'DOGE',  # Dogecoin
                    'AVAXUSDT': 'AVAX',  # Avalanche
                    'LINKUSDT': 'LINK',  # Chainlink
                    'MATICUSDT': 'MATIC',# Polygon
                    'LTCUSDT': 'LTC',    # Litecoin
                    'UNIUSDT': 'UNI',    # Uniswap
                    'ATOMUSDT': 'ATOM',  # Cosmos
                    'VETUSDT': 'VET',    # VeChain
                }
                
                filtered_currencies = []
                for currency in currencies:
                    mapped_currency = currency_mapping.get(currency, currency)
                    # Remove 'USDT' suffix if present
                    if mapped_currency.endswith('USDT'):
                        mapped_currency = mapped_currency[:-4]
                    filtered_currencies.append(mapped_currency)
                
                params['currencies'] = ','.join(filtered_currencies)
            
            # Make the request
            response = self._make_request('posts/', params)
            
            if not response or 'results' not in response:
                self.logger.warning("No news data received")
                return []
            
            self.logger.debug(f"API returned {len(response.get('results', []))} total articles")
            if currencies:
                self.logger.debug(f"Filtering for currencies: {currencies}")
            
            articles = []
            # Use timezone-aware datetime for comparison
            from datetime import timezone
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            for article_data in response['results']:
                try:
                    article = self._parse_cryptopanic_article(article_data)
                    
                    # Filter by time
                    if article.published_at < cutoff_time:
                        continue
                    
                    # Check cache to avoid duplicates
                    cache_key = f"{article.url}_{article.published_at.isoformat()}"
                    if cache_key not in self.news_cache:
                        self.news_cache[cache_key] = time.time()
                        articles.append(article)
                    
                except Exception as e:
                    self.logger.error(f"Error processing article: {e}")
                    continue
            
            # Clean old cache entries
            self._clean_cache()
            
            if articles:
                self.logger.info(f"📰 Found {len(articles)} news articles")
            else:
                self.logger.info("📰 No new articles found")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
    
    def fetch_currency_specific_news(self, currency: str, limit: int = 20) -> List[NewsArticle]:
        """
        Fetch news specific to a currency
        
        Args:
            currency: Currency symbol (e.g., 'BTC', 'ETH')
            limit: Maximum number of articles
            
        Returns:
            List of NewsArticle objects
        """
        # Remove USDT suffix if present
        if currency.endswith('USDT'):
            currency = currency[:-4]
        
        return self.fetch_recent_news(currencies=[currency], limit=limit)
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.news_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.news_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def get_market_sentiment_keywords(self) -> List[str]:
        """Get keywords that indicate market sentiment"""
        return [
            # Positive sentiment keywords
            'bullish', 'rally', 'surge', 'pump', 'moon', 'breakout', 'adoption',
            'partnership', 'upgrade', 'milestone', 'record', 'high', 'gains',
            'positive', 'optimistic', 'confident', 'growth', 'rise', 'increase',
            
            # Negative sentiment keywords  
            'bearish', 'crash', 'dump', 'decline', 'drop', 'fall', 'correction',
            'selloff', 'liquidation', 'panic', 'fear', 'uncertainty', 'regulation',
            'ban', 'hack', 'vulnerability', 'loss', 'negative', 'pessimistic'
        ]
    
    def filter_by_keywords(self, articles: List[NewsArticle], keywords: List[str]) -> List[NewsArticle]:
        """
        Filter articles that contain specific keywords
        
        Args:
            articles: List of articles to filter
            keywords: List of keywords to search for
            
        Returns:
            Filtered list of articles
        """
        filtered_articles = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for article in articles:
            title_lower = article.title.lower()
            content_lower = article.content.lower()
            
            if any(keyword in title_lower or keyword in content_lower for keyword in keywords_lower):
                filtered_articles.append(article)
        
        return filtered_articles
    
    def get_news_summary(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """
        Get a summary of fetched news
        
        Args:
            articles: List of articles to summarize
            
        Returns:
            Summary dictionary
        """
        if not articles:
            return {
                'total_articles': 0,
                'sources': [],
                'currencies': [],
                'time_range': None
            }
        
        # Get unique sources
        sources = list(set(article.source for article in articles))
        
        # Get unique currencies
        all_currencies = []
        for article in articles:
            all_currencies.extend(article.currencies)
        currencies = list(set(all_currencies))
        
        # Get time range
        publish_times = [article.published_at for article in articles]
        time_range = {
            'earliest': min(publish_times),
            'latest': max(publish_times)
        }
        
        return {
            'total_articles': len(articles),
            'sources': sources,
            'currencies': currencies,
            'time_range': time_range
        }