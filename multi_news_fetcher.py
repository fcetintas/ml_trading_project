import logging
import requests
import feedparser
import time
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

@dataclass
class NewsArticle:
    """Represents a news article from any source"""
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    sentiment_score: Optional[float] = None
    currencies: List[str] = None
    confidence: float = 1.0  # Source reliability confidence
    
    def __post_init__(self):
        if self.currencies is None:
            self.currencies = []

class MultiNewsAggregator:
    """Aggregates cryptocurrency news from multiple sources"""
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the multi-source news aggregator
        
        Args:
            api_keys: Dictionary with API keys for different services
                     Format: {'newsapi': 'key', 'cryptocompare': 'key', etc.}
        """
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoSentimentBot/2.0'
        })
        
        # Rate limiting per source
        self.rate_limits = {
            'newsapi': {'last_call': 0, 'min_interval': 1.0},
            'cryptocompare': {'last_call': 0, 'min_interval': 0.1},
            'messari': {'last_call': 0, 'min_interval': 0.5},
            'rss': {'last_call': 0, 'min_interval': 0.1},
        }
        
        # Cache for deduplication
        self.article_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # RSS feeds for crypto news
        self.rss_feeds = [
            {'url': 'https://cointelegraph.com/rss', 'source': 'CoinTelegraph'},
            {'url': 'https://coindesk.com/arc/outboundfeeds/rss/', 'source': 'CoinDesk'},
            {'url': 'https://www.coinjournal.net/news/feed/', 'source': 'CoinJournal'},
            {'url': 'https://cryptonews.com/news/feed/', 'source': 'CryptoNews'},
            {'url': 'https://bitcoinist.com/feed/', 'source': 'Bitcoinist'},
            {'url': 'https://www.newsbtc.com/feed/', 'source': 'NewsBTC'},
        ]
        
        # Currency mapping for different sources
        self.currency_mapping = {
            'BTCUSDT': ['BTC', 'Bitcoin', 'bitcoin'],
            'ETHUSDT': ['ETH', 'Ethereum', 'ethereum'],
            'BNBUSDT': ['BNB', 'Binance', 'binance coin'],
            'ARBUSDT': ['ARB', 'Arbitrum', 'arbitrum'],
            'ENAUSDT': ['ENA', 'Ethena', 'ethena'],
            'WLDUSDT': ['WLD', 'Worldcoin', 'worldcoin'],
            'XAIUSDT': ['XAI', 'Xai', 'xai'],
            'APTUSDT': ['APT', 'Aptos', 'aptos'],
            'MINAUSDT': ['MINA', 'Mina Protocol', 'mina'],
        }
        
        self.logger.info("🌐 Multi-source news aggregator initialized")
    
    def _rate_limit(self, source: str):
        """Implement rate limiting for each source"""
        if source not in self.rate_limits:
            return
        
        current_time = time.time()
        last_call = self.rate_limits[source]['last_call']
        min_interval = self.rate_limits[source]['min_interval']
        
        time_since_last = current_time - last_call
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.rate_limits[source]['last_call'] = time.time()
    
    def _make_request(self, url: str, params: Dict = None, headers: Dict = None, 
                     source: str = 'general') -> Optional[Dict]:
        """Make HTTP request with error handling"""
        self._rate_limit(source)
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"{source} request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.warning(f"{source} JSON decode failed: {e}")
            return None
    
    def _extract_currencies_from_text(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from text"""
        currencies = []
        text_lower = text.lower()
        
        for symbol, keywords in self.currency_mapping.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    base_currency = symbol.replace('USDT', '')
                    if base_currency not in currencies:
                        currencies.append(base_currency)
        
        return currencies
    
    def fetch_newsapi_articles(self, currencies: List[str] = None, 
                              limit: int = 50) -> List[NewsArticle]:
        """Fetch news from NewsAPI"""
        if 'newsapi' not in self.api_keys:
            return []
        
        try:
            # Build query for crypto news
            crypto_terms = ['cryptocurrency', 'bitcoin', 'ethereum', 'crypto', 'blockchain']
            if currencies:
                # Add specific currency terms
                for currency in currencies:
                    if currency in ['BTC', 'Bitcoin']:
                        crypto_terms.extend(['bitcoin', 'BTC'])
                    elif currency in ['ETH', 'Ethereum']:
                        crypto_terms.extend(['ethereum', 'ETH'])
                    # Add more mappings as needed
            
            query = ' OR '.join(crypto_terms[:10])  # Limit query length
            
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),
                'apiKey': self.api_keys['newsapi']
            }
            
            response = self._make_request(
                'https://newsapi.org/v2/everything',
                params=params,
                source='newsapi'
            )
            
            if not response or 'articles' not in response:
                return []
            
            articles = []
            for article_data in response['articles']:
                try:
                    # Parse date
                    published_str = article_data.get('publishedAt', '')
                    published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                    
                    # Extract content
                    title = article_data.get('title', '')
                    description = article_data.get('description', '')
                    content = f"{title}. {description}".strip()
                    
                    # Extract currencies mentioned
                    currencies_found = self._extract_currencies_from_text(content)
                    
                    article = NewsArticle(
                        title=title,
                        content=content,
                        url=article_data.get('url', ''),
                        published_at=published_at,
                        source=f"NewsAPI ({article_data.get('source', {}).get('name', 'Unknown')})",
                        currencies=currencies_found,
                        confidence=0.9  # High confidence for NewsAPI
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing NewsAPI article: {e}")
                    continue
            
            self.logger.info(f"📰 NewsAPI: Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            self.logger.error(f"NewsAPI fetch failed: {e}")
            return []
    
    def fetch_cryptocompare_articles(self, currencies: List[str] = None,
                                   limit: int = 50) -> List[NewsArticle]:
        """Fetch news from CryptoCompare"""
        if 'cryptocompare' not in self.api_keys:
            return []
        
        try:
            params = {
                'lang': 'EN',
                'api_key': self.api_keys['cryptocompare']
            }
            
            # Add currency filter if specified
            if currencies:
                params['categories'] = ','.join(currencies[:5])  # Limit categories
            
            response = self._make_request(
                'https://min-api.cryptocompare.com/data/v2/news/',
                params=params,
                source='cryptocompare'
            )
            
            if not response or 'Data' not in response:
                return []
            
            articles = []
            for article_data in response['Data'][:limit]:
                try:
                    # Parse date (Unix timestamp)
                    published_timestamp = article_data.get('published_on', 0)
                    published_at = datetime.fromtimestamp(published_timestamp)
                    
                    # Extract content
                    title = article_data.get('title', '')
                    body = article_data.get('body', '')
                    content = f"{title}. {body}".strip()
                    
                    # Extract currencies
                    currencies_found = self._extract_currencies_from_text(content)
                    
                    article = NewsArticle(
                        title=title,
                        content=content,
                        url=article_data.get('url', ''),
                        published_at=published_at,
                        source=f"CryptoCompare ({article_data.get('source', 'Unknown')})",
                        currencies=currencies_found,
                        confidence=0.85
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing CryptoCompare article: {e}")
                    continue
            
            self.logger.info(f"📰 CryptoCompare: Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            self.logger.error(f"CryptoCompare fetch failed: {e}")
            return []
    
    def fetch_coingecko_articles(self, limit: int = 10) -> List[NewsArticle]:
        """CoinGecko doesn't have a news API - returns empty list"""
        self.logger.info("📰 CoinGecko: No news API available (market data only)")
        return []
    
    def fetch_rss_articles(self, limit: int = 20) -> List[NewsArticle]:
        """Fetch articles from RSS feeds"""
        articles = []
        
        for feed_info in self.rss_feeds[:3]:  # Limit to 3 feeds to avoid overload
            try:
                self._rate_limit('rss')
                feed = feedparser.parse(feed_info['url'])
                
                for entry in feed.entries[:limit//3]:  # Distribute across feeds
                    try:
                        # Parse date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_at = datetime(*entry.published_parsed[:6])
                        else:
                            published_at = datetime.now()
                        
                        title = entry.get('title', '')
                        summary = entry.get('summary', '')
                        content = f"{title}. {summary}".strip()
                        
                        currencies_found = self._extract_currencies_from_text(content)
                        
                        article = NewsArticle(
                            title=title,
                            content=content,
                            url=entry.get('link', ''),
                            published_at=published_at,
                            source=feed_info['source'],
                            currencies=currencies_found,
                            confidence=0.7  # Lower confidence for RSS
                        )
                        articles.append(article)
                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                self.logger.warning(f"RSS feed {feed_info['source']} failed: {e}")
                continue
        
        self.logger.info(f"📰 RSS Feeds: Found {len(articles)} articles")
        return articles
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on title
            title_lower = article.title.lower()
            title_words = set(re.findall(r'\w+', title_lower))
            
            # Check for similarity with existing titles
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(re.findall(r'\w+', seen_title))
                # If 70% of words match, consider it a duplicate
                if len(title_words & seen_words) / max(len(title_words), 1) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(title_lower)
        
        removed_count = len(articles) - len(unique_articles)
        if removed_count > 0:
            self.logger.info(f"🔍 Removed {removed_count} duplicate articles")
        
        return unique_articles
    
    def fetch_aggregated_news(self, currencies: Optional[List[str]] = None,
                            limit: int = 50, hours_back: int = 24) -> List[NewsArticle]:
        """
        Fetch news from all sources and aggregate them
        
        Args:
            currencies: List of currency symbols to filter by
            limit: Maximum total articles to return
            hours_back: How many hours back to fetch news
            
        Returns:
            Aggregated and deduplicated list of articles
        """
        all_articles = []
        
        # Use ThreadPoolExecutor to fetch from multiple sources in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Submit fetch tasks
            if 'newsapi' in self.api_keys:
                futures.append(executor.submit(self.fetch_newsapi_articles, currencies, limit//4))
            
            if 'cryptocompare' in self.api_keys:
                futures.append(executor.submit(self.fetch_cryptocompare_articles, currencies, limit//4))
            
            # CoinGecko doesn't have news API, skip
            # futures.append(executor.submit(self.fetch_coingecko_articles, limit//4))
            futures.append(executor.submit(self.fetch_rss_articles, limit//4))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    articles = future.result(timeout=30)
                    all_articles.extend(articles)
                except Exception as e:
                    self.logger.warning(f"Source fetch failed: {e}")
        
        # Filter by currency if specified
        if currencies:
            filtered_articles = []
            for article in all_articles:
                # Check if article mentions any of the target currencies
                article_currencies = set(article.currencies)
                target_currencies = set(currencies)
                if article_currencies & target_currencies:  # Intersection
                    filtered_articles.append(article)
            all_articles = filtered_articles
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        all_articles = [a for a in all_articles if a.published_at >= cutoff_time]
        
        # Deduplicate
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Sort by confidence and recency
        unique_articles.sort(key=lambda x: (x.confidence, x.published_at), reverse=True)
        
        # Limit results
        result_articles = unique_articles[:limit]
        
        self.logger.info(f"🌐 Aggregated {len(result_articles)} unique articles from all sources")
        return result_articles
    
    def get_news_summary(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Get summary statistics of fetched news"""
        if not articles:
            return {
                'total_articles': 0,
                'sources': [],
                'currencies': [],
                'time_range': None,
                'avg_confidence': 0.0
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
        
        # Calculate average confidence
        avg_confidence = sum(article.confidence for article in articles) / len(articles)
        
        return {
            'total_articles': len(articles),
            'sources': sources,
            'currencies': currencies,
            'time_range': time_range,
            'avg_confidence': avg_confidence
        }