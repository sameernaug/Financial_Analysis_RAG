

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from typing import List, Dict, Any
import os

class FinancialDataProcessor:
    def __init__(self, data_dir="data"):
        # Look for data in both the local directory and one level up
        if os.path.exists(os.path.join(data_dir, "market_data_AAPL.json")):
            self.data_dir = data_dir
        elif os.path.exists(os.path.join("..", data_dir, "market_data_AAPL.json")):
            self.data_dir = os.path.join("..", data_dir)
        else:
            # Absolute fallback path if needed
            self.data_dir = "c:\\Users\\0603s\\Downloads\\new 2\\data"
        
        self.processed_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_market_data(self, symbol):
        """
        Load market data from JSON files
        
        Safely handles file existence and JSON parsing errors.
        Returns an empty dict if the file is missing or corrupted.
        
        Args:
            symbol (str): The stock symbol (e.g., 'AAPL')
            
        Returns:
            dict: Market data as a dictionary or empty dict if data unavailable
        """
        # Default return value if anything fails
        default_return = {}
        
        try:
            # Construct the filename
            filename = os.path.join(self.data_dir, f"market_data_{symbol}.json")
            
            # Check if file exists
            if not os.path.exists(filename):
                print(f"Warning: Market data file for {symbol} not found at {filename}")
                return default_return
                
            # Check if file is readable
            if not os.access(filename, os.R_OK):
                print(f"Warning: No read permission for {filename}")
                return default_return
                
            # Check if file is empty
            if os.path.getsize(filename) == 0:
                print(f"Warning: Market data file for {symbol} is empty")
                return default_return
            
            # Attempt to read and parse the JSON file
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Verify the data is a dictionary
                if not isinstance(data, dict):
                    print(f"Warning: Market data for {symbol} is not a valid dictionary")
                    return default_return
                
                return data
                
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse market data for {symbol}: {str(e)}")
        except PermissionError:
            print(f"Error: Permission denied when reading {symbol} market data")
        except Exception as e:
            print(f"Unexpected error loading market data for {symbol}: {str(e)}")
            
        return default_return
    
    def load_news_data(self):
        """Load financial news from CSV"""
        filename = os.path.join(self.data_dir, "financial_news.csv")
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return pd.DataFrame()
    
    def load_sec_filings(self):
        """Load SEC filings from CSV"""
        filename = os.path.join(self.data_dir, "sec_filings.csv")
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return pd.DataFrame()
    
    def normalize_financial_data(self, data):
        """Normalize financial data for consistency"""
        normalized = {}
        
        if isinstance(data, dict):
            # Handle market data
            if 'historical_data' in data:
                hist_df = pd.DataFrame(data['historical_data'])
                hist_df['Date'] = pd.to_datetime(hist_df['Date'])
                hist_df = hist_df.sort_values('Date')
                
                # Add technical indicators
                hist_df['SMA_20'] = hist_df['Close'].rolling(window=20).mean()
                hist_df['SMA_50'] = hist_df['Close'].rolling(window=50).mean()
                hist_df['Volatility'] = hist_df['Close'].pct_change().rolling(window=20).std()
                
                normalized['price_data'] = hist_df
            
            # Handle company info
            if 'company_info' in data:
                info = data['company_info']
                normalized['company_info'] = {
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'description': info.get('longBusinessSummary', '')
                }
        
        return normalized
    
    def create_temporal_chunks(self, data, chunk_size=30, overlap=7):
        """Create temporal chunks with overlap for time-series analysis"""
        if isinstance(data, pd.DataFrame) and 'Date' in data.columns:
            data = data.sort_values('Date')
            chunks = []
            
            start_idx = 0
            while start_idx < len(data):
                end_idx = min(start_idx + chunk_size, len(data))
                chunk = data.iloc[start_idx:end_idx]
                
                chunk_info = {
                    'start_date': chunk['Date'].iloc[0].isoformat(),
                    'end_date': chunk['Date'].iloc[-1].isoformat(),
                    'data': chunk.to_dict('records'),
                    'metadata': {
                        'days': len(chunk),
                        'avg_price': chunk['Close'].mean(),
                        'volatility': chunk['Close'].pct_change().std(),
                        'trend': 'up' if chunk['Close'].iloc[-1] > chunk['Close'].iloc[0] else 'down'
                    }
                }
                
                chunks.append(chunk_info)
                start_idx += (chunk_size - overlap)
            
            return chunks
        
        return []
    
    def chunk_news_by_sentiment_and_topic(self, news_df, max_chunk_size=500):
        """Chunk news articles by sentiment and topic"""
        chunks = []
        
        for _, article in news_df.iterrows():
            # Basic sentiment analysis (simple keyword-based)
            positive_words = ['growth', 'profit', 'gain', 'rise', 'surge', 'bullish', 'positive']
            negative_words = ['loss', 'decline', 'drop', 'fall', 'bearish', 'negative', 'risk']
            
            text = f"{article['title']} {article['summary']}"
            text_lower = text.lower()
            
            sentiment_score = 0
            sentiment_score += sum(1 for word in positive_words if word in text_lower)
            sentiment_score -= sum(1 for word in negative_words if word in text_lower)
            
            sentiment = 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
            
            # Create chunks
            words = text.split()
            for i in range(0, len(words), max_chunk_size):
                chunk_words = words[i:i+max_chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunk = {
                    'text': chunk_text,
                    'source': article.get('source', ''),
                    'published': article.get('published', ''),
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score,
                    'length': len(chunk_text)
                }
                
                chunks.append(chunk)
        
        return chunks
    
    def chunk_sec_filings(self, filings_df):
        """Chunk SEC filings into meaningful sections"""
        chunks = []
        
        for _, filing in filings_df.iterrows():
            # Split content into sections
            content = str(filing.get('content', ''))
            
            # Create structured chunks
            sections = [
                'business_overview',
                'financial_highlights',
                'risk_factors',
                'management_discussion',
                'financial_statements'
            ]
            
            for section in sections:
                chunk = {
                    'symbol': filing.get('symbol', ''),
                    'company_name': filing.get('company_name', ''),
                    'filing_type': filing.get('filing_type', ''),
                    'filing_date': filing.get('filing_date', ''),
                    'section': section,
                    'content': content[:1000],  # Truncate for chunking
                    'metadata': {
                        'revenue': filing.get('revenue', 0),
                        'net_income': filing.get('net_income', 0),
                        'market_cap': filing.get('market_cap', 0),
                        'pe_ratio': filing.get('pe_ratio', 0)
                    }
                }
                
                chunks.append(chunk)
        
        return chunks
    
    def create_composite_chunks(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']):
        """Create composite chunks combining all data sources"""
        all_chunks = []
        
        for symbol in symbols:
            # Market data chunks
            market_data = self.load_market_data(symbol)
            if market_data:
                normalized = self.normalize_financial_data(market_data)
                if 'price_data' in normalized:
                    price_chunks = self.create_temporal_chunks(normalized['price_data'])
                    
                    for chunk in price_chunks:
                        chunk['type'] = 'market_data'
                        chunk['symbol'] = symbol
                        all_chunks.append(chunk)
        
        # News chunks
        news_df = self.load_news_data()
        if not news_df.empty:
            news_chunks = self.chunk_news_by_sentiment_and_topic(news_df)
            for chunk in news_chunks:
                chunk['type'] = 'news'
                all_chunks.append(chunk)
        
        # SEC filing chunks
        filings_df = self.load_sec_filings()
        if not filings_df.empty:
            filing_chunks = self.chunk_sec_filings(filings_df)
            for chunk in filing_chunks:
                chunk['type'] = 'sec_filing'
                all_chunks.append(chunk)
        
        # Save processed chunks
        chunks_df = pd.DataFrame(all_chunks)
        chunks_df.to_json(f"{self.processed_dir}/all_chunks.json", orient='records', indent=2)
        
        return all_chunks
    
    def get_chunk_statistics(self):
        """Get statistics about processed chunks"""
        chunks_file = f"{self.processed_dir}/all_chunks.json"
        
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            
            stats = {
                'total_chunks': len(chunks),
                'chunk_types': {},
                'date_range': {'min': None, 'max': None},
                'avg_chunk_size': 0
            }
            
            for chunk in chunks:
                chunk_type = chunk.get('type', 'unknown')
                stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
                
                # Calculate date range
                dates = [chunk.get('start_date'), chunk.get('end_date'), chunk.get('published'), chunk.get('filing_date')]
                dates = [d for d in dates if d]
                
                for date_str in dates:
                    try:
                        date = pd.to_datetime(date_str)
                        if stats['date_range']['min'] is None or date < pd.to_datetime(stats['date_range']['min']):
                            stats['date_range']['min'] = date.isoformat()
                        if stats['date_range']['max'] is None or date > pd.to_datetime(stats['date_range']['max']):
                            stats['date_range']['max'] = date.isoformat()
                    except:
                        pass
            
            # Calculate average chunk size
            sizes = [len(str(chunk.get('content', ''))) for chunk in chunks]
            stats['avg_chunk_size'] = sum(sizes) / len(sizes) if sizes else 0
            
            return stats
        
        return {'error': 'No chunks found'}