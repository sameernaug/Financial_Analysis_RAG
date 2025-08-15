

import yfinance as yf
import pandas as pd
import feedparser
import requests
from datetime import datetime, timedelta
import json
import os

class FinancialDataAcquisition:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_market_data(self, symbols, period="1y"):
        """Fetch real market data for given symbols with rate limiting"""
        import time
        market_data = {}
        
        for i, symbol in enumerate(symbols):
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # Add delay between requests to avoid rate limiting
                    if i > 0:
                        time.sleep(2)  # 2 second delay between requests
                    
                    ticker = yf.Ticker(symbol)
                    
                    # Historical price data
                    hist_data = ticker.history(period=period)
                    
                    # Company info
                    info = ticker.info
                    
                    # Financial statements
                    financials = ticker.financials
                    balance_sheet = ticker.balance_sheet
                    cash_flow = ticker.cashflow
                    
                    market_data[symbol] = {
                        'historical_data': hist_data,
                        'company_info': info,
                        'financials': financials,
                        'balance_sheet': balance_sheet,
                        'cash_flow': cash_flow,
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    # Save to file
                    self.save_market_data(symbol, market_data[symbol])
                    print(f"Successfully fetched data for {symbol}")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                        wait_time = 2 ** retry_count  # Exponential backoff
                        print(f"Rate limited for {symbol}. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error fetching data for {symbol}: {e}")
                        break
                        
        return market_data
    
    def fetch_financial_news(self, limit=50):
        """Fetch financial news from public RSS feeds with rate limiting"""
        import time
        news_sources = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.investing.com/rss/news.rss',
            'https://www.marketwatch.com/rss/topstories.rss'
        ]
        
        all_news = []
        
        for i, source in enumerate(news_sources):
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # Add delay between RSS feeds
                    if i > 0:
                        time.sleep(1)
                    
                    feed = feedparser.parse(source)
                    
                    if not feed.entries:
                        print(f"No entries found for {source}")
                        break
                    
                    articles_per_source = max(1, limit // len(news_sources))
                    for entry in feed.entries[:articles_per_source]:
                        news_item = {
                            'title': entry.title,
                            'summary': entry.summary if hasattr(entry, 'summary') else '',
                            'link': entry.link,
                            'published': entry.published if hasattr(entry, 'published') else datetime.now().isoformat(),
                            'source': source
                        }
                        all_news.append(news_item)
                    
                    print(f"Successfully fetched {len(feed.entries[:articles_per_source])} articles from {source}")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    print(f"Error fetching from {source}: {e}. Retry {retry_count}/{max_retries} in {wait_time}s")
                    time.sleep(wait_time)
        
        # Save news data
        if all_news:
            news_df = pd.DataFrame(all_news)
            news_df.to_csv(f"{self.data_dir}/financial_news.csv", index=False)
            print(f"Saved {len(all_news)} news articles")
        
        return all_news
    
    def simulate_sec_filings(self, symbols):
        """Simulate SEC filings using available public data with rate limiting"""
        import time
        filings = []
        
        for i, symbol in enumerate(symbols):
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # Add delay between requests
                    if i > 0:
                        time.sleep(2)
                    
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Create simulated 10-K/10-Q style reports
                    filing = {
                        'symbol': symbol,
                        'company_name': info.get('longName', symbol),
                        'filing_type': '10-K',
                        'filing_date': datetime.now().isoformat(),
                        'revenue': info.get('totalRevenue', 0),
                        'net_income': info.get('netIncomeToCommon', 0),
                        'total_assets': info.get('totalAssets', 0),
                        'total_debt': info.get('totalDebt', 0),
                        'cash': info.get('totalCash', 0),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'content': f"""
                    Annual Report for {info.get('longName', symbol)}
                    
                    Business Overview: {info.get('longBusinessSummary', 'No description available')}
                    
                    Financial Highlights:
                    - Revenue: ${info.get('totalRevenue', 0):,}
                    - Net Income: ${info.get('netIncomeToCommon', 0):,}
                    - Total Assets: ${info.get('totalAssets', 0):,}
                    - Market Cap: ${info.get('marketCap', 0):,}
                    
                    Risk Factors: Standard business risks including market volatility, competition, and regulatory changes.
                    """
                    }
                    
                    filings.append(filing)
                    print(f"Successfully created filing for {symbol}")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                        wait_time = 2 ** retry_count
                        print(f"Rate limited for {symbol} filing. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error creating filing for {symbol}: {e}")
                        break
        
        # Save filings
        if filings:  # Only save if we have data
            filings_df = pd.DataFrame(filings)
            filings_df.to_csv(f"{self.data_dir}/sec_filings.csv", index=False)
            print(f"Saved {len(filings)} SEC filings")
        
        return filings
    
    def save_market_data(self, symbol, data):
        """Save market data to JSON files"""
        filename = f"{self.data_dir}/market_data_{symbol}.json"
        
        # Convert DataFrames to JSON-serializable format
        serializable_data = {}
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                serializable_data[key] = value.to_dict('records')
            elif isinstance(value, pd.Series):
                serializable_data[key] = value.to_dict()
            else:
                serializable_data[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
    
    def run_data_acquisition(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']):
        """Run complete data acquisition pipeline"""
        print("Starting data acquisition...")
        
        # 1. Market Data
        print("Fetching market data...")
        market_data = self.fetch_market_data(symbols)
        
        # 2. Financial News
        print("Fetching financial news...")
        news = self.fetch_financial_news()
        
        # 3. SEC Filings (simulated)
        print("Creating SEC filings...")
        filings = self.simulate_sec_filings(symbols)
        
        print(f"Data acquisition complete!")
        print(f"- Market data for {len(market_data)} symbols")
        print(f"- {len(news)} news articles")
        print(f"- {len(filings)} SEC filings")
        
        return {
            'market_data': market_data,
            'news': news,
            'filings': filings
        }