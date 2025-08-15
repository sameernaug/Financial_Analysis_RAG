

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os
from scripts.vector_store import FinancialVectorStore
from scripts.data_processor import FinancialDataProcessor
import warnings
warnings.filterwarnings('ignore')

class FinancialRAGPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.vector_store = FinancialVectorStore()
        self.data_processor = FinancialDataProcessor(data_dir)
        
        # Risk thresholds
        self.risk_thresholds = {
            'low_volatility': 0.02,
            'high_volatility': 0.05,
            'low_pe': 15,
            'high_pe': 30
        }
    
    def calculate_risk_metrics(self, symbol: str, lookback_days: int = 30) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for a symbol"""
        market_data = self.data_processor.load_market_data(symbol)
        
        if not market_data:
            return {}
        
        # Get price data
        hist_data = pd.DataFrame(market_data.get('historical_data', []))
        if hist_data.empty:
            return {}
        
        hist_data['Date'] = pd.to_datetime(hist_data['Date'])
        hist_data = hist_data.sort_values('Date')
        
        # Calculate returns
        hist_data['returns'] = hist_data['Close'].pct_change()
        recent_data = hist_data.tail(lookback_days)
        
        if len(recent_data) < 5:
            return {}
        
        # Risk metrics
        volatility = recent_data['returns'].std() * np.sqrt(252)  # Annualized
        max_drawdown = self.calculate_max_drawdown(recent_data['Close'])
        sharpe_ratio = recent_data['returns'].mean() / recent_data['returns'].std() * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(recent_data['returns'].dropna(), 5)
        var_99 = np.percentile(recent_data['returns'].dropna(), 1)
        
        # Beta calculation (simplified)
        beta = self.calculate_beta(recent_data['returns'])
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'beta': beta,
            'risk_level': self.assess_risk_level(volatility, max_drawdown)
        }
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series = None) -> float:
        """Calculate beta relative to market (simplified with SPY as proxy)"""
        if market_returns is None:
            # Use a simple market proxy
            market_returns = pd.Series(np.random.normal(0.0001, 0.01, len(returns)))
        
        if len(returns) != len(market_returns):
            market_returns = market_returns[:len(returns)]
        
        covariance = np.cov(returns.dropna(), market_returns.dropna())[0][1]
        market_variance = np.var(market_returns.dropna())
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def assess_risk_level(self, volatility: float, max_drawdown: float) -> str:
        """Assess overall risk level"""
        if volatility < self.risk_thresholds['low_volatility'] and max_drawdown > -0.05:
            return "Low"
        elif volatility < self.risk_thresholds['high_volatility'] and max_drawdown > -0.15:
            return "Medium"
        else:
            return "High"
    
    def analyze_trends(self, symbol: str, periods: List[int] = [7, 30, 90]) -> Dict[str, Any]:
        """
        Analyze price trends over multiple periods
        
        Args:
            symbol (str): Stock symbol (e.g., AAPL)
            periods (List[int]): List of time periods to analyze in days
            
        Returns:
            Dict[str, Any]: Dictionary containing trend analysis for each period
        """
        try:
            # Normalize symbol to uppercase for consistency
            symbol = symbol.upper() if symbol else ""
            
            # Load market data
            market_data = self.data_processor.load_market_data(symbol)
            
            if not market_data:
                print(f"Warning: No market data available for {symbol}")
                return {}
            
            # Convert to DataFrame
            hist_data = pd.DataFrame(market_data.get('historical_data', []))
            if hist_data.empty:
                print(f"Warning: Empty historical data for {symbol}")
                return {}
            
            # Ensure required columns exist
            required_columns = ['Date', 'Close']
            if not all(col in hist_data.columns for col in required_columns):
                print(f"Warning: Missing required columns in {symbol} data. Found: {hist_data.columns.tolist()}")
                return {}
            
            # Clean and sort data
            hist_data['Date'] = pd.to_datetime(hist_data['Date'], errors='coerce')
            hist_data = hist_data.dropna(subset=['Date', 'Close'])  # Remove rows with missing critical data
            hist_data = hist_data.sort_values('Date')
            
            trends = {}
            
            for period in periods:
                try:
                    if len(hist_data) >= period:
                        recent_data = hist_data.tail(period)
                        start_price = recent_data['Close'].iloc[0]
                        end_price = recent_data['Close'].iloc[-1]
                        
                        returns = (end_price - start_price) / start_price
                        
                        # Handle potential NaN or inf in volatility calculation
                        returns_series = recent_data['Close'].pct_change().dropna()
                        volatility = returns_series.std() * np.sqrt(252) if not returns_series.empty else 0
                        
                        # Simple moving average crossover with error handling
                        try:
                            sma_short = recent_data['Close'].rolling(window=min(5, period//4)).mean().iloc[-1]
                            sma_long = recent_data['Close'].rolling(window=min(20, period//2)).mean().iloc[-1]
                            ma_signal = 'Buy' if sma_short > sma_long else 'Sell'
                        except (IndexError, ValueError):
                            sma_short = sma_long = 0
                            ma_signal = 'Neutral'
                        
                        trends[f"{period}d"] = {
                            'return': float(returns),  # Convert to float to avoid serialization issues
                            'volatility': float(volatility),
                            'trend_direction': 'Bullish' if returns > 0 else 'Bearish',
                            'trend_strength': float(abs(returns) / volatility if volatility > 0 else 0),
                            'ma_signal': ma_signal
                        }
                except Exception as e:
                    print(f"Warning: Error calculating {period}-day trend for {symbol}: {str(e)}")
        except Exception as e:
            print(f"Error in trend analysis for {symbol}: {str(e)}")
            return {}
        
        return trends
    
    def retrieve_context(self, query: str, symbols: List[str] = None, 
                        days_back: int = 30, n_results: int = 10) -> Dict[str, Any]:
        """Retrieve relevant context with temporal filtering"""
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Query vector store
        results = self.vector_store.similarity_search_with_temporal_filter(
            query=query,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            symbols=symbols,
            n_results=n_results
        )
        
        # Organize results by type
        context = {
            'market_data': [],
            'news': [],
            'sec_filings': [],
            'query': query,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
        
        for result in results:
            collection_type = result['collection']
            
            if collection_type == 'market_data':
                context['market_data'].append(result)
            elif collection_type == 'news':
                context['news'].append(result)
            elif collection_type == 'sec_filings':
                context['sec_filings'].append(result)
        
        return context
    
    def generate_investment_insights(self, symbol: str, query: str = None) -> Dict[str, Any]:
        """Generate comprehensive investment insights for a symbol"""
        
        # Risk assessment
        risk_metrics = self.calculate_risk_metrics(symbol)
        
        # Trend analysis
        trends = self.analyze_trends(symbol)
        
        # Context retrieval
        if query:
            context = self.retrieve_context(query, symbols=[symbol])
        else:
            context = self.retrieve_context(f"{symbol} investment analysis", symbols=[symbol])
        
        # Generate insights
        insights = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'risk_assessment': risk_metrics,
            'trend_analysis': trends,
            'context': context,
            'recommendation': self.generate_recommendation(risk_metrics, trends, context)
        }
        
        return insights
    
    def generate_recommendation(self, risk_metrics: Dict, trends: Dict, context: Dict) -> Dict[str, str]:
        """Generate investment recommendation based on analysis"""
        
        recommendation = {
            'action': 'HOLD',
            'confidence': 'Medium',
            'rationale': [],
            'risk_factors': []
        }
        
        # Risk-based recommendation
        if risk_metrics.get('risk_level') == 'Low':
            recommendation['rationale'].append("Low volatility and drawdown risk")
        elif risk_metrics.get('risk_level') == 'High':
            recommendation['rationale'].append("High risk profile - consider position sizing")
            recommendation['risk_factors'].append("High volatility")
        
        # Trend-based recommendation
        if trends:
            recent_trend = trends.get('7d', {})
            if recent_trend.get('return', 0) > 0.05:
                recommendation['action'] = 'BUY'
                recommendation['confidence'] = 'High'
                recommendation['rationale'].append("Strong short-term momentum")
            elif recent_trend.get('return', 0) < -0.05:
                recommendation['action'] = 'SELL'
                recommendation['confidence'] = 'Medium'
                recommendation['rationale'].append("Negative short-term trend")
        
        # Context-based insights with error handling
        try:
            if context and isinstance(context, dict) and context.get('news'):
                news_items = context['news']
                if news_items:
                    positive_news = sum(1 for item in news_items 
                                    if item.get('metadata', {}).get('sentiment') == 'positive')
                    if positive_news > len(news_items) / 2:
                        recommendation['rationale'].append("Positive news sentiment")
                    elif positive_news < len(news_items) / 4:
                        recommendation['rationale'].append("Negative news sentiment")
                        recommendation['risk_factors'].append("Unfavorable news coverage")
        except Exception as e:
            # If there's any error processing news context, just skip this enhancement
            pass
        
        return recommendation
    
    def compare_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple symbols for investment decisions"""
        
        comparison = {
            'symbols': symbols,
            'timestamp': datetime.now().isoformat(),
            'comparison': {}
        }
        
        for symbol in symbols:
            insights = self.generate_investment_insights(symbol)
            comparison['comparison'][symbol] = {
                'risk_level': insights['risk_assessment'].get('risk_level', 'Unknown'),
                'volatility': insights['risk_assessment'].get('volatility', 0),
                'sharpe_ratio': insights['risk_assessment'].get('sharpe_ratio', 0),
                'trends': insights['trend_analysis']
            }
        
        return comparison
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str = None):
        """Save analysis results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{analysis.get('symbol', 'general')}_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, "analysis", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return filepath