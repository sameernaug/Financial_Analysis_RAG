# Financial Analysis RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that combines financial reports, market data, and news to provide intelligent investment insights and risk assessments with temporal context and trend analysis.

## 🎯 Project Overview

This system integrates multi-source financial data to deliver:
- Real-time market data analysis
- Risk assessment and volatility modeling
- Investment insights with temporal context
- Portfolio comparison tools
- Comprehensive trend analysis

## 🚀 Live Demo

Access the deployed application: [Financial Analysis RAG App on Streamlit Cloud](https://financial-analysis-rag.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-analysis-rag.streamlit.app/)

## 📊 Key Features

### Data Integration
- **Market Data**: Historical prices, company information, financial statements via Yahoo Finance
- **Financial News**: Real-time news feeds from RSS sources
- **SEC Filings**: Simulated regulatory filings for analysis

### Analysis Capabilities
- **Risk Assessment**: Volatility, VaR, Sharpe ratio, max drawdown calculations
- **Trend Analysis**: Multi-timeframe trend identification (7d, 30d, 90d, 365d)
- **Portfolio Comparison**: Side-by-side symbol analysis
- **Investment Insights**: Context-aware recommendations with confidence scores

### Technical Architecture
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB for efficient retrieval
- **Frontend**: Streamlit interactive interface
- **Processing**: Pandas for data manipulation, Statsmodels for statistical analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- No API keys required - uses free public data sources

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/financial-rag-system.git
   cd financial-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data acquisition**
   ```bash
   python scripts/data_acquisition.py
   ```

4. **Process and chunk data**
   ```bash
   python scripts/data_processor.py
   ```

5. **Initialize vector store**
   ```bash
   python scripts/vector_store.py
   ```

6. **Launch the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
financial-rag-system/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Data storage directory
│   ├── market_data/        # Historical market data
│   ├── news/               # Financial news articles
│   └── sec_filings/        # SEC filing documents
├── scripts/                 # Core system components
│   ├── data_acquisition.py # Data fetching from multiple sources
│   ├── data_processor.py   # Data normalization and chunking
│   ├── vector_store.py     # ChromaDB initialization and management
│   ├── rag_pipeline.py     # Main RAG pipeline logic
│   └── test_environment.py # Environment verification
├── models/                  # Local model storage
└── notebooks/              # Analysis notebooks (optional)
```

## 🔧 Technical Details

### Data Sources
- **Yahoo Finance (yfinance)**: Real-time market data
- **RSS Feeds**: Financial news from reputable sources
- **Simulated SEC Filings**: Based on public company information

### Embedding Strategy
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Chunking**: Temporal-based chunking for time-series data
- **Vector Storage**: ChromaDB collections for different data types

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Value at Risk (VaR)**: 95% confidence level
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Peak-to-trough decline
- **Beta**: Market correlation coefficient

## 📈 Usage Examples

### Single Stock Analysis
1. Enter a stock symbol (e.g., "AAPL")
2. Ask specific questions about investment potential
3. View comprehensive risk assessment and recommendations

### Portfolio Comparison
1. Select multiple symbols for comparison
2. Analyze risk-return profiles
3. Identify diversification opportunities

### Market Trends
1. Choose symbols for trend analysis
2. View heatmaps across different timeframes
3. Identify momentum patterns

## 🎨 User Interface

The Streamlit application provides:
- **Interactive Dashboard**: Clean, intuitive interface
- **Real-time Updates**: Automatic data refresh capabilities
- **Visual Analytics**: Charts and graphs for data visualization
- **Responsive Design**: Works on desktop and mobile devices

## 📊 Sample Outputs

### Investment Insight Example
```json
{
  "symbol": "AAPL",
  "risk_assessment": {
    "risk_level": "Medium",
    "volatility": 0.25,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.15
  },
  "recommendation": {
    "action": "BUY",
    "confidence": "High",
    "rationale": ["Strong fundamentals", "Positive trend", "Low risk"]
  }
}
```

## 🔍 Evaluation Metrics

### Retrieval Performance
- **Relevance Score**: Context-aware matching
- **Temporal Accuracy**: Time-series correlation
- **Coverage**: Multi-source integration completeness

### System Metrics
- **Response Time**: < 2 seconds for standard queries
- **Data Freshness**: Real-time market data updates
- **Accuracy**: Risk calculations validated against industry standards

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
1. **Streamlit Cloud**: Direct deployment from GitHub
2. **Heroku**: Container-based deployment
3. **AWS**: EC2 or Lambda deployment

### Environment Variables
No API keys required - system uses only public data sources

## 🧪 Testing

### Environment Verification
```bash
python scripts/test_environment.py
```

### Data Validation
- Verify data sources accessibility
- Check vector store initialization
- Validate risk calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review the example usage

## 🎯 Future Enhancements

- **Advanced Models**: Integration with larger language models
- **Real-time Streaming**: Live market data feeds
- **Extended Coverage**: Additional asset classes (bonds, commodities)
- **Machine Learning**: Predictive analytics integration
- **Mobile App**: Native mobile application

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app" 
5. Select this repository, branch (main/master), and set the main file path to `app.py`
6. Click "Deploy!"

### Alternative: Run the Simplified Dashboard

For a simpler version with direct Yahoo Finance data:

```bash
streamlit run scripts/m.py
```

The simplified dashboard offers:
- Real-time stock data retrieval
- Basic risk assessment
- Price trend visualization
- Returns distribution analysis

### Local Setup

```bash
# Clone the repository
git clone https://github.com/sameernaug/Financial_Analysis_RAG.git

# Navigate to project directory
cd Financial_Analysis_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🏆 Achievement

This system demonstrates:
- **End-to-End RAG Implementation**: From data acquisition to user interface
- **Real-world Application**: Practical financial analysis tools
- **Technical Excellence**: Best practices in data engineering and ML
- **User Experience**: Intuitive, professional-grade interface
- **No Dependencies**: Works without external API keys or paid services

---

**Built for**: Financial analysis and investment decision-making
**Technology Stack**: Python, Streamlit, ChromaDB, Sentence Transformers, Matplotlib, yfinance
**Data Sources**: Yahoo Finance, RSS feeds, public financial data
**Deployment**: Streamlit-ready for immediate use