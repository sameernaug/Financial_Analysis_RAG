"""
Financial Analysis RAG System - Streamlit Application
Provides interactive interface for investment insights and risk assessment
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.rag_pipeline import FinancialRAGPipeline
from scripts.data_processor import FinancialDataProcessor
from scripts.data_acquisition import FinancialDataAcquisition

# Page configuration
st.set_page_config(
    page_title="Financial Analysis RAG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low { color: #28a745; }
    .risk-medium { color: #ffc107; }
    .risk-high { color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

# Initialize pipeline
@st.cache_resource
def initialize_pipeline():
    return FinancialRAGPipeline()

# Main app
def main():
    st.markdown('<h1 class="main-header">📈 Financial Analysis RAG System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Data refresh option
    if st.sidebar.button("🔄 Refresh Data"):
        with st.spinner("Fetching latest financial data..."):
            acquirer = FinancialDataAcquisition()
            acquirer.run_data_acquisition()
            st.success("Data refreshed successfully!")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Investment Insights", "Portfolio Comparison", "Risk Assessment", "Market Trends", "System Overview"]
    )
    
    # Initialize pipeline
    pipeline = initialize_pipeline()
    
    if page == "Investment Insights":
        investment_insights_page(pipeline)
    elif page == "Portfolio Comparison":
        portfolio_comparison_page(pipeline)
    elif page == "Risk Assessment":
        risk_assessment_page(pipeline)
    elif page == "Market Trends":
        market_trends_page(pipeline)
    elif page == "System Overview":
        system_overview_page(pipeline)

def investment_insights_page(pipeline):
    st.header("🔍 Investment Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, GOOGL, MSFT")
        query = st.text_area("Investment Query", placeholder="Ask about investment potential, risks, or trends...")
    
    with col2:
        lookback_days = st.slider("Analysis Period (days)", 7, 365, 30)
        
    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analyzing..."):
            insights = pipeline.generate_investment_insights(symbol, query)
            
            # Display insights
            display_investment_insights(insights)

def display_investment_insights(insights):
    st.subheader(f"Analysis for {insights['symbol']}")
    
    # Risk Assessment
    st.markdown("### 📊 Risk Assessment")
    risk_data = insights.get('risk_assessment', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_level = risk_data.get('risk_level', 'Unknown')
        risk_color = {
            'Low': 'risk-low',
            'Medium': 'risk-medium',
            'High': 'risk-high'
        }.get(risk_level, 'risk-medium')
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Risk Level</h4>
            <h2 class="{risk_color}">{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        volatility = risk_data.get('volatility', 0)
        st.metric("Volatility", f"{volatility:.2%}")
    
    with col3:
        sharpe = risk_data.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col4:
        max_dd = risk_data.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # Trend Analysis
    st.markdown("### 📈 Trend Analysis")
    trends = insights.get('trend_analysis', {})
    
    if trends:
        trend_df = pd.DataFrame(trends).T
        trend_df.index = [f"{idx} Days" for idx in trend_df.index]
        
        st.dataframe(trend_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Recommendation
    st.markdown("### 💡 Recommendation")
    recommendation = insights.get('recommendation', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Action", recommendation.get('action', 'HOLD'))
    with col2:
        st.metric("Confidence", recommendation.get('confidence', 'Medium'))
    
    if recommendation.get('rationale'):
        st.markdown("**Rationale:**")
        for rationale in recommendation['rationale']:
            st.write(f"• {rationale}")
    
    # Context
    st.markdown("### 📰 Relevant Context")
    context = insights.get('context', {})
    
    if context.get('news'):
        st.markdown("**Recent News:**")
        for news_item in context['news'][:3]:
            st.write(f"• {news_item.get('document', '')[:200]}...")

def portfolio_comparison_page(pipeline):
    st.header("📊 Portfolio Comparison")
    
    symbols_input = st.text_input(
        "Enter symbols to compare (comma-separated)", 
        value="AAPL,GOOGL,MSFT,TSLA,AMZN"
    )
    
    if st.button("Compare Symbols"):
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        with st.spinner("Comparing symbols..."):
            comparison = pipeline.compare_symbols(symbols)
            
            # Create comparison dataframe
            comp_data = []
            for symbol, data in comparison['comparison'].items():
                comp_data.append({
                    'Symbol': symbol,
                    'Risk Level': data['risk_level'],
                    'Volatility': data['volatility'],
                    'Sharpe Ratio': data['sharpe_ratio'],
                    '7d Return': data['trends'].get('7d', {}).get('return', 0),
                    '30d Return': data['trends'].get('30d', {}).get('return', 0)
                })
            
            df = pd.DataFrame(comp_data)
            
            # Display comparison table
            st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Symbol', y='Volatility', 
                           title='Volatility Comparison', color='Risk Level')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df, x='Volatility', y='Sharpe Ratio', 
                               text='Symbol', title='Risk-Return Profile')
                st.plotly_chart(fig, use_container_width=True)

def risk_assessment_page(pipeline):
    st.header("⚠️ Risk Assessment")
    
    symbol = st.text_input("Enter Symbol for Risk Analysis", value="AAPL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lookback_days = st.slider("Lookback Period", 7, 252, 30)
    
    with col2:
        confidence_level = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
    
    if st.button("Analyze Risk"):
        with st.spinner("Calculating risk metrics..."):
            risk_metrics = pipeline.calculate_risk_metrics(symbol, lookback_days)
            
            if risk_metrics:
                # Risk metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Annual Volatility", f"{risk_metrics['volatility']:.2%}")
                
                with col2:
                    st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
                
                with col3:
                    st.metric("VaR (95%)", f"{risk_metrics['var_95']:.2%}")
                
                with col4:
                    st.metric("Beta", f"{risk_metrics['beta']:.2f}")
                
                # Risk level indicator
                risk_level = risk_metrics['risk_level']
                color = {
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }.get(risk_level, 'gray')
                
                st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid {color}; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: {color};">Risk Level: {risk_level}</h3>
                </div>
                """, unsafe_allow_html=True)

def market_trends_page(pipeline):
    st.header("📈 Market Trends Analysis")
    
    symbols = st.multiselect(
        "Select symbols for trend analysis",
        ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX'],
        default=['AAPL', 'GOOGL', 'MSFT']
    )
    
    if st.button("Analyze Trends"):
        trends_data = []
        
        for symbol in symbols:
            trends = pipeline.analyze_trends(symbol)
            
            if trends:
                for period, data in trends.items():
                    trends_data.append({
                        'Symbol': symbol,
                        'Period': period,
                        'Return': data['return'],
                        'Volatility': data['volatility'],
                        'Direction': data['trend_direction']
                    })
        
        if trends_data:
            df = pd.DataFrame(trends_data)
            
            # Heatmap
            pivot_df = df.pivot(index='Symbol', columns='Period', values='Return')
            fig = px.imshow(pivot_df, text_auto='.2%', aspect="auto",
                           title="Returns Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))

def system_overview_page(pipeline):
    st.header("🎯 System Overview")
    
    # System statistics
    st.markdown("### System Statistics")
    
    # Vector store stats
    stats = pipeline.vector_store.get_collection_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_docs = sum(stats.get(col, {}).get('count', 0) for col in ['market_data', 'news', 'sec_filings'])
        st.metric("Total Documents", total_docs)
    
    with col2:
        collections = len([col for col in stats.values() if col.get('count', 0) > 0])
        st.metric("Active Collections", collections)
    
    with col3:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
    
    # Data sources
    st.markdown("### Data Sources")
    
    sources = [
        {"Source": "Market Data", "Provider": "Yahoo Finance", "Status": "✅ Active"},
        {"Source": "Financial News", "Provider": "RSS Feeds", "Status": "✅ Active"},
        {"Source": "SEC Filings", "Provider": "Simulated Data", "Status": "✅ Active"}
    ]
    
    st.table(pd.DataFrame(sources))
    
    # Features
    st.markdown("### Key Features")
    
    features = [
        "🔍 Real-time market data retrieval",
        "📊 Comprehensive risk assessment",
        "📈 Trend analysis with multiple timeframes",
        "⚠️ Value at Risk (VaR) calculations",
        "🎯 Investment recommendations",
        "📰 News sentiment analysis",
        "🔮 Portfolio comparison tools",
        "🕒 Temporal context preservation"
    ]
    
    for feature in features:
        st.write(feature)

if __name__ == "__main__":
    main()