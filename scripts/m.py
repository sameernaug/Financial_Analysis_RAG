import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


class DataProcessor:
    def load_market_data(self, symbol, period="6mo", interval="1d"):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                st.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            df.reset_index(inplace=True)
            return df

        except Exception as e:
            st.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()


class RagPipeline:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def calculate_risk_metrics(self, symbol):
        market_data = self.data_processor.load_market_data(symbol, period="1y", interval="1d")
        if market_data.empty:
            return {"error": f"No market data for {symbol}"}

        market_data["Return"] = market_data["Close"].pct_change()
        volatility = market_data["Return"].std() * (252 ** 0.5)  # Annualized
        avg_return = market_data["Return"].mean() * 252

        return {
            "volatility": volatility,
            "average_return": avg_return,
            "data": market_data
        }

    def generate_investment_insights(self, symbol):
        metrics = self.calculate_risk_metrics(symbol)
        if "error" in metrics:
            return metrics["error"], None

        risk_level = "High" if metrics['volatility'] > 0.3 else "Moderate" if metrics['volatility'] > 0.15 else "Low"

        insights = (
            f"📊 **Investment Insights for {symbol}**\n"
            f"- **Annualized Volatility:** {metrics['volatility']:.2%}\n"
            f"- **Annualized Average Return:** {metrics['average_return']:.2%}\n"
            f"- **Risk Assessment:** {risk_level}"
        )
        return insights, metrics["data"]


# --- Streamlit Frontend ---
st.set_page_config(page_title="Stock Risk & Trend Overview", layout="wide")
st.title("📈 Stock Risk & Trend Dashboard")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")

if st.button("Generate Insights"):
    data_processor = DataProcessor()
    pipeline = RagPipeline(data_processor=data_processor)

    insights, df = pipeline.generate_investment_insights(symbol)

    if df is not None:
        st.markdown(insights)

        # Price chart
        st.subheader("Price Trend (1 Year)")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Close"], label="Close Price", color="blue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # Returns distribution
        st.subheader("Daily Returns Distribution")
        fig2, ax2 = plt.subplots()
        df["Return"].dropna().hist(ax=ax2, bins=50, color="orange")
        ax2.set_xlabel("Daily Return")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
    else:
        st.error(insights)
