import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import percentileofscore
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Advanced IT Companies Financial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS for better UI styling"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #0D47A1;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .card {
            border-radius: 5px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
            border-left: 5px solid #1E88E5;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .positive-value {
            color: #4CAF50;
            font-weight: bold;
        }
        .negative-value {
            color: #F44336;
            font-weight: bold;
        }
        .neutral-value {
            color: #757575;
            font-weight: bold;
        }
        .company-name {
            font-size: 1.3rem;
            font-weight: bold;
            color: #0D47A1;
        }
        .company-code {
            font-size: 1rem;
            color: #546E7A;
        }
        .ranking-reason {
            font-style: italic;
            color: #546E7A;
            margin-top: 0.5rem;
        }
        .dashboard-metric {
            background-color: #f0f7ff;
            border-radius: 5px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            height: 100%;
        }
        .dashboard-metric-title {
            font-size: 1rem;
            color: #555;
            margin-bottom: 0.5rem;
        }
        .dashboard-metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f2ff;
            border-bottom: 2px solid #1E88E5;
        }
        .sentiment-positive {
            color: #4CAF50;
            font-weight: bold;
        }
        .sentiment-negative {
            color: #F44336;
            font-weight: bold;
        }
        .sentiment-neutral {
            color: #757575;
            font-weight: bold;
        }
        .macro-indicator {
            padding: 1rem;
            border-radius: 5px;
            background-color: #f5f5f5;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .strategy-panel {
            background-color: #f8f9fa;
            padding: i0.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            border-left: 4px solid #0D47A1;
        }
        .small-text {
            font-size: 0.8rem;
            color: #757575;
        }
        div[data-testid="stSidebarNav"] {
            padding-top: 2rem;
        }
        div[data-testid="stSidebar"] [data-testid="stMarkdown"] h1 {
            font-size: 1.5rem;
        }
        div[data-testid="stSidebar"] button[kind="primary"] {
            width: 100%;
        }
        .compare-table th {
            background-color: #f0f7ff;
            font-weight: 600;
        }
        .compare-table td {
            text-align: center;
        }
        .trend-up {
            color: #4CAF50;
            font-weight: bold;
        }
        .trend-down {
            color: #F44336;
            font-weight: bold;
        }
        .pdf-download-btn {
            background-color: #1E88E5;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            text-decoration: none;
            display: inline-block;
            font-weight: bold;
            text-align: center;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    """Initialize session state variables for the application"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'ranked_df' not in st.session_state:
        st.session_state.ranked_df = None
    
    if 'saved_analyses' not in st.session_state:
        st.session_state.saved_analyses = {}
    
    if 'selected_companies' not in st.session_state:
        st.session_state.selected_companies = []
    
    if 'comparison_metrics' not in st.session_state:
        st.session_state.comparison_metrics = [
            'Return on equity', 'Return on capital employed', 
            'Free cash flow last year', 'Price to Earning', 
            'Debt to equity', 'Profit growth 5Years',
            'Buffett Score', 'Lynch Score', 'Graham Score'
        ]
    
    if 'custom_strategy_params' not in st.session_state:
        st.session_state.custom_strategy_params = {
            'Return on equity_threshold': 15,
            'Return on equity_direction': True,
            'Return on equity_weight': 0.2,
            'Price to Earning_threshold': 20,
            'Price to Earning_direction': False,
            'Price to Earning_weight': 0.2,
            'Debt to equity_threshold': 0.5,
            'Debt to equity_direction': False,
            'Debt to equity_weight': 0.15,
            'Profit growth 5Years_threshold': 10,
            'Profit growth 5Years_direction': True,
            'Profit growth 5Years_weight': 0.15,
            'Free cash flow last year_threshold': 0,
            'Free cash flow last year_direction': True,
            'Free cash flow last year_weight': 0.15,
            'Dividend yield_threshold': 1.5,
            'Dividend yield_direction': True,
            'Dividend yield_weight': 0.15
        }
    
    if 'api_connections' not in st.session_state:
        st.session_state.api_connections = {
            'financial_api_key': '',
            'news_api_key': '',
            'macro_api_key': ''
        }
    
    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = None
    
    if 'macro_indicators' not in st.session_state:
        st.session_state.macro_indicators = None
    
    if 'sentiment_data' not in st.session_state:
        st.session_state.sentiment_data = None

# Define the main functions for the financial model
def normalize_series(series, higher_is_better=True):
    """Normalize a series to 0-1 scale"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    if higher_is_better:
        return (series - min_val) / (max_val - min_val)
    else:
        return (max_val - series) / (max_val - min_val)

def calculate_buffett_score(df):
    """Calculate Warren Buffett investment score"""
    # Consistent ROE over time
    df['Consistent ROE'] = df['Return on equity'] > 15
    
    # Debt to Equity ratio < 0.5 (conservative)
    df['Low Debt'] = df['Debt to equity'] < 0.5
    
    # High profit margins (using OPM - Operating Profit Margin)
    df['High Margins'] = df['OPM'] > 20
    
    # Consistent earnings growth
    df['Earnings Growth'] = df['Profit growth 5Years'] > 10
    
    # Convert boolean columns to float for calculation
    buffett_components = {
        'Consistent ROE': 0.25,
        'Low Debt': 0.25,
        'High Margins': 0.25,
        'Earnings Growth': 0.25
    }
    
    for col in buffett_components.keys():
        df[col] = df[col].astype(float)
    
    df['Buffett Score'] = sum(df[col] * weight for col, weight in buffett_components.items()) * 100
    return df

def calculate_lynch_score(df):
    """Calculate Peter Lynch investment score"""
    # PEG Ratio < 1 is excellent, < 2 is good
    df['Good PEG'] = df['PEG Ratio'] < 2
    
    # Companies with high growth relative to P/E
    df['Growth to PE'] = df['Sales growth 5Years'] / df['Price to Earning']
    
    # Normalize Growth to PE
    df['Growth to PE'] = normalize_series(df['Growth to PE'])
    df['Growth to PE'] = df['Growth to PE'].fillna(0)
    
    # Lynch score components
    lynch_components = {
        'Good PEG': 0.3,
        'Growth to PE': 0.3,
        'ROE Score': 0.2,
        'Sales Growth Score': 0.2
    }
    
    # Convert boolean columns to float for calculation
    df['Good PEG'] = df['Good PEG'].astype(float)
    
    df['Lynch Score'] = sum(df[col] * weight for col, weight in lynch_components.items()) * 100
    return df

def calculate_graham_score(df):
    """Calculate Benjamin Graham's value investing score"""
    # 1. Adequate Size (Market Cap > 1000 cr)
    df['Graham_Size'] = df['Market Capitalization'] > 1000
    
    # 2. Strong Financial Condition (Current Ratio > 2)
    # Add this if you have current ratio data
    if 'Current ratio' in df.columns:
        df['Graham_Financial'] = df['Current ratio'] > 2
    else:
        df['Graham_Financial'] = True  # Default assumption
    
    # 3. Earnings Stability (Positive earnings for past 5 years)
    df['Graham_Earnings'] = df['Profit growth 5Years'] > 0
    
    # 4. Dividend Record (Uninterrupted dividend payments)
    df['Graham_Dividend'] = df['Dividend yield'] > 0
    
    # 5. Earnings Growth (5-year growth > 5%)
    df['Graham_Growth'] = df['Profit growth 5Years'] > 5
    
    # 6. Moderate P/E Ratio (< 15)
    df['Graham_PE'] = df['Price to Earning'] < 15
    
    # 7. Moderate Price to Book (P/B < 1.5)
    if 'Price to Book' in df.columns:
        df['Graham_PB'] = df['Price to Book'] < 1.5
    else:
        df['Graham_PB'] = True  # Default assumption
    
    # Calculate final Graham score (out of 100)
    graham_components = {
        'Graham_Size': 0.15,
        'Graham_Financial': 0.15,
        'Graham_Earnings': 0.15,
        'Graham_Dividend': 0.15,
        'Graham_Growth': 0.15,
        'Graham_PE': 0.15,
        'Graham_PB': 0.10
    }
    
    # Convert boolean columns to float
    for col in graham_components.keys():
        df[col] = df[col].astype(float)
    
    df['Graham Score'] = sum(df[col] * weight for col, weight in graham_components.items()) * 100
    
    return df

def calculate_dalio_score(df):
    """Calculate Ray Dalio's economic principles score"""
    # 1. Strong Return on Assets (ROA)
    if 'Return on assets' not in df.columns:
        df['Return on assets'] = df['Return on equity'] * (1 - df['Debt to equity']/(1 + df['Debt to equity']))
    
    df['Dalio_ROA'] = df['Return on assets'] > 10
    
    # 2. Low Debt/EBITDA (< 3)
    if 'Debt to EBITDA' in df.columns:
        df['Dalio_Debt'] = df['Debt to EBITDA'] < 3
    else:
        df['Dalio_Debt'] = df['Debt to equity'] < 0.5  # Approximation
    
    # 3. Consistent Cash Flow Growth
    df['Dalio_CashFlow'] = df['Free cash flow last year'] > 0
    
    # 4. Low Earnings Volatility (look for stable earnings)
    if 'Earnings volatility' in df.columns:
        df['Dalio_Stability'] = df['Earnings volatility'] < 0.25
    else:
        df['Dalio_Stability'] = True  # Default assumption
    
    # 5. Diversified Revenue Streams (not dependent on one product/service)
    # This is harder to approximate without product mix data
    df['Dalio_Diversification'] = 0.5  # Neutral score as default
    
    # Calculate Dalio score (out of 100)
    dalio_components = {
        'Dalio_ROA': 0.25,
        'Dalio_Debt': 0.25,
        'Dalio_CashFlow': 0.20,
        'Dalio_Stability': 0.15,
        'Dalio_Diversification': 0.15
    }
    
    # Convert boolean columns to float for calculation
    for col in dalio_components.keys():
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)
    
    df['Dalio Score'] = sum(df[col] * weight for col, weight in dalio_components.items()) * 100
    
    return df

def build_custom_strategy(df, parameters):
    """Build a custom investment strategy with user-defined parameters"""
    # Initialize component scores
    for key in parameters:
        if key.endswith('_threshold'):
            metric = key.split('_threshold')[0]
            threshold = parameters[key]
            higher_is_better = parameters.get(f"{metric}_direction", True)
            
            if higher_is_better:
                df[f'Custom_{metric}'] = df[metric] > threshold
            else:
                df[f'Custom_{metric}'] = df[metric] < threshold
    
    # Get weights for each component
    weights = {}
    for key in parameters:
        if key.endswith('_weight'):
            metric = key.split('_weight')[0]
            weights[f'Custom_{metric}'] = parameters[key]
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    # Convert boolean columns to float
    for col in weights.keys():
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(float)
    
    # Calculate custom score
    df['Custom Strategy Score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns) * 100
    
    return df

def calculate_financial_metrics(df):
    """Calculate comprehensive financial metrics"""
    # Create normalized scores for key metrics
    df['ROE Score'] = normalize_series(df['Return on equity'])
    df['ROCE Score'] = normalize_series(df['Return on capital employed'])
    df['FCF Score'] = normalize_series(df['Free cash flow last year'])
    df['PE Score'] = normalize_series(df['Price to Earning'], higher_is_better=False)
    df['Debt Score'] = normalize_series(df['Debt to equity'], higher_is_better=False)
    df['Profit Growth Score'] = normalize_series(df['Profit growth 5Years'])
    df['Sales Growth Score'] = normalize_series(df['Sales growth 5Years'])
    df['Dividend Score'] = normalize_series(df['Dividend yield'])
    
    # Handle NaN values in scores
    score_columns = ['ROE Score', 'ROCE Score', 'FCF Score', 'PE Score', 'Debt Score', 
                    'Profit Growth Score', 'Sales Growth Score', 'Dividend Score']
    df[score_columns] = df[score_columns].fillna(0)
    
    # Calculate investment philosophy scores
    df = calculate_buffett_score(df)
    df = calculate_lynch_score(df)
    df = calculate_graham_score(df)
    df = calculate_dalio_score(df)
    
    # Build custom strategy if parameters are set
    if 'custom_strategy_params' in st.session_state:
        df = build_custom_strategy(df, st.session_state.custom_strategy_params)
    
    return df

def rank_companies(df, weights=None):
    """Rank companies based on weighted metrics"""
    if weights is None:
        # Default weights based on user's preferences
        weights = {
            'ROE Score': 0.20,
            'ROCE Score': 0.20,
            'FCF Score': 0.15,
            'PE Score': 0.15,
            'Debt Score': 0.10,
            'Profit Growth Score': 0.10,
            'Sales Growth Score': 0.05,
            'Dividend Score': 0.05
        }
    
    # Calculate the weighted ranking score
    df['Ranking Score'] = sum(df[col] * weight for col, weight in weights.items())
    
    # Add investment philosophy scores to the ranking
    df['Ranking Score'] += (df['Buffett Score'] / 100) * 0.15  # 15% weight to Buffett score
    df['Ranking Score'] += (df['Lynch Score'] / 100) * 0.15    # 15% weight to Lynch score
    
    # Rank the companies based on the final score
    df['Rank'] = df['Ranking Score'].rank(ascending=False, method='min').astype(int)
    
    # Sort the dataframe by rank
    return df.sort_values('Rank')

def generate_ranking_reason(row, df):
    """Generate a reason for the company's ranking"""
    reasons = []
    
    # Check ROE and ROCE
    if row['Return on equity'] > 15:
        reasons.append(f"Strong ROE of {row['Return on equity']:.2f}%")
    if row['Return on capital employed'] > 15:
        reasons.append(f"Excellent ROCE of {row['Return on capital employed']:.2f}%")
    
    # Check free cash flow
    if row['Free cash flow last year'] > 0:
        reasons.append(f"Positive free cash flow of ₹{row['Free cash flow last year']:.2f} crores")
    
    # Check PE ratio
    if row['Price to Earning'] < df['Price to Earning'].median():
        reasons.append(f"Attractive P/E ratio of {row['Price to Earning']:.2f}")
    
    # Check growth metrics
    if row['Sales growth 5Years'] > 10:
        reasons.append(f"Strong 5-year sales growth of {row['Sales growth 5Years']:.2f}%")
    if row['Profit growth 5Years'] > 10:
        reasons.append(f"Impressive 5-year profit growth of {row['Profit growth 5Years']:.2f}%")
    
    # Check debt levels
    if row['Debt to equity'] < 0.5:
        reasons.append(f"Low debt-to-equity ratio of {row['Debt to equity']:.2f}")
    
    # Check investment philosophy alignment
    if row['Buffett Score'] > 75:
        reasons.append("Aligns well with Warren Buffett's investment principles")
    if row['Lynch Score'] > 50:
        reasons.append("Matches Peter Lynch's growth at reasonable price criteria")
    if row['Graham Score'] > 60:
        reasons.append("Follows Benjamin Graham's value investing principles")
    if row['Dalio Score'] > 60:
        reasons.append("Consistent with Ray Dalio's economic principles")
    
    # Combine reasons into a paragraph
    if reasons:
        return ". ".join(reasons) + "."
    else:
        return "Good overall financial performance across multiple metrics."

def get_buffett_companies(df):
    """Get companies that match Warren Buffett's investment criteria"""
    return df[df['Buffett Score'] > 75][['Name', 'NSE Code', 'Buffett Score', 'Rank']].sort_values('Buffett Score', ascending=False)

def get_lynch_companies(df):
    """Get companies that match Peter Lynch's investment criteria"""
    return df[df['Lynch Score'] > 50][['Name', 'NSE Code', 'Lynch Score', 'Rank']].sort_values('Lynch Score', ascending=False)

def get_graham_companies(df):
    """Get companies that match Benjamin Graham's investment criteria"""
    return df[df['Graham Score'] > 60][['Name', 'NSE Code', 'Graham Score', 'Rank']].sort_values('Graham Score', ascending=False)

def get_dalio_companies(df):
    """Get companies that match Ray Dalio's investment criteria"""
    return df[df['Dalio Score'] > 60][['Name', 'NSE Code', 'Dalio Score', 'Rank']].sort_values('Dalio Score', ascending=False)

# Time-series analysis function
def analyze_time_series(company_df, time_period='quarterly'):
    """Perform time-series analysis on company financial data"""
    try:
        # For demo purposes, generate synthetic time-series data
        # In a real application, this would use actual historical data
        from datetime import datetime, timedelta
        import random
        
        # Current metrics as starting point
        roe_current = company_df['Return on equity'].iloc[0]
        roce_current = company_df['Return on capital employed'].iloc[0]
        fcf_current = company_df['Free cash flow last year'].iloc[0]
        pe_current = company_df['Price to Earning'].iloc[0]
        
        # Generate dates for the last 8 quarters or years
        end_date = datetime.now()
        
        if time_period == 'quarterly':
            periods = 8
            date_list = [(end_date - timedelta(days=i*90)) for i in range(periods)]
        else:  # yearly
            periods = 5
            date_list = [(end_date - timedelta(days=i*365)) for i in range(periods)]
        
        date_list.reverse()  # Oldest to newest
        
        # Generate time series data with some randomness but a general trend
        # We'll use random walk with drift
        data = []
        roe = roe_current * 0.8  # Start at 80% of current value
        roce = roce_current * 0.8
        fcf = fcf_current * 0.7
        pe = pe_current * 1.1
        
        for date in date_list:
            # Add some random fluctuation with a positive drift
            roe = max(0, roe * (1 + random.uniform(-0.05, 0.15)))
            roce = max(0, roce * (1 + random.uniform(-0.05, 0.15)))
            fcf = fcf * (1 + random.uniform(-0.1, 0.2))
            pe = max(5, pe * (1 + random.uniform(-0.08, 0.08)))
            
            data.append({
                'Date': date,
                'Return on equity': roe,
                'Return on capital employed': roce,
                'Free cash flow': fcf,
                'Price to Earning': pe
            })
        
        # Create DataFrame
        ts_df = pd.DataFrame(data)
        
        # Calculate moving averages
        ts_df['ROE_MA'] = ts_df['Return on equity'].rolling(window=min(3, len(ts_df))).mean()
        ts_df['ROCE_MA'] = ts_df['Return on capital employed'].rolling(window=min(3, len(ts_df))).mean()
        
        # Calculate growth rates
        ts_df['ROE_Growth'] = ts_df['Return on equity'].pct_change() * 100
        ts_df['ROCE_Growth'] = ts_df['Return on capital employed'].pct_change() * 100
        
        # Detect trends using exponential weighted moving average
        ts_df['ROE_Trend'] = ts_df['Return on equity'].ewm(span=min(3, len(ts_df))).mean()
        ts_df['ROCE_Trend'] = ts_df['Return on capital employed'].ewm(span=min(3, len(ts_df))).mean()
        
        return ts_df
    
    except Exception as e:
        st.error(f"Error in time series analysis: {str(e)}")
        return pd.DataFrame()

# Peer comparison function
def perform_peer_comparison(company_data, all_companies_data, metrics):
    """Compare a company against its peers and industry averages"""
    try:
        # Get company sector/industry
        company_sector = company_data.get('Sector', 'IT')
        
        # Filter peer companies in the same sector
        # If sector information is not available, just use all companies
        peer_companies = all_companies_data
        
        # Calculate industry averages
        industry_avg = peer_companies[metrics].mean()
        
        # Prepare comparison dataframe
        comparison = pd.DataFrame({
            'Metric': metrics,
            'Company': [company_data[metric] for metric in metrics],
            'Industry Average': [industry_avg[metric] for metric in metrics]
        })
        
        # Calculate percentiles
        for i, metric in enumerate(metrics):
            percentile = percentileofscore(peer_companies[metric].dropna(), company_data[metric])
            comparison.loc[i, 'Percentile'] = percentile
            
            # Calculate how many standard deviations from the mean
            std = peer_companies[metric].std()
            if std != 0:
                comparison.loc[i, 'Z-Score'] = (company_data[metric] - industry_avg[metric]) / std
            else:
                comparison.loc[i, 'Z-Score'] = 0
        
        return comparison
    
    except Exception as e:
        st.error(f"Error in peer comparison: {str(e)}")
        return pd.DataFrame({'Metric': metrics, 'Error': str(e)})

# Sentiment analysis
def generate_demo_sentiment_data(company_name):
    """Generate demo sentiment data for a company"""
    # For demo purposes, create synthetic news sentiment data
    import random
    from datetime import datetime, timedelta
    
    # Generate random sentiment data for the last 20 days
    news_items = []
    titles = [
        f"{company_name} Announces Quarterly Results",
        f"Analysts Upgrade {company_name} Stock",
        f"New Product Launch from {company_name}",
        f"{company_name} Expands Operations",
        f"Industry Outlook Positive for {company_name}",
        f"{company_name} CEO Discusses Future Strategy",
        f"Market Share Increases for {company_name}",
        f"Regulatory Approval for {company_name} Product",
        f"{company_name} Partners with Tech Giant",
        f"Investor Conference: {company_name} Presents Growth Plans"
    ]
    
    sources = ["Economic Times", "Bloomberg", "Reuters", "CNBC", "Financial Express", 
              "MoneyControl", "LiveMint", "Business Standard", "Financial Times", "WSJ"]
    
    for i in range(20):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        title = random.choice(titles)
        source = random.choice(sources)
        
        # Generate random sentiment values with a slight positive bias
        polarity = random.uniform(-0.5, 0.7)
        subjectivity = random.uniform(0.3, 0.8)
        
        sentiment = "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"
        
        news_items.append({
            'Date': date,
            'Title': title,
            'Source': source,
            'Sentiment': sentiment,
            'Polarity': polarity,
            'Subjectivity': subjectivity,
            'URL': f"https://example.com/news/{i}"
        })
    
    # Create DataFrame
    sentiment_df = pd.DataFrame(news_items)
    
    # Calculate overall sentiment metrics
    overall_sentiment = sentiment_df['Polarity'].mean()
    sentiment_count = sentiment_df['Sentiment'].value_counts()
    
    return {
        'sentiment_df': sentiment_df,
        'overall_sentiment': overall_sentiment,
        'sentiment_count': sentiment_count
    }

    

def fetch_and_analyze_company_sentiment(company_name, company_code):
    """Fetch news about a company and analyze sentiment"""
    # In a real application, this would call a news API
    # For demo purposes, we'll use the demo data generator
    return generate_demo_sentiment_data(company_name)

# Macroeconomic indicator functions
def generate_demo_macro_data():
    """Generate demo macroeconomic indicators"""
    import random
    from datetime import datetime, timedelta
    
    # Current date
    current_date = datetime.now()
    
    # Generate historical data for the past 12 periods
    dates = [(current_date - timedelta(days=30*i)).strftime('%Y-%m-%d') for i in range(12)]
    dates.reverse()  # Oldest to newest
    
    # Generate indicators with realistic values and trends
    indicators = {
        'GDP Growth': {
            'value': round(random.uniform(3.5, 6.5), 2),
            'date': dates[-1],
            'history': [round(random.uniform(3.0, 7.0), 2) for _ in dates],
            'dates': dates,
            'description': 'Annual GDP growth rate in percentage',
            'trend': 'Positive' if random.random() > 0.3 else 'Negative'
        },
        'Inflation': {
            'value': round(random.uniform(3.0, 5.5), 2),
            'date': dates[-1],
            'history': [round(random.uniform(2.5, 6.0), 2) for _ in dates],
            'dates': dates,
            'description': 'Consumer Price Index (CPI) annual change',
            'trend': 'Negative' if random.random() > 0.7 else 'Positive'
        },
        'Interest Rate': {
            'value': round(random.uniform(4.0, 6.0), 2),
            'date': dates[-1],
            'history': [round(random.uniform(3.5, 6.5), 2) for _ in dates],
            'dates': dates,
            'description': 'Central bank policy interest rate',
            'trend': 'Stable' if random.random() > 0.5 else ('Positive' if random.random() > 0.5 else 'Negative')
        },
        'Unemployment': {
            'value': round(random.uniform(3.5, 7.0), 2),
            'date': dates[-1],
            'history': [round(random.uniform(3.0, 8.0), 2) for _ in dates],
            'dates': dates,
            'description': 'Unemployment rate as percentage of workforce',
            'trend': 'Negative' if random.random() > 0.4 else 'Positive'
        },
        'Market Index': {
            'value': round(random.uniform(17500, 19500)),
            'date': dates[-1],
            'history': [round(random.uniform(16000, 20000)) for _ in dates],
            'dates': dates,
            'description': 'Stock market index value',
            'trend': 'Positive' if random.random() > 0.3 else 'Negative'
        }
    }
    
    # Add impact on IT sector
    for key in indicators:
        if key == 'GDP Growth' or key == 'Market Index':
            indicators[key]['sector_impact'] = 'Positive'
        elif key == 'Unemployment':
            indicators[key]['sector_impact'] = 'Mixed'
        else:
            indicators[key]['sector_impact'] = 'Negative' if indicators[key]['value'] > 5 else 'Neutral'
    
    return indicators

def fetch_macroeconomic_indicators():
    """Fetch macroeconomic indicators from public APIs"""
    # In a real application, this would call economic data APIs
    # For demo purposes, we'll use the demo data generator
    return generate_demo_macro_data()

# Visualization functions
def create_interactive_charts(ranked_df, top_n=15):
    """Create interactive charts using Plotly"""
    figures = {}
    
    if ranked_df is None or len(ranked_df) == 0:
        return {}
    
    # Limit to top N companies
    top_companies = ranked_df.head(top_n)
    
    # Ranking Score Bar Chart
    fig_ranking = px.bar(
        top_companies, 
        x='Name', 
        y='Ranking Score',
        color='Ranking Score',
        hover_data=['NSE Code', 'Rank', 'ROE Score', 'ROCE Score', 'FCF Score'],
        title=f'Top {top_n} Companies by Ranking Score',
        color_continuous_scale='Viridis'
    )
    fig_ranking.update_layout(
        xaxis_title='Company',
        yaxis_title='Ranking Score',
        xaxis_tickangle=-45,
        height=600
    )
    figures['ranking'] = fig_ranking
    
    # ROE vs ROCE Scatter Plot
    fig_roe_roce = px.scatter(
        top_companies,
        x='Return on equity',
        y='Return on capital employed',
        size='Market Capitalization',
        color='Rank',
        hover_name='Name',
        size_max=60,
        color_continuous_scale='Viridis_r',
        title='ROE vs ROCE Analysis'
    )
    fig_roe_roce.update_layout(
        xaxis_title='Return on Equity (%)',
        yaxis_title='Return on Capital Employed (%)',
        height=600
    )
    figures['roe_roce'] = fig_roe_roce
    
    # Buffett vs Lynch Score
    fig_philosophy = px.scatter(
        top_companies,
        x='Buffett Score',
        y='Lynch Score',
        size='Market Capitalization',
        color='Ranking Score',
        hover_name='Name',
        hover_data=['NSE Code', 'Rank'],
        size_max=60,
        color_continuous_scale='Viridis',
        title='Investment Philosophy Alignment'
    )
    fig_philosophy.update_layout(
        xaxis_title='Buffett Score',
        yaxis_title='Lynch Score',
        height=600
    )
    figures['philosophy'] = fig_philosophy
    
    # Radar Chart for Top 5 Companies
    top5 = ranked_df.head(5)
    categories = ['ROE Score', 'ROCE Score', 'FCF Score', 'PE Score', 
                 'Debt Score', 'Profit Growth Score', 'Sales Growth Score']
    
    fig_radar = go.Figure()
    for i, (idx, row) in enumerate(top5.iterrows()):
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=row['Name']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Top 5 Companies - Key Metrics Comparison',
        height=600
    )
    figures['radar'] = fig_radar
    
    return figures

def create_correlation_matrix(df):
    """Create an interactive correlation matrix for financial metrics"""
    if df is None or len(df) == 0:
        return None
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out some columns that might not be meaningful for correlation
    exclude_cols = ['Rank', 'Market Capitalization']
    metrics_cols = [col for col in numeric_cols if col not in exclude_cols 
                    and not col.endswith('Score')][:12]  # Limit to top metrics
    
    # Calculate correlation
    corr_matrix = df[metrics_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix of Key Financial Metrics',
        zmin=-1, zmax=1
    )
    
    fig.update_layout(
        height=700,
        width=800
    )
    
    return fig

def create_metrics_heatmap(df, top_n=15):
    """Create a heatmap to compare metrics across multiple companies"""
    if df is None or len(df) == 0:
        return None
    
    # Select top companies
    top_companies = df.head(top_n)
    
    # Select key metrics for comparison
    key_metrics = ['Return on equity', 'Return on capital employed', 
                  'Free cash flow last year', 'Price to Earning', 
                  'Debt to equity', 'OPM', 'Profit growth 5Years', 
                  'Sales growth 5Years', 'Dividend yield']
    
    # Create a normalized version of the metrics for better visualization
    normalized_data = pd.DataFrame()
    normalized_data['Name'] = top_companies['Name']
    
    for metric in key_metrics:
        if metric in ['Price to Earning', 'Debt to equity']:
            # For these metrics, lower is better
            normalized_data[metric] = 1 - ((top_companies[metric] - top_companies[metric].min()) / 
                                         (top_companies[metric].max() - top_companies[metric].min()))
        else:
            # For other metrics, higher is better
            normalized_data[metric] = (top_companies[metric] - top_companies[metric].min()) / \
                                    (top_companies[metric].max() - top_companies[metric].min())
    
    # Melt the dataframe for heatmap format
    melted_data = normalized_data.melt(
        id_vars=['Name'],
        value_vars=key_metrics,
        var_name='Metric',
        value_name='Normalized Value'
    )
    
    # Create heatmap
    fig = px.density_heatmap(
        melted_data,
        x='Metric',
        y='Name',
        z='Normalized Value',
        color_continuous_scale='Viridis',
        title=f'Multi-Company Metric Comparison (Top {top_n} Companies)'
    )
    
    fig.update_layout(
        xaxis_title='Metric',
        yaxis_title='Company',
        xaxis_tickangle=-45,
        height=800
    )
    
    return fig

def create_time_series_charts(time_series_data, company_name):
    """Create time series visualization charts"""
    if time_series_data is None or len(time_series_data) == 0:
        return None
    
    # Convert date to string format for better display
    time_series_data['Date'] = time_series_data['Date'].dt.strftime('%Y-%m-%d')
    
    # Create subplot figure with 2 rows, 2 cols
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Return on Equity (ROE) Over Time',
            'Return on Capital Employed (ROCE) Over Time',
            'Free Cash Flow Trend',
            'Price to Earnings Ratio'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add ROE trace
    fig.add_trace(
        go.Scatter(
            x=time_series_data['Date'], 
            y=time_series_data['Return on equity'],
            name='ROE',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    
    # Add ROE moving average
    if 'ROE_MA' in time_series_data.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series_data['Date'], 
                y=time_series_data['ROE_MA'],
                name='ROE Moving Avg',
                line=dict(color='firebrick', dash='dot')
            ),
            row=1, col=1
        )
    
    # Add ROCE trace
    fig.add_trace(
        go.Scatter(
            x=time_series_data['Date'], 
            y=time_series_data['Return on capital employed'],
            name='ROCE',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # Add ROCE moving average
    if 'ROCE_MA' in time_series_data.columns:
        fig.add_trace(
            go.Scatter(
                x=time_series_data['Date'], 
                y=time_series_data['ROCE_MA'],
                name='ROCE Moving Avg',
                line=dict(color='orange', dash='dot')
            ),
            row=1, col=2
        )
    
    # Add Free Cash Flow trace
    fig.add_trace(
        go.Scatter(
            x=time_series_data['Date'], 
            y=time_series_data['Free cash flow'],
            name='Free Cash Flow',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Add P/E Ratio trace
    fig.add_trace(
        go.Scatter(
            x=time_series_data['Date'], 
            y=time_series_data['Price to Earning'],
            name='P/E Ratio',
            line=dict(color='teal')
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f'Financial Metrics Over Time - {company_name}',
        height=800,
        showlegend=True
    )
    
    return fig

def create_macro_charts(macro_data):
    """Create charts for macroeconomic indicators"""
    if not macro_data:
        return None
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'GDP Growth Rate (%)',
            'Inflation Rate (%)',
            'Interest Rate (%)',
            'Unemployment Rate (%)',
            'Market Index Performance',
            'Indicator Comparison'
        ),
        vertical_spacing=0.1
    )
    
    # Add traces for each indicator
    if 'GDP Growth' in macro_data:
        fig.add_trace(
            go.Scatter(
                x=macro_data['GDP Growth']['dates'],
                y=macro_data['GDP Growth']['history'],
                name='GDP Growth',
                line=dict(color='green')
            ),
            row=1, col=1
        )
    
    if 'Inflation' in macro_data:
        fig.add_trace(
            go.Scatter(
                x=macro_data['Inflation']['dates'],
                y=macro_data['Inflation']['history'],
                name='Inflation',
                line=dict(color='red')
            ),
            row=1, col=2
        )
    
    if 'Interest Rate' in macro_data:
        fig.add_trace(
            go.Scatter(
                x=macro_data['Interest Rate']['dates'],
                y=macro_data['Interest Rate']['history'],
                name='Interest Rate',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
    
    if 'Unemployment' in macro_data:
        fig.add_trace(
            go.Scatter(
                x=macro_data['Unemployment']['dates'],
                y=macro_data['Unemployment']['history'],
                name='Unemployment',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
    
    if 'Market Index' in macro_data:
        fig.add_trace(
            go.Scatter(
                x=macro_data['Market Index']['dates'],
                y=macro_data['Market Index']['history'],
                name='Market Index',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
    
    # Add comparison chart (latest values)
    indicators = ['GDP Growth', 'Inflation', 'Interest Rate', 'Unemployment']
    values = [macro_data[ind]['value'] for ind in indicators if ind in macro_data]
    
    fig.add_trace(
        go.Bar(
            x=indicators,
            y=values,
            name='Current Values',
            marker_color=['green', 'red', 'blue', 'orange']
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='Macroeconomic Indicators',
        height=1000,
        showlegend=True
    )
    
    return fig

def create_sentiment_charts(sentiment_data):
    """Create charts for sentiment analysis"""
    if not sentiment_data or 'sentiment_df' not in sentiment_data:
        return None
    
    sentiment_df = sentiment_data['sentiment_df']
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sentiment Distribution',
            'Sentiment Over Time',
            'Polarity vs Subjectivity',
            'Source Distribution'
        ),
        specs=[
            [{"type": "pie"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # Add sentiment distribution pie chart
    sentiment_counts = sentiment_df['Sentiment'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red
        ),
        row=1, col=1
    )
    
    # Add sentiment over time
    fig.add_trace(
        go.Scatter(
            x=sentiment_df['Date'],
            y=sentiment_df['Polarity'],
            mode='lines+markers',
            name='Sentiment Polarity',
            line=dict(color='royalblue')
        ),
        row=1, col=2
    )
    
    # Add polarity vs subjectivity scatter
    fig.add_trace(
        go.Scatter(
            x=sentiment_df['Polarity'],
            y=sentiment_df['Subjectivity'],
            mode='markers',
            marker=dict(
                size=10,
                color=sentiment_df['Polarity'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Polarity')
            ),
            text=sentiment_df['Title'],
            name='Articles'
        ),
        row=2, col=1
    )
    
    # Add source distribution
    source_counts = sentiment_df['Source'].value_counts().head(10)
    fig.add_trace(
        go.Bar(
            x=source_counts.index,
            y=source_counts.values,
            name='News Sources',
            marker_color='lightblue'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='Sentiment Analysis Results',
        height=800,
        showlegend=True
    )
    
    return fig

def create_side_by_side_comparison(df, selected_companies, metrics):
    """Create a side-by-side comparison of selected companies"""
    if df is None or len(selected_companies) == 0:
        return None
    
    # Filter dataframe for selected companies and metrics
    company_data = df[df['Name'].isin(selected_companies)]
    
    if len(company_data) == 0:
        return None
    
    # Create comparison figure
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Company Comparison',)
    )
    
    # Create a bar chart for each metric
    for i, company in enumerate(selected_companies):
        if company in company_data['Name'].values:
            company_row = company_data[company_data['Name'] == company].iloc[0]
            
            values = [company_row[metric] for metric in metrics if metric in company_row]
            
            fig.add_trace(
                go.Bar(
                    name=company,
                    x=metrics,
                    y=values,
                    text=values,
                    textposition='auto'
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f'Comparison of {", ".join(selected_companies)}',
        xaxis_title='Metrics',
        yaxis_title='Values',
        barmode='group',
        height=600
    )
    
    return fig

def create_download_link(df, filename, link_text):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def generate_pdf_report(ranked_df, company_name=None):
    """
    Generate a PDF report for a company or the top companies
    
    This is a placeholder function. In a real application, this would use
    a library like ReportLab or WeasyPrint to generate a PDF.
    """
    return "pdf_report.pdf"

# Function to handle direct API connections
def fetch_financial_data_api(company_code, api_key):
    """
    Fetch financial data from an external API
    
    This is a placeholder function. In a real application, this would
    connect to a financial data API service.
    """
    # For demo purposes, we'll return a notice
    return f"API connection established for {company_code}. API Key: {api_key[:3]}..."

# Main sidebar function
def setup_sidebar():
    """Set up the sidebar with controls and parameters"""
    st.sidebar.header("Controls & Parameters")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with company data", type=["csv"])
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use sample data", value=False)
    
    # Direct API connection
    api_section = st.sidebar.expander("API Connections", expanded=False)
    with api_section:
        st.markdown("Connect to financial data providers")
        api_key = st.text_input("Financial API Key", value="", type="password")
        if st.button("Connect to API"):
            if api_key:
                st.session_state.api_connections['financial_api_key'] = api_key
                st.success("API connection configured successfully!")
            else:
                st.error("Please enter an API key")
    
    # Weight sliders
    st.sidebar.subheader("Ranking Weights")
    
    # Use tabs for different weight categories
    weight_tabs = st.sidebar.tabs(["Financial", "Growth", "Investment Philosophy"])
    
    with weight_tabs[0]:
        roe_weight = st.slider("Return on Equity", 0.0, 0.5, 0.20, 0.05)
        roce_weight = st.slider("Return on Capital Employed", 0.0, 0.5, 0.20, 0.05)
        fcf_weight = st.slider("Free Cash Flow", 0.0, 0.5, 0.15, 0.05)
        pe_weight = st.slider("Price to Earnings", 0.0, 0.5, 0.15, 0.05)
        debt_weight = st.slider("Debt to Equity", 0.0, 0.5, 0.10, 0.05)
    
    with weight_tabs[1]:
        profit_growth_weight = st.slider("Profit Growth", 0.0, 0.5, 0.10, 0.05)
        sales_growth_weight = st.slider("Sales Growth", 0.0, 0.5, 0.05, 0.05)
        dividend_weight = st.slider("Dividend Yield", 0.0, 0.5, 0.05, 0.05)
    
    with weight_tabs[2]:
        buffett_weight = st.slider("Warren Buffett Criteria", 0.0, 0.5, 0.15, 0.05)
        lynch_weight = st.slider("Peter Lynch Criteria", 0.0, 0.5, 0.15, 0.05)
        graham_weight = st.slider("Benjamin Graham Criteria", 0.0, 0.5, 0.10, 0.05)
        dalio_weight = st.slider("Ray Dalio Criteria", 0.0, 0.5, 0.10, 0.05)
    
    # Calculate total weight for financial and growth metrics
    total_weight = (roe_weight + roce_weight + fcf_weight + pe_weight + debt_weight + 
                   profit_growth_weight + sales_growth_weight + dividend_weight)
    
    # Display total weight
    st.sidebar.markdown(f"Total financial & growth weight: {total_weight:.2f}")
    
    # Display total investment philosophy weight
    total_philosophy_weight = buffett_weight + lynch_weight + graham_weight + dalio_weight
    st.sidebar.markdown(f"Total philosophy weight: {total_philosophy_weight:.2f}")
    
    # Normalize weights if total is not 1.0
    if total_weight != 1.0:
        factor = 1.0 / total_weight
        roe_weight *= factor
        roce_weight *= factor
        fcf_weight *= factor
        pe_weight *= factor
        debt_weight *= factor
        profit_growth_weight *= factor
        sales_growth_weight *= factor
        dividend_weight *= factor
    
    # Create weights dictionary
    weights = {
        'ROE Score': roe_weight,
        'ROCE Score': roce_weight,
        'FCF Score': fcf_weight,
        'PE Score': pe_weight,
        'Debt Score': debt_weight,
        'Profit Growth Score': profit_growth_weight,
        'Sales Growth Score': sales_growth_weight,
        'Dividend Score': dividend_weight
    }
    
    # Create philosophy weights dictionary
    philosophy_weights = {
        'buffett_weight': buffett_weight,
        'lynch_weight': lynch_weight,
        'graham_weight': graham_weight,
        'dalio_weight': dalio_weight
    }
    
    # Cache control
    cache_section = st.sidebar.expander("Cache Control", expanded=False)
    with cache_section:
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared successfully!")
    
    # Get scheduled refresh TTL
    refresh_ttl = 3600  # 1 hour by default
    
    # Save analysis feature
    if st.sidebar.button("Save Current Analysis"):
        save_current_analysis()
    
    # Return values for use in the main app
    return uploaded_file, use_sample_data, weights, philosophy_weights, refresh_ttl

def save_current_analysis():
    """Save the current analysis to session state"""
    if 'ranked_df' in st.session_state and st.session_state.ranked_df is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analysis_name = f"Analysis {len(st.session_state.saved_analyses) + 1} - {timestamp}"
        
        st.session_state.saved_analyses[analysis_name] = {
            'data': st.session_state.ranked_df.to_dict(),
            'timestamp': timestamp
        }
        
        st.sidebar.success(f"Analysis saved as '{analysis_name}'!")
    else:
        st.sidebar.error("No analysis data to save!")

# Validate and process the uploaded data
@st.cache_data(ttl=3600)  # Cache for 1 hour by default
def validate_and_process_data(df, weights, philosophy_weights):
    """Validate and process the uploaded data"""
    # Check if required columns exist
    required_columns = [
        'Name', 'NSE Code', 'Return on equity', 'Return on capital employed', 
        'Free cash flow last year', 'Price to Earning', 'Debt to equity', 
        'OPM', 'Profit growth 5Years', 'Sales growth 5Years', 'Dividend yield',
        'Market Capitalization', 'PEG Ratio'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        missing_cols_str = ', '.join(missing_columns)
        raise ValueError(f"The uploaded CSV is missing the following required columns: {missing_cols_str}")
    
    # Calculate financial metrics
    df = calculate_financial_metrics(df)
    
    # Rank companies
    ranked_df = rank_companies(df, weights)
    
    # Add investment philosophy weights
    base_weight = 1 - sum(philosophy_weights.values())
    ranked_df['Ranking Score'] = ranked_df['Ranking Score'] * base_weight + \
                                (ranked_df['Buffett Score'] / 100) * philosophy_weights['buffett_weight'] + \
                                (ranked_df['Lynch Score'] / 100) * philosophy_weights['lynch_weight'] + \
                                (ranked_df['Graham Score'] / 100) * philosophy_weights['graham_weight'] + \
                                (ranked_df['Dalio Score'] / 100) * philosophy_weights['dalio_weight']
    
    # Re-rank based on updated scores
    ranked_df['Rank'] = ranked_df['Ranking Score'].rank(ascending=False, method='min').astype(int)
    ranked_df = ranked_df.sort_values('Rank')
    
    # Generate ranking reasons
    ranked_df['Ranking Reason'] = ranked_df.apply(lambda row: generate_ranking_reason(row, ranked_df), axis=1)
    
    return ranked_df

# Display sample data and instructions
def show_sample_data():
    """Display sample data and instructions when no file is uploaded"""
    st.info("Please upload a CSV file with company financial data or use the sample data option to begin analysis.")
    
    st.markdown("### Required CSV Format")
    st.markdown("""
    Your CSV file should include the following columns:
    - **Name**: Company name
    - **NSE Code**: Stock symbol on NSE
    - **Return on equity**: ROE percentage
    - **Return on capital employed**: ROCE percentage
    - **Free cash flow last year**: FCF in crores
    - **Price to Earning**: P/E ratio
    - **Debt to equity**: D/E ratio
    - **OPM**: Operating Profit Margin percentage
    - **Profit growth 5Years**: 5-year profit growth percentage
    - **Sales growth 5Years**: 5-year sales growth percentage
    - **Dividend yield**: Dividend yield percentage
    - **Market Capitalization**: Market cap in crores
    - **PEG Ratio**: Price/Earnings to Growth ratio
    """)
    
    # Sample data download

    # Sample data download
    st.markdown("### Download Sample Data Format")
    sample_data = pd.DataFrame({
        'Name': ['Company A', 'Company B', 'Company C'],
        'NSE Code': ['COMPA', 'COMPB', 'COMPC'],
        'Return on equity': [25.5, 18.3, 12.7],
        'Return on capital employed': [30.2, 22.1, 15.8],
        'Free cash flow last year': [1250.5, 850.3, 320.7],
        'Price to Earning': [22.5, 18.3, 15.2],
        'Debt to equity': [0.25, 0.42, 0.65],
        'OPM': [28.5, 22.3, 15.7],
        'Profit growth 5Years': [15.3, 12.5, 8.2],
        'Sales growth 5Years': [18.5, 14.2, 10.5],
        'Dividend yield': [2.5, 3.2, 1.8],
        'Market Capitalization': [25000, 15000, 8000],
        'PEG Ratio': [1.2, 1.5, 1.8]
    })
    
    st.markdown(create_download_link(sample_data, "sample_data_format.csv", "Download Sample CSV Template"), unsafe_allow_html=True)

def generate_sample_it_data():
    """Generate a sample dataset of IT companies"""
    return pd.DataFrame({
        'Name': ['TCS', 'Infosys', 'HCL Tech', 'Wipro', 'Tech Mahindra', 
                'L&T Infotech', 'Mindtree', 'Mphasis', 'Oracle Financial', 'Coforge',
                'Persistent Systems', 'Birlasoft', 'Hexaware', 'NIIT Tech', 'Zensar'],
        'NSE Code': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 
                    'LTI', 'MINDTREE', 'MPHASIS', 'OFSS', 'COFORGE',
                    'PERSISTENT', 'BSOFT', 'HEXAWARE', 'NITEC', 'ZENSARTECH'],
        'Return on equity': [51.51, 31.83, 23.30, 18.75, 15.42, 
                            27.51, 24.82, 19.96, 31.05, 20.83,
                            18.76, 16.92, 21.34, 19.25, 14.86],
        'Return on capital employed': [64.28, 39.99, 29.60, 24.15, 19.87, 
                                    35.21, 32.18, 26.57, 38.92, 27.46,
                                    25.11, 21.63, 28.43, 24.79, 18.94],
        'Free cash flow last year': [41688.00, 23009.00, 21432.00, 12500.00, 8750.00, 
                                    3542.00, 2765.00, 4123.00, 5432.00, 2187.00,
                                    1954.00, 1245.00, 1876.00, 1543.00, 987.00],
        'Price to Earning': [26.36, 24.17, 24.77, 28.50, 22.30, 
                            31.42, 29.86, 26.93, 19.84, 27.50,
                            32.14, 24.76, 22.93, 25.18, 29.65],
        'Debt to equity': [0.09, 0.09, 0.08, 0.12, 0.15, 
                        0.11, 0.14, 0.10, 0.05, 0.21,
                        0.18, 0.22, 0.16, 0.19, 0.25],
        'OPM': [25.50, 23.84, 21.75, 18.90, 16.50, 
                22.40, 20.93, 19.82, 24.65, 18.76,
                17.94, 16.85, 18.43, 17.65, 15.78],
        'Profit growth 5Years': [10.46, 11.24, 12.71, 8.50, 7.20, 
                                13.25, 12.87, 10.96, 9.65, 11.32,
                                14.76, 9.85, 11.54, 10.23, 8.76],
        'Sales growth 5Years': [10.46, 13.20, 12.71, 9.30, 8.50, 
                                14.65, 13.54, 11.87, 8.96, 12.43,
                                15.32, 10.76, 12.65, 11.34, 9.87],
        'Dividend yield': [1.50, 1.80, 2.10, 1.20, 1.90, 
                        1.65, 1.32, 1.87, 2.54, 1.42,
                        1.21, 1.76, 1.98, 1.56, 1.24],
        'Market Capitalization': [1250000, 750000, 450000, 350000, 120000, 
                                95000, 82000, 76000, 68000, 45000,
                                38000, 32000, 28000, 25000, 18000],
        'PEG Ratio': [2.52, 2.15, 1.95, 3.35, 3.10, 
                    2.37, 2.32, 2.46, 2.06, 2.43,
                    2.18, 2.51, 1.99, 2.46, 3.38],
        'Sector': ['IT Services', 'IT Services', 'IT Services', 'IT Services', 'IT Services', 
                'IT Services', 'IT Services', 'IT Services', 'IT Services', 'IT Services',
                'IT Services', 'IT Services', 'IT Services', 'IT Services', 'IT Services'],
        'Current ratio': [2.5, 2.8, 2.2, 1.9, 2.1, 
                        2.4, 2.6, 2.3, 3.1, 2.0,
                        2.2, 1.8, 2.5, 2.3, 1.7],
        'Price to Book': [8.5, 6.2, 4.8, 5.3, 3.9, 
                        6.7, 5.8, 4.9, 7.1, 4.6,
                        5.2, 3.8, 4.5, 4.1, 3.6],
        'Earnings volatility': [0.12, 0.15, 0.18, 0.22, 0.25, 
                            0.16, 0.19, 0.21, 0.14, 0.23,
                            0.20, 0.26, 0.17, 0.22, 0.28]
    })

# Display rankings tab
def display_rankings_tab(ranked_df):
    """Display the rankings tab with top companies and their metrics"""
    st.markdown('<h2 class="sub-header">Company Rankings</h2>', unsafe_allow_html=True)
    st.markdown("Companies ranked based on financial metrics and investment criteria")
    
    # Create a top metrics dashboard
    st.markdown('<h3 class="sub-header">Industry Overview</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-metric-title">Average ROE</div>', unsafe_allow_html=True)
        avg_roe = ranked_df['Return on equity'].mean()
        st.markdown(f'<div class="dashboard-metric-value">{avg_roe:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-metric-title">Average P/E Ratio</div>', unsafe_allow_html=True)
        avg_pe = ranked_df['Price to Earning'].mean()
        st.markdown(f'<div class="dashboard-metric-value">{avg_pe:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-metric-title">Avg Profit Growth (5Y)</div>', unsafe_allow_html=True)
        avg_growth = ranked_df['Profit growth 5Years'].mean()
        st.markdown(f'<div class="dashboard-metric-value">{avg_growth:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-metric-title">Total Market Cap</div>', unsafe_allow_html=True)
        total_mcap = ranked_df['Market Capitalization'].sum() / 100000  # Convert to lakh crores
        st.markdown(f'<div class="dashboard-metric-value">₹{total_mcap:.2f}L Cr</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display top 10 companies with details
    st.markdown('<h3 class="sub-header">Top 10 Companies</h3>', unsafe_allow_html=True)
    for i, (_, row) in enumerate(ranked_df.head(10).iterrows()):
        with st.expander(f"{i+1}. {row['Name']} (NSE: {row['NSE Code']})"):
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<span class="company-name">{row["Name"]}</span> <span class="company-code">({row["NSE Code"]})</span>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Rank:** <span class='metric-value'>{row['Rank']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Ranking Score:** <span class='metric-value'>{row['Ranking Score']:.4f}</span>", unsafe_allow_html=True)
                st.markdown(f"**ROE:** <span class='metric-value'>{row['Return on equity']:.2f}%</span>", unsafe_allow_html=True)
                st.markdown(f"**ROCE:** <span class='metric-value'>{row['Return on capital employed']:.2f}%</span>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Free Cash Flow:** <span class='metric-value'>₹{row['Free cash flow last year']:.2f} cr</span>", unsafe_allow_html=True)
                st.markdown(f"**P/E Ratio:** <span class='metric-value'>{row['Price to Earning']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Debt to Equity:** <span class='metric-value'>{row['Debt to equity']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Market Cap:** <span class='metric-value'>₹{row['Market Capitalization']:.2f} cr</span>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**Buffett Score:** <span class='metric-value'>{row['Buffett Score']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Lynch Score:** <span class='metric-value'>{row['Lynch Score']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Graham Score:** <span class='metric-value'>{row['Graham Score']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Dalio Score:** <span class='metric-value'>{row['Dalio Score']:.2f}</span>", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Add additional metrics and growth analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Growth Metrics")
                st.markdown(f"**Sales Growth (5Y):** <span class='metric-value'>{row['Sales growth 5Years']:.2f}%</span>", unsafe_allow_html=True)
                st.markdown(f"**Profit Growth (5Y):** <span class='metric-value'>{row['Profit growth 5Years']:.2f}%</span>", unsafe_allow_html=True)
                st.markdown(f"**PEG Ratio:** <span class='metric-value'>{row['PEG Ratio']:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Dividend Yield:** <span class='metric-value'>{row['Dividend yield']:.2f}%</span>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Company Analysis")
                
                # Add buttons for time series and sentiment analysis
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"Time Series Analysis for {row['Name']}", key=f"ts_{row['NSE Code']}"):
                        st.session_state.time_series_data = analyze_time_series(pd.DataFrame([row]), 'quarterly')
                        st.session_state.selected_company = row['Name']
                
                with col_b:
                    if st.button(f"Sentiment Analysis for {row['Name']}", key=f"sent_{row['NSE Code']}"):
                        st.session_state.sentiment_data = fetch_and_analyze_company_sentiment(row['Name'], row['NSE Code'])
                        st.session_state.selected_company = row['Name']
            
            st.markdown(f'<div class="ranking-reason"><strong>Why Ranked #{row["Rank"]}:</strong> {row["Ranking Reason"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Download links for ranked data
    st.markdown("### Download Ranked Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(create_download_link(ranked_df, "ranked_companies.csv", "Download Full Rankings CSV"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_download_link(ranked_df.head(10), "top10_companies.csv", "Download Top 10 Companies CSV"), unsafe_allow_html=True)
    
    # Company selector for detailed analysis
    st.markdown('<h3 class="sub-header">Select Companies for Comparison</h3>', unsafe_allow_html=True)
    
    # Allow user to select multiple companies for comparison
    selected_companies = st.multiselect(
        "Select companies to compare:",
        options=ranked_df['Name'].tolist(),
        default=ranked_df['Name'].head(3).tolist()
    )
    
    if selected_companies:
        st.session_state.selected_companies = selected_companies
        
        # Add to comparison button
        if st.button("Compare Selected Companies"):
            st.success(f"Companies added to comparison: {', '.join(selected_companies)}")
            # Switch to the Comparison tab is handled in the main app

# Display visualizations tab
def display_visualizations_tab(ranked_df):
    """Display the visualizations tab with charts and graphs"""
    st.markdown('<h2 class="sub-header">Advanced Visualizations</h2>', unsafe_allow_html=True)
    
    # Create charts using Plotly
    if ranked_df is not None and len(ranked_df) > 0:
        # Create tabs for different visualization categories
        viz_tabs = st.tabs(["Rankings", "Metrics Analysis", "Investment Styles", "Custom Visualizations"])
        
        with viz_tabs[0]:
            st.markdown("### Top Companies by Ranking Score")
            
            # Top companies ranking chart
            fig_ranking = px.bar(
                ranked_df.head(15), 
                x='Name', 
                y='Ranking Score',
                color='Ranking Score',
                hover_data=['NSE Code', 'Rank', 'Return on equity', 'Return on capital employed'],
                title='Top 15 Companies by Ranking Score',
                color_continuous_scale='Viridis'
            )
            fig_ranking.update_layout(
                xaxis_title='Company',
                yaxis_title='Ranking Score',
                xaxis_tickangle=-45,
                height=600
            )
            st.plotly_chart(fig_ranking, use_container_width=True)
            
            # Metrics heatmap
            st.markdown("### Multi-Company Metric Comparison")
            metrics_heatmap = create_metrics_heatmap(ranked_df)
            if metrics_heatmap:
                st.plotly_chart(metrics_heatmap, use_container_width=True)
        
        with viz_tabs[1]:
            st.markdown("### ROE vs ROCE Analysis")
            
            # ROE vs ROCE scatter plot
            fig_roe_roce = px.scatter(
                ranked_df.head(20),
                x='Return on equity',
                y='Return on capital employed',
                size='Market Capitalization',
                color='Rank',
                hover_name='Name',
                hover_data=['NSE Code', 'Free cash flow last year', 'Price to Earning'],
                size_max=60,
                color_continuous_scale='Viridis_r',
                title='ROE vs ROCE Analysis (Bubble Size = Market Cap)'
            )
            fig_roe_roce.update_layout(
                xaxis_title='Return on Equity (%)',
                yaxis_title='Return on Capital Employed (%)',
                height=600
            )
            st.plotly_chart(fig_roe_roce, use_container_width=True)
            
            # Correlation matrix
            st.markdown("### Correlation Matrix of Key Financial Metrics")
            corr_matrix = create_correlation_matrix(ranked_df)
            if corr_matrix:
                st.plotly_chart(corr_matrix, use_container_width=True)
            
            # P/E vs Growth Rate
            st.markdown("### P/E Ratio vs Growth Rate")
            fig_pe_growth = px.scatter(
                ranked_df.head(20),
                x='Price to Earning',
                y='Profit growth 5Years',
                size='Market Capitalization',
                color='PEG Ratio',
                hover_name='Name',
                hover_data=['NSE Code', 'Return on equity', 'Rank'],
                size_max=60,
                color_continuous_scale='RdYlGn_r',  # Red for high PEG (expensive), green for low PEG (cheap)
                title='P/E Ratio vs Growth Rate (Bubble Size = Market Cap, Color = PEG Ratio)'
            )
            fig_pe_growth.update_layout(
                xaxis_title='Price to Earnings Ratio',
                yaxis_title='5-Year Profit Growth (%)',
                height=600
            )
            
            # Add a PEG = 1 reference line
            pe_range = list(range(0, int(ranked_df['Price to Earning'].max()) + 5, 5))
            growth_values = pe_range  # For PEG = 1, Growth = P/E
            
            fig_pe_growth.add_trace(
                go.Scatter(
                    x=pe_range,
                    y=growth_values,
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='PEG = 1 (Fair Value)'
                )
            )
            
            st.plotly_chart(fig_pe_growth, use_container_width=True)
        
        with viz_tabs[2]:
            st.markdown("### Investment Philosophy Comparison")
            
            # Investment philosophy comparison
            fig_philosophy = px.scatter(
                ranked_df.head(20),
                x='Buffett Score',
                y='Lynch Score',
                size='Market Capitalization',
                color='Graham Score',
                hover_name='Name',
                hover_data=['NSE Code', 'Rank', 'Dalio Score'],
                size_max=60,
                color_continuous_scale='Viridis',
                title='Investment Philosophy Comparison (Color = Graham Score)'
            )
            fig_philosophy.update_layout(
                xaxis_title='Buffett Score',
                yaxis_title='Lynch Score',
                height=600
            )
            st.plotly_chart(fig_philosophy, use_container_width=True)
            
            # Radar Chart for Top 5 Companies Investment Styles
            st.markdown("### Investment Style Radar Chart (Top 5 Companies)")
            
            top5 = ranked_df.head(5)
            categories = ['Buffett Score', 'Lynch Score', 'Graham Score', 'Dalio Score', 'Custom Strategy Score']
            
            # Normalize the scores to 0-1 range for better visualization
            normalized_data = pd.DataFrame()
            normalized_data['Name'] = top5['Name']
            
            for category in categories:
                if category in top5.columns:
                    normalized_data[category] = top5[category] / 100
                else:
                    normalized_data[category] = 0.5  # Default value if the score doesn't exist
            
            # Create radar chart
            fig_radar = go.Figure()
            
            for i, (idx, row) in enumerate(normalized_data.iterrows()):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[cat] for cat in categories],
                    theta=categories,
                    fill='toself',
                    name=row['Name']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title='Investment Philosophy Alignment - Top 5 Companies',
                height=600
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with viz_tabs[3]:
            st.markdown("### Custom Visualization Builder")
            
            # Allow user to select X and Y axes for custom scatter plot
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_axis = st.selectbox(
                    "Select X-Axis Metric:",
                    options=[col for col in ranked_df.select_dtypes(include=[np.number]).columns if not col.endswith('Score')],
                    index=list(ranked_df.columns).index('Return on equity') if 'Return on equity' in ranked_df.columns else 0
                )
            
            with col2:
                y_axis = st.selectbox(
                    "Select Y-Axis Metric:",
                    options=[col for col in ranked_df.select_dtypes(include=[np.number]).columns if not col.endswith('Score')],
                    index=list(ranked_df.columns).index('Price to Earning') if 'Price to Earning' in ranked_df.columns else 0
                )
            
            with col3:
                color_by = st.selectbox(
                    "Color By:",
                    options=['Rank', 'Buffett Score', 'Lynch Score', 'Graham Score', 'Dalio Score', 'Custom Strategy Score', 'ROE Score', 'Debt Score'],
                    index=0
                )
            
            # Create custom scatter plot
            fig_custom = px.scatter(
                ranked_df.head(20),
                x=x_axis,
                y=y_axis,
                size='Market Capitalization',
                color=color_by,
                hover_name='Name',
                hover_data=['NSE Code', 'Rank'],
                size_max=60,
                title=f'Custom Visualization: {x_axis} vs {y_axis} (Color = {color_by})'
            )
            fig_custom.update_layout(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                height=600
            )
            
            st.plotly_chart(fig_custom, use_container_width=True)
    else:
        st.info("Upload data or use sample data to generate visualizations.")

# Display investment philosophy tab
def display_investment_tab(ranked_df):
    """Display the investment philosophy tab with different investment strategies"""
    st.markdown('<h2 class="sub-header">Investment Philosophies</h2>', unsafe_allow_html=True)
    
    # Create tabs for different investment philosophies
    philosophy_tabs = st.tabs(["Warren Buffett", "Peter Lynch", "Benjamin Graham", "Ray Dalio", "Custom Strategy"])
    
    with philosophy_tabs[0]:
        st.markdown("## Warren Buffett Investment Philosophy")
        st.markdown("""
        Warren Buffett's investment approach focuses on identifying companies with strong fundamentals, 
        sustainable competitive advantages (economic moats), and excellent management. His philosophy 
        emphasizes long-term value investing rather than short-term market fluctuations.
        """)
        
        st.markdown("### Key Principles")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Consistent ROE**: Return on equity consistently above 15% indicates a sustainable competitive advantage
            - **Low Debt**: Low debt-to-equity ratio (< 0.5) shows financial stability and reduced risk
            - **High Margins**: High operating profit margins (> 20%) indicate pricing power and competitive advantage
            - **Earnings Growth**: Consistent earnings growth over 5+ years shows business strength
            """)
        
        with col2:
            st.markdown("""
            - **Economic Moat**: Companies with strong competitive advantages that protect market share and profitability
            - **Circle of Competence**: Investing in businesses that are understandable
            - **Management Integrity**: Honest and capable management with shareholder interests at heart
            - **Margin of Safety**: Buying at a price significantly below intrinsic value
            """)
        
        if ranked_df is not None and len(ranked_df) > 0:
            buffett_companies = get_buffett_companies(ranked_df)
            
            if not buffett_companies.empty:
                st.markdown("### Top Companies Matching Buffett's Criteria")
                
                # Create bar chart for Buffett scores
                fig_buffett = px.bar(
                    buffett_companies.head(10), 
                    x='Name', 
                    y='Buffett Score',
                    color='Buffett Score',
                    hover_data=['NSE Code', 'Rank'],
                    title='Top Companies by Buffett Score',
                    color_continuous_scale='Blues'
                )
                fig_buffett.update_layout(
                    xaxis_title='Company',
                    yaxis_title='Buffett Score',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_buffett, use_container_width=True)
                
                st.dataframe(buffett_companies.head(10))
                st.markdown(create_download_link(buffett_companies, "buffett_companies.csv", "Download Buffett Companies CSV"), unsafe_allow_html=True)
            else:
                st.info("No companies strongly match Warren Buffett's investment criteria based on our analysis.")
    
    with philosophy_tabs[1]:
        st.markdown("## Peter Lynch Investment Philosophy")
        st.markdown("""
        Peter Lynch's investment strategy focuses on finding companies with good growth at reasonable prices. 
        He believes in investing in what you know and understand, and looks for companies with strong growth 
        potential that are undervalued by the market.
        """)
        
        st.markdown("### Key Principles")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **PEG Ratio**: Price/Earnings to Growth ratio < 1 is excellent, < 2 is good
            - **Growth Potential**: Companies with strong growth potential in sales and earnings
            - **Reasonable P/E**: P/E ratio should be reasonable relative to growth rate
            - **Strong Balance Sheet**: Low debt and strong financial position
            """)
        
        with col2:
            st.markdown("""
            - **Cash Flow**: Strong and consistent cash flow generation
            - **Invest in What You Know**: Understand the business and industry
            - **Classification System**: Categorizing stocks (stalwarts, fast growers, slow growers, etc.)
            - **Story & Numbers**: Both the business narrative and financials must align
            """)
        
        if ranked_df is not None and len(ranked_df) > 0:
            lynch_companies = get_lynch_companies(ranked_df)
            
            if not lynch_companies.empty:
                st.markdown("### Top Companies Matching Lynch's Criteria")
                
                # Create bar chart for Lynch scores
                fig_lynch = px.bar(
                    lynch_companies.head(10), 
                    x='Name', 
                    y='Lynch Score',
                    color='Lynch Score',
                    hover_data=['NSE Code', 'Rank'],
                    title='Top Companies by Lynch Score',
                    color_continuous_scale='Greens'
                )
                fig_lynch.update_layout(
                    xaxis_title='Company',
                    yaxis_title='Lynch Score',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_lynch, use_container_width=True)
                
                st.dataframe(lynch_companies.head(10))
                st.markdown(create_download_link(lynch_companies, "lynch_companies.csv", "Download Lynch Companies CSV"), unsafe_allow_html=True)
            else:
                st.info("No companies strongly match Peter Lynch's investment criteria based on our analysis.")
    
    with philosophy_tabs[2]:
        st.markdown("## Benjamin Graham Investment Philosophy")
        st.markdown("""
        Benjamin Graham, known as the father of value investing, focused on buying undervalued companies 
        with strong balance sheets and consistent earnings. His approach emphasizes margin of safety and 
        fundamental analysis rather than market sentiment or growth projections.
        """)
        
        st.markdown("### Key Principles")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Adequate Size**: Sufficient market cap to ensure stability
            - **Strong Financial Condition**: Current ratio > 2.0
            - **Earnings Stability**: Positive earnings for the past 5 years
            - **Dividend Record**: Uninterrupted dividend payments
            """)
        
        with col2:
            st.markdown("""
            - **Earnings Growth**: At least 5% increase over 5 years
            - **Moderate P/E Ratio**: P/E ratio < 15
            - **Moderate Price to Book**: P/B ratio < 1.5
            - **Margin of Safety**: Buy at a significant discount to intrinsic value
            """)
        
        if ranked_df is not None and len(ranked_df) > 0:
            graham_companies = get_graham_companies(ranked_df)
            
            if not graham_companies.empty:
                st.markdown("### Top Companies Matching Graham's Criteria")
                
                # Create bar chart for Graham scores
                fig_graham = px.bar(
                    graham_companies.head(10), 
                    x='Name', 
                    y='Graham Score',
                    color='Graham Score',
                    hover_data=['NSE Code', 'Rank'],
                    title='Top Companies by Graham Score',
                    color_continuous_scale='Oranges'
                )
                fig_graham.update_layout(
                    xaxis_title='Company',
                    yaxis_title='Graham Score',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_graham, use_container_width=True)
                
                st.dataframe(graham_companies.head(10))
                st.markdown(create_download_link(graham_companies, "graham_companies.csv", "Download Graham Companies CSV"), unsafe_allow_html=True)
            else:
                st.info("No companies strongly match Benjamin Graham's investment criteria based on our analysis.")
    
    with philosophy_tabs[3]:
        st.markdown("## Ray Dalio Investment Philosophy")
        st.markdown("""
        Ray Dalio, founder of Bridgewater Associates, focuses on understanding economic cycles and creating 
        all-weather portfolio strategies. His principles emphasize diversification, risk parity, and understanding 
        the macroeconomic forces that drive asset prices.
        """)
        
        st.markdown("### Key Principles")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Strong Return on Assets (ROA)**: Efficiently utilizing assets
            - **Low Debt to EBITDA**: Conservative debt levels relative to earnings
            - **Consistent Cash Flow**: Strong and reliable cash flow generation
            - **Low Earnings Volatility**: Stable and predictable earnings
            """)
        
        with col2:
            st.markdown("""
            - **Diversified Revenue Streams**: Multiple sources of income
            - **Economic Machine Understanding**: Alignment with economic principles
            - **Risk Parity**: Balancing risk across different economic scenarios
            - **Systematic Decision Making**: Rule-based investment approach
            """)
        
        if ranked_df is not None and len(ranked_df) > 0:
            dalio_companies = get_dalio_companies(ranked_df)
            
            if not dalio_companies.empty:
                st.markdown("### Top Companies Matching Dalio's Criteria")
                
                # Create bar chart for Dalio scores
                fig_dalio = px.bar(
                    dalio_companies.head(10), 
                    x='Name', 
                    y='Dalio Score',
                    color='Dalio Score',
                    hover_data=['NSE Code', 'Rank'],
                    title='Top Companies by Dalio Score',
                    color_continuous_scale='Purples'
                )
                fig_dalio.update_layout(
                    xaxis_title='Company',
                    yaxis_title='Dalio Score',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_dalio, use_container_width=True)
                
                st.dataframe(dalio_companies.head(10))
                st.markdown(create_download_link(dalio_companies, "dalio_companies.csv", "Download Dalio Companies CSV"), unsafe_allow_html=True)
            else:
                st.info("No companies strongly match Ray Dalio's investment criteria based on our analysis.")
    
    with philosophy_tabs[4]:
        st.markdown("## Custom Investment Strategy Builder")
        st.markdown("""
        Build your own investment strategy by setting criteria and weights for different financial metrics. 
        This allows you to create a personalized approach that aligns with your investment goals and risk tolerance.
        """)
        
        st.markdown("### Define Your Strategy Parameters")
        
        # Create columns for parameters
        col1, col2, col3 = st.columns(3)
        
        # Dictionary to store parameters
        custom_params = {}
        
        # ROE settings
        with col1:
            st.markdown("#### Return on Equity (ROE)")
            roe_threshold = st.number_input("ROE Threshold (%)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
            roe_direction = st.radio("ROE Direction", ["Higher is better", "Lower is better"], index=0, key="roe_dir")
            roe_weight = st.slider("ROE Weight", 0.0, 1.0, 0.2, 0.05, key="roe_weight")
            
            custom_params['Return on equity_threshold'] = roe_threshold
            custom_params['Return on equity_direction'] = roe_direction == "Higher is better"
            custom_params['Return on equity_weight'] = roe_weight
        
        # P/E settings
        with col2:
            st.markdown("#### Price to Earnings (P/E)")
            pe_threshold = st.number_input("P/E Threshold", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            pe_direction = st.radio("P/E Direction", ["Higher is better", "Lower is better"], index=1, key="pe_dir")
            pe_weight = st.slider("P/E Weight", 0.0, 1.0, 0.2, 0.05, key="pe_weight")
            
            custom_params['Price to Earning_threshold'] = pe_threshold
            custom_params['Price to Earning_direction'] = pe_direction == "Higher is better"
            custom_params['Price to Earning_weight'] = pe_weight
        
        # Debt to Equity settings
        with col3:
            st.markdown("#### Debt to Equity")
            debt_threshold = st.number_input("Debt to Equity Threshold", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
            debt_direction = st.radio("Debt Direction", ["Higher is better", "Lower is better"], index=1, key="debt_dir")
            debt_weight = st.slider("Debt Weight", 0.0, 1.0, 0.15, 0.05, key="debt_weight")
            
            custom_params['Debt to equity_threshold'] = debt_threshold
            custom_params['Debt to equity_direction'] = debt_direction == "Higher is better"
            custom_params['Debt to equity_weight'] = debt_weight
        
        # Second row of parameters
        col4, col5, col6 = st.columns(3)
        
        # Profit Growth settings
        with col4:
            st.markdown("#### Profit Growth (5Y)")
            profit_threshold = st.number_input("Profit Growth Threshold (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
            profit_direction = st.radio("Profit Growth Direction", ["Higher is better", "Lower is better"], index=0, key="profit_dir")
            profit_weight = st.slider("Profit Growth Weight", 0.0, 1.0, 0.15, 0.05, key="profit_weight")
            
            custom_params['Profit growth 5Years_threshold'] = profit_threshold
            custom_params['Profit growth 5Years_direction'] = profit_direction == "Higher is better"
            custom_params['Profit growth 5Years_weight'] = profit_weight
        
        # Free Cash Flow settings
        with col5:
            st.markdown("#### Free Cash Flow")
            fcf_threshold = st.number_input("FCF Threshold (cr)", min_value=0.0, max_value=50000.0, value=1000.0, step=100.0)
            fcf_direction = st.radio("FCF Direction", ["Higher is better", "Lower is better"], index=0, key="fcf_dir")
            fcf_weight = st.slider("FCF Weight", 0.0, 1.0, 0.15, 0.05, key="fcf_weight")
            
            custom_params['Free cash flow last year_threshold'] = fcf_threshold
            custom_params['Free cash flow last year_direction'] = fcf_direction == "Higher is better"
            custom_params['Free cash flow last year_weight'] = fcf_weight
        
        # Dividend Yield settings
        with col6:
            st.markdown("#### Dividend Yield")
            div_threshold = st.number_input("Dividend Yield Threshold (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            div_direction = st.radio("Dividend Direction", ["Higher is better", "Lower is better"], index=0, key="div_dir")
            div_weight = st.slider("Dividend Weight", 0.0, 1.0, 0.15, 0.05, key="div_weight")
            
            custom_params['Dividend yield_threshold'] = div_threshold
            custom_params['Dividend yield_direction'] = div_direction == "Higher is better"
            custom_params['Dividend yield_weight'] = div_weight
        
        # Save custom strategy parameters to session state
        if st.button("Apply Custom Strategy"):
            st.session_state.custom_strategy_params = custom_params
            
            if ranked_df is not None and len(ranked_df) > 0:
                # Apply custom strategy to the data
                custom_ranked_df = build_custom_strategy(ranked_df.copy(), custom_params)
                
                # Sort by Custom Strategy Score
                custom_ranked_df = custom_ranked_df.sort_values('Custom Strategy Score', ascending=False)
                
                st.session_state.custom_ranked_df = custom_ranked_df
                
                st.success("Custom strategy applied successfully!")
                
                # Display top companies based on custom strategy
                st.markdown("### Top Companies Based on Your Custom Strategy")
                
                top_custom = custom_ranked_df.head(10)[['Name', 'NSE Code', 'Custom Strategy Score', 'Rank']]
                
                # Create bar chart for custom scores
                fig_custom = px.bar(
                    top_custom, 
                    x='Name', 
                    y='Custom Strategy Score',
                    color='Custom Strategy Score',
                    hover_data=['NSE Code', 'Rank'],
                    title='Top Companies by Your Custom Strategy',
                    color_continuous_scale='Turbo'
                )
                fig_custom.update_layout(
                    xaxis_title='Company',
                    yaxis_title='Custom Strategy Score',
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_custom, use_container_width=True)
                
                st.dataframe(top_custom)
                st.markdown(create_download_link(top_custom, "custom_strategy_companies.csv", "Download Custom Strategy Companies CSV"), unsafe_allow_html=True)
            else:
                st.warning("Please upload data or use sample data to apply your custom strategy.")

# Display data explorer tab
def display_data_explorer_tab(ranked_df):
    """Display the data explorer tab with data filtering and search functionality"""
    st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
    st.markdown("Explore the complete dataset with all calculated metrics")
    
    if ranked_df is not None and len(ranked_df) > 0:
        # Add search and filter functionality
        st.markdown("### Filter and Search")
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("Search by Company Name or NSE Code")
        
        with col2:
            # Create filter ranges for important metrics
            filter_metric = st.selectbox(
                "Filter by Metric",
                options=[
                    "None", "Return on equity", "Return on capital employed", 
                    "Free cash flow last year", "Price to Earning", "Debt to equity",
                    "Buffett Score", "Lynch Score", "Graham Score", "Dalio Score"
                ]
            )
        
        if filter_metric != "None":
            col3, col4 = st.columns(2)
            
            # Get min and max values for the selected metric
            min_val = float(ranked_df[filter_metric].min())
            max_val = float(ranked_df[filter_metric].max())
            
            with col3:
                min_filter = st.number_input(f"Minimum {filter_metric}", 
                                           min_value=min_val, 
                                           max_value=max_val, 
                                           value=min_val)
            
            with col4:
                max_filter = st.number_input(f"Maximum {filter_metric}", 
                                           min_value=min_val, 
                                           max_value=max_val, 
                                           value=max_val)
            
            # Apply filter
            filtered_df = ranked_df[(ranked_df[filter_metric] >= min_filter) & 
                                    (ranked_df[filter_metric] <= max_filter)]
        else:
            filtered_df = ranked_df
        
        # Apply search term if provided
        if search_term:
            filtered_df = filtered_df[
                filtered_df['Name'].str.contains(search_term, case=False) | 
                filtered_df['NSE Code'].str.contains(search_term, case=False)
            ]
        
        # Column selector
        st.markdown("### Select Columns to Display")
        all_columns = ranked_df.columns.tolist()
        default_columns = ['Name', 'NSE Code', 'Rank', 'Ranking Score', 'Return on equity', 
                          'Return on capital employed', 'Free cash flow last year', 'Price to Earning',
                          'Buffett Score', 'Lynch Score', 'Graham Score', 'Dalio Score']
        selected_columns = st.multiselect("Select columns", all_columns, default=default_columns)
        
        if selected_columns:
            # Display the filtered dataframe
            st.dataframe(filtered_df[selected_columns], height=500)
        else:
            st.dataframe(filtered_df, height=500)
        
        # Download link for filtered data
        st.markdown("### Download Filtered Data")
        st.markdown(create_download_link(filtered_df, "filtered_data.csv", "Download Filtered Data CSV"), unsafe_allow_html=True)
    else:
        st.info("Upload data or use sample data to explore.")

# Display time series analysis tab
def display_time_series_tab(time_series_data, company_name):
    """Display the time series analysis tab with historical performance charts"""
    st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)
    
    if time_series_data is not None and len(time_series_data) > 0:
        st.markdown(f"### Historical Performance for {company_name}")
        
        # Create time series chart
        fig = create_time_series_charts(time_series_data, company_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate performance metrics
        st.markdown("### Key Performance Indicators")
        
        # Get the oldest and newest data points
        oldest = time_series_data.iloc[0]
        newest = time_series_data.iloc[-1]
        
        # Calculate changes
        roe_change = (newest['Return on equity'] - oldest['Return on equity']) / oldest['Return on equity'] * 100
        roce_change = (newest['Return on capital employed'] - oldest['Return on capital employed']) / oldest['Return on capital employed'] * 100
        fcf_change = (newest['Free cash flow'] - oldest['Free cash flow']) / oldest['Free cash flow'] * 100 if oldest['Free cash flow'] != 0 else 0
        pe_change = (newest['Price to Earning'] - oldest['Price to Earning']) / oldest['Price to Earning'] * 100
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### ROE Trend")
            st.markdown(f"**Current:** {newest['Return on equity']:.2f}%")
            st.markdown(f"**Change:** <span class='{'trend-up' if roe_change >= 0 else 'trend-down'}'>{roe_change:.2f}%</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ROCE Trend")
            st.markdown(f"**Current:** {newest['Return on capital employed']:.2f}%")
            st.markdown(f"**Change:** <span class='{'trend-up' if roce_change >= 0 else 'trend-down'}'>{roce_change:.2f}%</span>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### FCF Trend")
            st.markdown(f"**Current:** ₹{newest['Free cash flow']:.2f} cr")
            st.markdown(f"**Change:** <span class='{'trend-up' if fcf_change >= 0 else 'trend-down'}'>{fcf_change:.2f}%</span>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("#### P/E Trend")
            st.markdown(f"**Current:** {newest['Price to Earning']:.2f}")
            st.markdown(f"**Change:** <span class='{'trend-down' if pe_change <= 0 else 'trend-up'}'>{pe_change:.2f}%</span>", unsafe_allow_html=True)
        
        # Show time series data table
        st.markdown("### Historical Data")
        st.dataframe(time_series_data)
        
        # Download time series data
        st.markdown(create_download_link(time_series_data, f"{company_name}_time_series.csv", "Download Time Series Data"), unsafe_allow_html=True)
    else:
        st.info("Select a company from the Rankings tab to view time series analysis.")

# Display macroeconomic indicators tab
def display_macro_tab(macro_data):
    """Display the macroeconomic indicators tab with economic data"""
    st.markdown('<h2 class="sub-header">Macroeconomic Indicators</h2>', unsafe_allow_html=True)
    st.markdown("Analysis of key economic indicators and their impact on IT sector")
    
    if macro_data:
        # Create dashboard of current indicators
        st.markdown("### Current Economic Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'GDP Growth' in macro_data:
                st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
                st.markdown('<div class="dashboard-metric-title">GDP Growth Rate</div>', unsafe_allow_html=True)
                value = macro_data['GDP Growth']['value']
                trend = macro_data['GDP Growth']['trend']
                trend_class = 'trend-up' if trend == 'Positive' else ('trend-down' if trend == 'Negative' else '')
                st.markdown(f'<div class="dashboard-metric-value {trend_class}">{value:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="small-text">Impact on IT Sector: {macro_data["GDP Growth"]["sector_impact"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if 'Inflation' in macro_data:
                st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
                st.markdown('<div class="dashboard-metric-title">Inflation Rate</div>', unsafe_allow_html=True)
                value = macro_data['Inflation']['value']
                trend = macro_data['Inflation']['trend']
                trend_class = 'trend-down' if trend == 'Negative' else ('trend-up' if trend == 'Positive' else '')
                st.markdown(f'<div class="dashboard-metric-value {trend_class}">{value:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="small-text">Impact on IT Sector: {macro_data["Inflation"]["sector_impact"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            if 'Interest Rate' in macro_data:
                st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
                st.markdown('<div class="dashboard-metric-title">Interest Rate</div>', unsafe_allow_html=True)
                value = macro_data['Interest Rate']['value']
                trend = macro_data['Interest Rate']['trend']
                trend_class = 'trend-down' if trend == 'Negative' else ('trend-up' if trend == 'Positive' else '')
                st.markdown(f'<div class="dashboard-metric-value {trend_class}">{value:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="small-text">Impact on IT Sector: {macro_data["Interest Rate"]["sector_impact"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            if 'Unemployment' in macro_data:
                st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
                st.markdown('<div class="dashboard-metric-title">Unemployment Rate</div>', unsafe_allow_html=True)
                value = macro_data['Unemployment']['value']
                trend = macro_data['Unemployment']['trend']
                trend_class = 'trend-down' if trend == 'Negative' else ('trend-up' if trend == 'Positive' else '')
                st.markdown(f'<div class="dashboard-metric-value {trend_class}">{value:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="small-text">Impact on IT Sector: {macro_data["Unemployment"]["sector_impact"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Create charts for macroeconomic data
        st.markdown("### Economic Indicators Over Time")
        macro_charts = create_macro_charts(macro_data)
        if macro_charts:
            st.plotly_chart(macro_charts, use_container_width=True)
        
        # Economic analysis
        st.markdown("### Economic Outlook Analysis")
        
        # Calculate overall economic sentiment
        positive_indicators = sum(1 for key in macro_data if macro_data[key].get('sector_impact', '') == 'Positive')
        negative_indicators = sum(1 for key in macro_data if macro_data[key].get('sector_impact', '') == 'Negative')
        neutral_indicators = sum(1 for key in macro_data if macro_data[key].get('sector_impact', '') == 'Neutral' or macro_data[key].get('sector_impact', '') == 'Mixed')
        
        total_indicators = positive_indicators + negative_indicators + neutral_indicators
        if total_indicators > 0:
            economic_sentiment = "Positive" if positive_indicators > negative_indicators and positive_indicators > neutral_indicators else \
                              "Negative" if negative_indicators > positive_indicators and negative_indicators > neutral_indicators else \
                              "Neutral"
            
            sentiment_color = "positive-value" if economic_sentiment == "Positive" else \
                           "negative-value" if economic_sentiment == "Negative" else \
                           "neutral-value"
            
            st.markdown(f"#### Overall Economic Sentiment: <span class='{sentiment_color}'>{economic_sentiment}</span>", unsafe_allow_html=True)
            
            # Generate economic analysis based on indicators
            analysis = """
            Based on the current macroeconomic indicators, the IT sector outlook appears to be influenced by several key factors:
            
            1. **GDP Growth**: A positive GDP growth rate typically leads to increased IT spending by businesses and consumers. Companies often invest in digital transformation and IT modernization during growth periods.
            
            2. **Inflation Rate**: Higher inflation can pressure IT companies' margins due to increased costs but may also lead to higher pricing for services. IT services companies often have some pricing power to pass on inflation.
            
            3. **Interest Rates**: Higher interest rates can impact capital-intensive tech investments and startup funding, but most established IT services companies typically have low debt and are less affected than other sectors.
            
            4. **Unemployment Rate**: Lower unemployment rates may create talent acquisition challenges for IT companies but also indicate a healthy economy with spending power.
            
            The IT services sector tends to be resilient during moderate economic volatility due to the increasing digitalization trend across industries. Companies with strong cash positions, recurring revenue models, and exposure to growth areas like cloud, AI, and digital transformation are better positioned.
            """
            
            st.markdown(analysis)
    else:
        # Create a button to fetch macroeconomic data
        if st.button("Fetch Macroeconomic Indicators"):
            with st.spinner("Fetching macroeconomic data..."):
                st.session_state.macro_indicators = fetch_macroeconomic_indicators()
            st.experimental_rerun()
        else:
            st.info("Click the button above to fetch macroeconomic indicators.")

# Display sentiment analysis tab
def display_sentiment_tab(sentiment_data, company_name):
    """Display the sentiment analysis tab with news sentiment data"""
    st.markdown('<h2 class="sub-header">News Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    if sentiment_data and 'sentiment_df' in sentiment_data:
        st.markdown(f"### Sentiment Analysis for {company_name}")
        
        # Create sentiment dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Overall sentiment score
            overall_sentiment = sentiment_data['overall_sentiment']
            sentiment_class = "positive-value" if overall_sentiment > 0.2 else \
                           "negative-value" if overall_sentiment < -0.2 else \
                           "neutral-value"
            sentiment_label = "Positive" if overall_sentiment > 0.2 else \
                           "Negative" if overall_sentiment < -0.2 else \
                           "Neutral"
            
            st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
            st.markdown('<div class="dashboard-metric-title">Overall Sentiment</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="dashboard-metric-value {sentiment_class}">{sentiment_label} ({overall_sentiment:.2f})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Sentiment distribution
            sentiment_counts = sentiment_data['sentiment_count']
            total = sentiment_counts.sum()
            
            st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
            st.markdown('<div class="dashboard-metric-title">Sentiment Distribution</div>', unsafe_allow_html=True)
            
            # Create a mini bar chart using HTML/CSS
            positive_pct = sentiment_counts.get('Positive', 0) / total * 100 if total > 0 else 0
            neutral_pct = sentiment_counts.get('Neutral', 0) / total * 100 if total > 0 else 0
            negative_pct = sentiment_counts.get('Negative', 0) / total * 100 if total > 0 else 0
            
            st.markdown(f'''
            <div style="display: flex; height: 20px; width: 100%; margin-bottom: 5px;">
                <div style="background-color: #4CAF50; width: {positive_pct}%;"></div>
                <div style="background-color: #FFC107; width: {neutral_pct}%;"></div>
                <div style="background-color: #F44336; width: {negative_pct}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <div><span class="positive-value">Positive: {positive_pct:.1f}%</span></div>
                <div><span class="neutral-value">Neutral: {neutral_pct:.1f}%</span></div>
                <div><span class="negative-value">Negative: {negative_pct:.1f}%</span></div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            # News volume
            news_count = len(sentiment_data['sentiment_df'])
            
            st.markdown('<div class="dashboard-metric">', unsafe_allow_html=True)
            st.markdown('<div class="dashboard-metric-title">News Volume</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="dashboard-metric-value">{news_count} articles</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create sentiment charts
        sentiment_charts = create_sentiment_charts(sentiment_data)
        if sentiment_charts:
            st.plotly_chart(sentiment_charts, use_container_width=True)
        
        # Display sentiment data table
        st.markdown("### Recent News Articles")
        
        # Format the sentiment dataframe for display
        sentiment_df = sentiment_data['sentiment_df'].copy()
        # Sort by date (newest first)
        sentiment_df = sentiment_df.sort_values('Date', ascending=False)
        
        # Display as an interactive table
        for i, row in sentiment_df.iterrows():
            with st.expander(f"{row['Date']} - {row['Title']}"):
                # Apply color based on sentiment
                sentiment_class = "positive-value" if row['Sentiment'] == 'Positive' else \
                               "negative-value" if row['Sentiment'] == 'Negative' else \
                               "neutral-value"
                
                st.markdown(f"**Source:** {row['Source']}")
st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{row['Sentiment']}</span> (Polarity: {row['Polarity']:.2f}, Subjectivity: {row['Subjectivity']:.2f})", unsafe_allow_html=True)
                
                if 'URL' in row and row['URL']:
                    st.markdown(f"[Read full article]({row['URL']})")
        
        # Download sentiment data
        st.markdown("### Download Sentiment Data")
        st.markdown(create_download_link(sentiment_df, f"{company_name}_sentiment.csv", "Download Sentiment Data CSV"), unsafe_allow_html=True)
    else:
        # Create a button to fetch sentiment data
        if st.button("Analyze News Sentiment"):
            company_name = st.session_state.get('selected_company', 'Company')
            company_code = "NSE:CODE"  # Default code if not available
            
            with st.spinner(f"Analyzing sentiment for {company_name}..."):
                st.session_state.sentiment_data = fetch_and_analyze_company_sentiment(company_name, company_code)
            
            st.experimental_rerun()
        else:
            st.info("Select a company from the Rankings tab and click 'Analyze News Sentiment' to view sentiment analysis.")

# Display company comparison tab
def display_comparison_tab(ranked_df, selected_companies):
    """Display the company comparison tab with side-by-side comparison"""
    st.markdown('<h2 class="sub-header">Company Comparison</h2>', unsafe_allow_html=True)
    
    if ranked_df is not None and len(ranked_df) > 0:
        # Allow user to select companies to compare if not already selected
        if not selected_companies:
            selected_companies = st.multiselect(
                "Select companies to compare:",
                options=ranked_df['Name'].tolist(),
                default=ranked_df['Name'].head(3).tolist()
            )
        
        if selected_companies and len(selected_companies) >= 2:
            # Allow user to select metrics to compare
            st.markdown("### Select Metrics for Comparison")
            
            # Default comparison metrics
            default_metrics = [
                'Return on equity', 'Return on capital employed', 
                'Free cash flow last year', 'Price to Earning', 
                'Debt to equity', 'Profit growth 5Years',
                'Buffett Score', 'Lynch Score', 'Graham Score'
            ]
            
            # Select metrics to compare
            comparison_metrics = st.multiselect(
                "Select metrics to compare:",
                options=[col for col in ranked_df.columns if col not in ['Name', 'NSE Code', 'Ranking Reason']],
                default=default_metrics
            )
            
            if comparison_metrics:
                # Filter data for selected companies and metrics
                comparison_data = ranked_df[ranked_df['Name'].isin(selected_companies)].reset_index(drop=True)
                
                # Create comparison chart
                fig = create_side_by_side_comparison(comparison_data, selected_companies, comparison_metrics)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create a comparison table
                st.markdown("### Detailed Metrics Comparison")
                
                # Create a comparison table with color coding
                comparison_table = pd.DataFrame({'Metric': comparison_metrics})
                
                for company in selected_companies:
                    if company in comparison_data['Name'].values:
                        company_row = comparison_data[comparison_data['Name'] == company].iloc[0]
                        comparison_table[company] = [company_row[metric] for metric in comparison_metrics]
                
                # Format the table as HTML with color coding
                html_table = '<table class="compare-table" style="width:100%"><tr><th>Metric</th>'
                
                # Add company headers
                for company in selected_companies:
                    html_table += f'<th>{company}</th>'
                html_table += '</tr>'
                
                # Add rows for each metric
                for i, row in comparison_table.iterrows():
                    html_table += f'<tr><td>{row["Metric"]}</td>'
                    
                    # Determine the best value for this metric
                    metric = row['Metric']
                    values = [row[company] for company in selected_companies if company in row.index]
                    
                    if len(values) > 0:
                        # Determine if higher or lower is better for this metric
                        lower_is_better = metric in ['Price to Earning', 'Debt to equity', 'PEG Ratio']
                        
                        best_value = min(values) if lower_is_better else max(values)
                        
                        # Add cells for each company
                        for company in selected_companies:
                            if company in row.index:
                                value = row[company]
                                
                                # Format based on the metric type
                                if isinstance(value, (int, float)):
                                    if metric.endswith('Score') or metric.endswith('score'):
                                        formatted_value = f"{value:.2f}"
                                    elif "ratio" in metric.lower() or metric in ['Debt to equity', 'Price to Earning', 'PEG Ratio']:
                                        formatted_value = f"{value:.2f}"
                                    elif "growth" in metric.lower() or metric in ['Return on equity', 'Return on capital employed', 'Dividend yield', 'OPM']:
                                        formatted_value = f"{value:.2f}%"
                                    elif metric == 'Free cash flow last year' or metric == 'Market Capitalization':
                                        formatted_value = f"₹{value:.2f} cr"
                                    else:
                                        formatted_value = f"{value:.2f}"
                                else:
                                    formatted_value = str(value)
                                
                                # Highlight the best value
                                if value == best_value:
                                    html_table += f'<td style="background-color:#e6f7e6; font-weight:bold;">{formatted_value}</td>'
                                else:
                                    html_table += f'<td>{formatted_value}</td>'
                            else:
                                html_table += '<td>N/A</td>'
                    
                    html_table += '</tr>'
                
                html_table += '</table>'
                
                # Display the HTML table
                st.markdown(html_table, unsafe_allow_html=True)
                
                # Add a download option for the comparison
                comparison_csv = comparison_table.to_csv(index=False)
                b64 = base64.b64encode(comparison_csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="company_comparison.csv">Download Comparison CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Option to generate PDF report
                if st.button("Generate PDF Comparison Report"):
                    st.info("Generating PDF report... This feature would create a downloadable PDF report comparing the selected companies.")
                    # In a real application, this would generate a PDF using a library like ReportLab
                    
                    # For demo purposes, just show a sample download link
                    sample_pdf = f'<a href="#" class="pdf-download-btn">Download Comparison Report PDF</a>'
                    st.markdown(sample_pdf, unsafe_allow_html=True)
            else:
                st.warning("Please select at least one metric to compare.")
        else:
            st.warning("Please select at least two companies to compare.")
    else:
        st.info("Upload data or use sample data to perform company comparisons.")

# Main application function
def main():
    """Main application function"""
    # Load custom CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    st.markdown('<h1 class="main-header">Advanced IT Companies Financial Model</h1>', unsafe_allow_html=True)
    
    # Set up sidebar controls
    uploaded_file, use_sample_data, weights, philosophy_weights, refresh_ttl = setup_sidebar()
    
    # Process data if file is uploaded or sample data is selected
    process_data = False
    
    if uploaded_file is not None:
        try:
            # Load data from uploaded file
            df = pd.read_csv(uploaded_file)
            process_data = True
        except Exception as e:
            st.error(f"Error reading the uploaded file: {str(e)}")
    elif use_sample_data:
        # Use sample data
        df = generate_sample_it_data()
        process_data = True
    elif 'data' in st.session_state and st.session_state.data is not None:
        # Use data from session state
        df = st.session_state.data
        process_data = True
    
    if process_data:
        try:
            # Store data in session state
            st.session_state.data = df
            
            # Validate and process data
            ranked_df = validate_and_process_data(df, weights, philosophy_weights)
            
            # Store ranked data in session state
            st.session_state.ranked_df = ranked_df
            
            # Create main tabs
            main_tabs = st.tabs([
                "Rankings", 
                "Visualizations", 
                "Investment Philosophies", 
                "Data Explorer",
                "Time Series Analysis",
                "Macroeconomic Indicators",
                "Sentiment Analysis",
                "Company Comparison"
            ])
            
            # Display tabs
            with main_tabs[0]:
                display_rankings_tab(ranked_df)
            
            with main_tabs[1]:
                display_visualizations_tab(ranked_df)
            
            with main_tabs[2]:
                display_investment_tab(ranked_df)
            
            with main_tabs[3]:
                display_data_explorer_tab(ranked_df)
            
            with main_tabs[4]:
                time_series_data = st.session_state.get('time_series_data')
                selected_company = st.session_state.get('selected_company', '')
                display_time_series_tab(time_series_data, selected_company)
            
            with main_tabs[5]:
                macro_indicators = st.session_state.get('macro_indicators')
                display_macro_tab(macro_indicators)
            
            with main_tabs[6]:
                sentiment_data = st.session_state.get('sentiment_data')
                selected_company = st.session_state.get('selected_company', '')
                display_sentiment_tab(sentiment_data, selected_company)
            
            with main_tabs[7]:
                selected_companies = st.session_state.get('selected_companies', [])
                display_comparison_tab(ranked_df, selected_companies)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)
    else:
        # Show sample data and instructions
        show_sample_data()

# Run the main application
if __name__ == "__main__":
    main()
        
