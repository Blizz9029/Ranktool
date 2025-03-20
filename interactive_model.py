import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import json

# Set page configuration
st.set_page_config(
    page_title="IT Companies Financial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1E88E5;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
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
</style>
""", unsafe_allow_html=True)

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
    
    # Calculate Buffett and Lynch scores
    df = calculate_buffett_score(df)
    df = calculate_lynch_score(df)
    
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

def create_download_link(df, filename, link_text):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def plot_top_companies(df, n=10):
    """Plot the top N companies by ranking score"""
    top_n = df.head(n)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x='Name', y='Ranking Score', data=top_n, ax=ax, hue='Name', legend=False)
    plt.title(f'Top {n} Companies by Ranking Score', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add the rank on top of each bar
    for i, row in enumerate(top_n.iterrows()):
        idx, data = row
        ax.text(i, data['Ranking Score'] + 0.01, f"Rank: {data['Rank']}", 
                ha='center', va='bottom', fontsize=10)
    
    return fig

def plot_roe_roce(df, n=20):
    """Plot ROE vs ROCE for top N companies"""
    top_n = df.head(n)
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        top_n['Return on equity'], 
        top_n['Return on capital employed'],
        s=top_n['Market Capitalization'] / top_n['Market Capitalization'].max() * 500,
        c=top_n['Rank'], 
        cmap='viridis', 
        alpha=0.7
    )
    
    # Add company names as labels
    for i, row in top_n.iterrows():
        ax.text(row['Return on equity'], row['Return on capital employed'], row['Name'], fontsize=9)
    
    plt.colorbar(scatter, label='Rank')
    plt.title(f'ROE vs ROCE for Top {n} Companies', fontsize=16)
    plt.xlabel('Return on Equity (%)')
    plt.ylabel('Return on Capital Employed (%)')
    plt.tight_layout()
    return fig

def plot_buffett_lynch(df, n=20):
    """Plot Buffett Score vs Lynch Score for top N companies"""
    top_n = df.head(n)
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        top_n['Buffett Score'], 
        top_n['Lynch Score'],
        s=top_n['Market Capitalization'] / top_n['Market Capitalization'].max() * 500,
        c=top_n['Rank'], 
        cmap='viridis', 
        alpha=0.7
    )
    
    # Add company names as labels
    for i, row in top_n.iterrows():
        ax.text(row['Buffett Score'], row['Lynch Score'], row['Name'], fontsize=9)
    
    plt.colorbar(scatter, label='Rank')
    plt.title(f'Buffett Score vs Lynch Score for Top {n} Companies', fontsize=16)
    plt.xlabel('Buffett Score')
    plt.ylabel('Lynch Score')
    plt.tight_layout()
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">IT Companies Financial Model</h1>', unsafe_allow_html=True)
    st.markdown("""
    This tool analyzes IT companies based on financial metrics and ranks them according to 
    investment criteria including ROE, ROCE, free cash flow, PE ratio, and principles from 
    Warren Buffett and Peter Lynch. Upload your company data to get started!
    """)
    
    # Sidebar for file upload and parameters
    st.sidebar.header("Upload Data & Set Parameters")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with company data", type=["csv"])
    
    # Weight sliders
    st.sidebar.subheader("Adjust Ranking Weights")
    st.sidebar.markdown("Set the importance of each factor in the ranking (total: 100%)")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        roe_weight = st.slider("Return on Equity", 0.0, 0.5, 0.20, 0.05)
        roce_weight = st.slider("Return on Capital Employed", 0.0, 0.5, 0.20, 0.05)
        fcf_weight = st.slider("Free Cash Flow", 0.0, 0.5, 0.15, 0.05)
        pe_weight = st.slider("Price to Earnings", 0.0, 0.5, 0.15, 0.05)
    
    with col2:
        debt_weight = st.slider("Debt to Equity", 0.0, 0.5, 0.10, 0.05)
        profit_growth_weight = st.slider("Profit Growth", 0.0, 0.5, 0.10, 0.05)
        sales_growth_weight = st.slider("Sales Growth", 0.0, 0.5, 0.05, 0.05)
        dividend_weight = st.slider("Dividend Yield", 0.0, 0.5, 0.05, 0.05)
    
    # Calculate total weight
    total_weight = roe_weight + roce_weight + fcf_weight + pe_weight + debt_weight + profit_growth_weight + sales_growth_weight + dividend_weight
    st.sidebar.markdown(f"Total weight: {total_weight:.2f}")
    
    # Normalize weights if total is not 1.0
    if total_weight != 1.0:
        roe_weight /= total_weight
        roce_weight /= total_weight
        fcf_weight /= total_weight
        pe_weight /= total_weight
        debt_weight /= total_weight
        profit_growth_weight /= total_weight
        sales_growth_weight /= total_weight
        dividend_weight /= total_weight
    
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
    
    # Investment philosophy weights
    st.sidebar.subheader("Investment Philosophy Weights")
    buffett_weight = st.sidebar.slider("Warren Buffett Criteria", 0.0, 0.5, 0.15, 0.05)
    lynch_weight = st.sidebar.slider("Peter Lynch Criteria", 0.0, 0.5, 0.15, 0.05)
    
    # Process data if file is uploaded
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = [
            'Name', 'NSE Code', 'Return on equity', 'Return on capital employed', 
            'Free cash flow last year', 'Price to Earning', 'Debt to equity', 
            'OPM', 'Profit growth 5Years', 'Sales growth 5Years', 'Dividend yield',
            'Market Capitalization', 'PEG Ratio'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_columns)}")
            st.stop()
        
        # Calculate financial metrics
        with st.spinner("Calculating financial metrics..."):
            df = calculate_financial_metrics(df)
        
        # Rank companies
        with st.spinner("Ranking companies..."):
            ranked_df = rank_companies(df, weights)
            
            # Add Buffett and Lynch weights
            ranked_df['Ranking Score'] = ranked_df['Ranking Score'] * (1 - buffett_weight - lynch_weight) + \
                                        (ranked_df['Buffett Score'] / 100) * buffett_weight + \
                                        (ranked_df['Lynch Score'] / 100) * lynch_weight
            
            # Re-rank based on updated scores
            ranked_df['Rank'] = ranked_df['Ranking Score'].rank(ascending=False, method='min').astype(int)
            ranked_df = ranked_df.sort_values('Rank')
        
        # Generate ranking reasons
        with st.spinner("Generating ranking explanations..."):
            ranked_df['Ranking Reason'] = ranked_df.apply(lambda row: generate_ranking_reason(row, ranked_df), axis=1)
        
        # Get companies matching investment philosophies
        buffett_companies = get_buffett_companies(ranked_df)
        lynch_companies = get_lynch_companies(ranked_df)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Rankings", "Visualizations", "Investment Philosophies", "Data Explorer"])
        
        with tab1:
            st.markdown('<h2 class="sub-header">Company Rankings</h2>', unsafe_allow_html=True)
            st.markdown("Companies ranked based on financial metrics and investment criteria")
            
            # Display top 10 companies with details
            st.markdown('<h3 class="sub-header">Top 10 Companies</h3>', unsafe_allow_html=True)
            for i, (_, row) in enumerate(ranked_df.head(10).iterrows()):
                with st.expander(f"{i+1}. {row['Name']} (NSE: {row['NSE Code']})"):
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
                    st.markdown(f'<span class="company-name">{row["Name"]}</span> <span class="company-code">({row["NSE Code"]})</span>', u<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>