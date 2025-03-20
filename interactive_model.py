import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os

# Set page configuration
st.set_page_config(
    page_title="Companies Financial Model",
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

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    st.warning("Visualization libraries (matplotlib/seaborn) are not available. Some charts will be disabled.")

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

# Visualization functions - only used if matplotlib is available
if VISUALIZATION_AVAILABLE:
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
        process_data(df, weights, buffett_weight, lynch_weight)
    else:
        # Display sample data and instructions when no file is uploaded
        st.info("Please upload a CSV file with company financial data to begin analysis.")
        
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
        
        # Use the original dataset as an example
        st.markdown("### Try with Sample Dataset")
        if st.button("Load Sample IT Companies Dataset"):
            # Create a simple sample dataset
            sample_it_data = pd.DataFrame({
                'Name': ['TCS', 'Infosys', 'HCL Tech', 'Wipro', 'Tech Mahindra'],
                'NSE Code': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
                'Return on equity': [51.51, 31.83, 23.30, 18.75, 15.42],
                'Return on capital employed': [64.28, 39.99, 29.60, 24.15, 19.87],
                'Free cash flow last year': [41688.00, 23009.00, 21432.00, 12500.00, 8750.00],
                'Price to Earning': [26.36, 24.17, 24.77, 28.50, 22.30],
                'Debt to equity': [0.09, 0.09, 0.08, 0.12, 0.15],
                'OPM': [25.50, 23.84, 21.75, 18.90, 16.50],
                'Profit growth 5Years': [10.46, 11.24, 12.71, 8.50, 7.20],
                'Sales growth 5Years': [10.46, 13.20, 12.71, 9.30, 8.50],
                'Dividend yield': [1.50, 1.80, 2.10, 1.20, 1.90],
                'Market Capitalization': [1250000, 750000, 450000, 350000, 120000],
                'PEG Ratio': [2.52, 2.15, 1.95, 3.35, 3.10]
            })
            process_data(sample_it_data, weights, buffett_weight, lynch_weight)

def process_data(df, weights, buffett_weight, lynch_weight):
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
                    st.markdown(f"**Sales Growth (5Y):** <span class='metric-value'>{row['Sales growth 5Years']:.2f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**Profit Growth (5Y):** <span class='metric-value'>{row['Profit growth 5Years']:.2f}%</span>", unsafe_allow_html=True)
                
                st.markdown(f'<div class="ranking-reason"><strong>Why Ranked #{row["Rank"]}:</strong> {row["Ranking Reason"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Download links for ranked data
        st.markdown("### Download Ranked Data")
        st.markdown(create_download_link(ranked_df, "ranked_companies.csv", "Download Full Rankings CSV"), unsafe_allow_html=True)
        st.markdown(create_download_link(ranked_df.head(10), "top10_companies.csv", "Download Top 10 Companies CSV"), unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Visualizations</h2>', unsafe_allow_html=True)
        
        if VISUALIZATION_AVAILABLE:
            # Top companies bar chart
            st.markdown('<h3 class="sub-header">Top Companies by Ranking Score</h3>', unsafe_allow_html=True)
            fig1 = plot_top_companies(ranked_df)
            st.pyplot(fig1)
            
            # ROE vs ROCE scatter plot
            st.markdown('<h3 class="sub-header">ROE vs ROCE Analysis</h3>', unsafe_allow_html=True)
            st.markdown("Bubble size represents market capitalization, color represents rank")
            fig2 = plot_roe_roce(ranked_df)
            st.pyplot(fig2)
            
            # Buffett vs Lynch scatter plot
            st.markdown('<h3 class="sub-header">Buffett vs Lynch Investment Philosophy Alignment</h3>', unsafe_allow_html=True)
            st.markdown("Bubble size represents market capitalization, color represents rank")
            fig3 = plot_buffett_lynch(ranked_df)
            st.pyplot(fig3)
        else:
            st.warning("Visualizations are not available due to missing dependencies. Please check your requirements.txt file includes matplotlib and seaborn.")
            
            # Display alternative text-based visualizations
            st.markdown("### Top 10 Companies by Ranking Score")
            top10_df = ranked_df.head(10)[['Name', 'Ranking Score', 'Rank']].sort_values('Ranking Score', ascending=False)
            st.dataframe(top10_df)
            
            st.markdown("### ROE vs ROCE Analysis")
            roe_roce_df = ranked_df.head(20)[['Name', 'Return on equity', 'Return on capital employed', 'Rank']].sort_values('Rank')
            st.dataframe(roe_roce_df)
            
            st.markdown("### Buffett vs Lynch Investment Philosophy Alignment")
            philosophy_df = ranked_df.head(20)[['Name', 'Buffett Score', 'Lynch Score', 'Rank']].sort_values('Rank')
            st.dataframe(philosophy_df)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Investment Philosophy Analysis</h2>', unsafe_allow_html=True)
        
        # Warren Buffett section
        st.markdown('<h3 class="sub-header">Warren Buffett Investment Philosophy</h3>', unsafe_allow_html=True)
        st.markdown("""
        Warren Buffett's investment approach focuses on identifying companies with strong fundamentals, 
        sustainable competitive advantages, and excellent management. His philosophy emphasizes long-term 
        value investing rather than short-term market fluctuations.
        """)
        
        st.markdown("#### Key Principles")
        st.markdown("""
        - **Consistent ROE**: Return on equity consistently above 15% indicates a company with a sustainable competitive advantage
        - **Low Debt**: Low debt-to-equity ratio (< 0.5) shows financial stability and reduced risk
        - **High Margins**: High operating profit margins (> 20%) indicate pricing power and competitive advantage
        - **Earnings Growth**: Consistent earnings growth over 5+ years shows business strength and management capability
        - **Economic Moat**: Companies with strong competitive advantages that protect market share and profitability
        """)
        
        if not buffett_companies.empty:
            st.markdown("#### Top Companies Matching Buffett's Criteria")
            st.dataframe(buffett_companies)
            st.markdown(create_download_link(buffett_companies, "buffett_companies.csv", "Download Buffett Companies CSV"), unsafe_allow_html=True)
        else:
            st.info("No companies strongly match Warren Buffett's investment criteria based on our analysis.")
        
        # Peter Lynch section
        st.markdown('<h3 class="sub-header">Peter Lynch Investment Philosophy</h3>', unsafe_allow_html=True)
        st.markdown("""
        Peter Lynch's investment strategy focuses on finding companies with good growth at reasonable prices. 
        He believes in investing in what you know and understand, and looks for companies with strong growth 
        potential that are undervalued by the market.
        """)
        
        st.markdown("#### Key Principles")
        st.markdown("""
        - **PEG Ratio**: Price/Earnings to Growth ratio < 1 is excellent, < 2 is good - shows growth at reasonable price
        - **Growth Potential**: Companies with strong growth potential in sales and earnings
        - **Reasonable P/E**: P/E ratio should be reasonable relative to growth rate
        - **Strong Balance Sheet**: Low debt and strong financial position
        - **Cash Flow**: Strong and consistent cash flow generation
        """)
        
        if not lynch_companies.empty:
            st.markdown("#### Top Companies Matching Lynch's Criteria")
            st.dataframe(lynch_companies)
            st.markdown(create_download_link(lynch_companies, "lynch_companies.csv", "Download Lynch Companies CSV"), unsafe_allow_html=True)
        else:
            st.info("No companies strongly match Peter Lynch's investment criteria based on our analysis.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">Data Explorer</h2>', unsafe_allow_html=True)
        st.markdown("Explore the complete dataset with all calculated metrics")
        
        # Column selector
        all_columns = ranked_df.columns.tolist()
        default_columns = ['Name', 'NSE Code', 'Rank', 'Ranking Score', 'Return on equity', 
                          'Return on capital employed', 'Free cash flow last year', 'Price to Earning',
                          'Buffett Score', 'Lynch Score']
        selected_columns = st.multiselect("Select columns to display", all_columns, default=default_columns)
        
        if selected_columns:
            st.dataframe(ranked_df[selected_columns])
        else:
            st.dataframe(ranked_df)
        
        # Download link for full data
        st.markdown(create_download_link(ranked_df, "full_data_with_metrics.csv", "Download Complete Dataset CSV"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

