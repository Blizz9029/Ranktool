import pandas as pd
import numpy as np

# Load the enhanced data with calculated metrics
df = pd.read_csv('/home/ubuntu/financial_model/companies_with_metrics.csv')

# Create a comprehensive ranking algorithm
# This will combine multiple factors with appropriate weights based on user preferences

# Define the weights for different metrics in the final ranking
ranking_weights = {
    # Core financial metrics (50%)
    'Return on equity': 0.15,              # User specified ROE > 15%
    'Return on capital employed': 0.15,    # User specified ROCE > 15%
    'Free cash flow last year': 0.10,      # User specified free cash flow generation
    'Price to Earning': -0.10,             # User specified lower PE (negative weight as lower is better)
    
    # Growth metrics (20%)
    'Sales growth 5Years': 0.05,
    'Profit growth 5Years': 0.10,
    'Profit growth 3Years': 0.05,
    
    # Balance sheet strength (15%)
    'Debt to equity': -0.05,               # Lower is better
    'Current ratio': 0.05,                 # Higher is better
    'Interest Coverage Ratio': 0.05,       # Higher is better
    
    # Valuation metrics (15%)
    'Price to book value': -0.05,          # Lower is better
    'Price to Sales': -0.05,               # Lower is better
    'Dividend yield': 0.05,                # Higher is better
}

# Function to normalize a series to 0-1 scale
def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

# Normalize all metrics for fair comparison
normalized_df = pd.DataFrame(index=df.index)
for metric, weight in ranking_weights.items():
    if metric in df.columns:
        # For metrics where higher is better (positive weight)
        if weight > 0:
            normalized_df[metric] = normalize_series(df[metric].fillna(df[metric].min()))
        # For metrics where lower is better (negative weight)
        else:
            normalized_df[metric] = 1 - normalize_series(df[metric].fillna(df[metric].max()))

# Calculate the weighted ranking score
df['Ranking Score'] = 0
for metric, weight in ranking_weights.items():
    if metric in normalized_df.columns:
        df['Ranking Score'] += normalized_df[metric] * abs(weight)

# Add investment philosophy scores to the ranking
# Buffett and Lynch scores are already on a 0-100 scale
df['Ranking Score'] += (df['Buffett Score'].fillna(0) / 100) * 0.15  # 15% weight to Buffett score
df['Ranking Score'] += (df['Lynch Score'].fillna(0) / 100) * 0.15    # 15% weight to Lynch score

# Rank the companies based on the final score
df['Rank'] = df['Ranking Score'].rank(ascending=False, method='min').astype(int)

# Sort the dataframe by rank
ranked_df = df.sort_values('Rank')

# Create a summary of the top companies with reasons for their ranking
top_companies = ranked_df.head(10).copy()

# Function to generate ranking reasons
def generate_ranking_reason(row):
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
    if row['Price to Earning'] < ranked_df['Price to Earning'].median():
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

# Generate ranking reasons for top companies
top_companies['Ranking Reason'] = top_companies.apply(generate_ranking_reason, axis=1)

# Save the ranked companies to CSV
ranked_df.to_csv('/home/ubuntu/financial_model/ranked_companies.csv', index=False)

# Save the top companies with reasons to a separate CSV
top_columns = ['Name', 'NSE Code', 'Rank', 'Ranking Score', 'Return on equity', 
               'Return on capital employed', 'Free cash flow last year', 'Price to Earning',
               'Sales growth 5Years', 'Profit growth 5Years', 'Debt to equity',
               'Buffett Score', 'Lynch Score', 'Ranking Reason']
top_companies[top_columns].to_csv('/home/ubuntu/financial_model/top_companies.csv', index=False)

# Create lists of companies that match Buffett and Lynch criteria
buffett_companies = ranked_df[ranked_df['Buffett Score'] > 75][['Name', 'NSE Code', 'Buffett Score', 'Rank']].sort_values('Buffett Score', ascending=False)
lynch_companies = ranked_df[ranked_df['Lynch Score'] > 50][['Name', 'NSE Code', 'Lynch Score', 'Rank']].sort_values('Lynch Score', ascending=False)

# Save these lists to CSV
buffett_companies.to_csv('/home/ubuntu/financial_model/buffett_companies.csv', index=False)
lynch_companies.to_csv('/home/ubuntu/financial_model/lynch_companies.csv', index=False)

print("Ranking algorithm created and applied successfully.")
print(f"Total companies ranked: {len(ranked_df)}")
print(f"Top 10 companies saved to: /home/ubuntu/financial_model/top_companies.csv")
print(f"All ranked companies saved to: /home/ubuntu/financial_model/ranked_companies.csv")
print(f"Companies matching Buffett criteria: {len(buffett_companies)}")
print(f"Companies matching Lynch criteria: {len(lynch_companies)}")
