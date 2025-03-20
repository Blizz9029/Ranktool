import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the ranked companies data
ranked_df = pd.read_csv('/home/ubuntu/financial_model/ranked_companies.csv')
top_companies = pd.read_csv('/home/ubuntu/financial_model/top_companies.csv')
buffett_companies = pd.read_csv('/home/ubuntu/financial_model/buffett_companies.csv')
lynch_companies = pd.read_csv('/home/ubuntu/financial_model/lynch_companies.csv')

# Create a directory for visualizations
os.makedirs('/home/ubuntu/financial_model/visualizations', exist_ok=True)

# 1. Create a detailed analysis of the top 10 companies
print("Analyzing Top 10 Companies...")
top10_analysis = top_companies.head(10).copy()

# Format the output for better readability
analysis_columns = ['Name', 'NSE Code', 'Rank', 'Return on equity', 'Return on capital employed', 
                    'Free cash flow last year', 'Price to Earning', 'Sales growth 5Years', 
                    'Profit growth 5Years', 'Debt to equity', 'Buffett Score', 'Lynch Score']

# Save detailed analysis to CSV
top10_analysis[analysis_columns].to_csv('/home/ubuntu/financial_model/visualizations/top10_detailed_analysis.csv', index=False)

# 2. Create visualizations for the top companies

# Set the style for the plots
sns.set(style="whitegrid")

# Plot ROE vs ROCE for top 20 companies
plt.figure(figsize=(12, 8))
top20 = ranked_df.sort_values('Rank').head(20)
sns.scatterplot(data=top20, x='Return on equity', y='Return on capital employed', 
                size='Market Capitalization', sizes=(100, 1000), 
                hue='Rank', palette='viridis', alpha=0.7)

# Add company names as labels
for i, row in top20.iterrows():
    plt.text(row['Return on equity'], row['Return on capital employed'], 
             row['Name'], fontsize=9)

plt.title('ROE vs ROCE for Top 20 Companies', fontsize=16)
plt.xlabel('Return on Equity (%)', fontsize=12)
plt.ylabel('Return on Capital Employed (%)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ubuntu/financial_model/visualizations/roe_vs_roce.png')
plt.close()

# Plot Free Cash Flow vs P/E Ratio
plt.figure(figsize=(12, 8))
sns.scatterplot(data=top20, x='Price to Earning', y='Free cash flow last year', 
                size='Market Capitalization', sizes=(100, 1000), 
                hue='Rank', palette='viridis', alpha=0.7)

# Add company names as labels
for i, row in top20.iterrows():
    plt.text(row['Price to Earning'], row['Free cash flow last year'], 
             row['Name'], fontsize=9)

plt.title('P/E Ratio vs Free Cash Flow for Top 20 Companies', fontsize=16)
plt.xlabel('Price to Earnings Ratio', fontsize=12)
plt.ylabel('Free Cash Flow (Last Year in Crores)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ubuntu/financial_model/visualizations/pe_vs_fcf.png')
plt.close()

# Plot Buffett Score vs Lynch Score
plt.figure(figsize=(12, 8))
sns.scatterplot(data=top20, x='Buffett Score', y='Lynch Score', 
                size='Market Capitalization', sizes=(100, 1000), 
                hue='Rank', palette='viridis', alpha=0.7)

# Add company names as labels
for i, row in top20.iterrows():
    plt.text(row['Buffett Score'], row['Lynch Score'], 
             row['Name'], fontsize=9)

plt.title('Buffett Score vs Lynch Score for Top 20 Companies', fontsize=16)
plt.xlabel('Buffett Score', fontsize=12)
plt.ylabel('Lynch Score', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ubuntu/financial_model/visualizations/buffett_vs_lynch.png')
plt.close()

# 3. Create a bar chart of the top 10 companies by ranking score
plt.figure(figsize=(14, 8))
top10 = ranked_df.sort_values('Rank').head(10)
ax = sns.barplot(x='Name', y='Ranking Score', data=top10, hue='Name', legend=False)
plt.title('Top 10 Companies by Ranking Score', fontsize=16)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Ranking Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add the rank on top of each bar
for i, row in enumerate(top10.iterrows()):
    idx, data = row
    ax.text(i, data['Ranking Score'] + 0.01, f"Rank: {data['Rank']}", 
            ha='center', va='bottom', fontsize=10)

plt.savefig('/home/ubuntu/financial_model/visualizations/top10_ranking.png')
plt.close()

# 4. Create a detailed report on companies matching Buffett and Lynch criteria
print("Analyzing companies matching investment philosophies...")

# Prepare Buffett analysis
buffett_analysis = pd.DataFrame()
if not buffett_companies.empty:
    buffett_analysis = ranked_df[ranked_df['NSE Code'].isin(buffett_companies['NSE Code'])].copy()
    buffett_analysis = buffett_analysis[['Name', 'NSE Code', 'Rank', 'Buffett Score', 'Return on equity', 
                                        'Return on capital employed', 'Debt to equity', 'OPM', 
                                        'Profit growth 5Years', 'Price to Earning']]
    buffett_analysis.to_csv('/home/ubuntu/financial_model/visualizations/buffett_companies_analysis.csv', index=False)
    
    # Create visualization for Buffett companies
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Name', y='Buffett Score', data=buffett_analysis.sort_values('Buffett Score', ascending=False).head(10), 
                hue='Name', legend=False)
    plt.title('Top Companies by Buffett Score', fontsize=16)
    plt.xlabel('Company', fontsize=12)
    plt.ylabel('Buffett Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/financial_model/visualizations/buffett_companies.png')
    plt.close()

# Prepare Lynch analysis
lynch_analysis = pd.DataFrame()
if not lynch_companies.empty:
    lynch_analysis = ranked_df[ranked_df['NSE Code'].isin(lynch_companies['NSE Code'])].copy()
    lynch_analysis = lynch_analysis[['Name', 'NSE Code', 'Rank', 'Lynch Score', 'PEG Ratio', 
                                    'Sales growth 5Years', 'Price to Earning', 'Return on equity']]
    lynch_analysis.to_csv('/home/ubuntu/financial_model/visualizations/lynch_companies_analysis.csv', index=False)
    
    # Create visualization for Lynch companies if there are enough
    if len(lynch_analysis) > 0:
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Name', y='Lynch Score', data=lynch_analysis.sort_values('Lynch Score', ascending=False), 
                    hue='Name', legend=False)
        plt.title('Companies by Lynch Score', fontsize=16)
        plt.xlabel('Company', fontsize=12)
        plt.ylabel('Lynch Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('/home/ubuntu/financial_model/visualizations/lynch_companies.png')
        plt.close()

# 5. Create a summary report
with open('/home/ubuntu/financial_model/visualizations/ranking_summary.txt', 'w') as f:
    f.write("# IT Companies Financial Analysis Summary\n\n")
    
    f.write("## Top 10 Companies Overview\n\n")
    for i, row in top10.iterrows():
        f.write(f"{row['Rank']}. **{row['Name']}** (NSE: {row['NSE Code']})\n")
        f.write(f"   - Ranking Score: {row['Ranking Score']:.4f}\n")
        f.write(f"   - ROE: {row['Return on equity']:.2f}%, ROCE: {row['Return on capital employed']:.2f}%\n")
        f.write(f"   - Free Cash Flow: ₹{row['Free cash flow last year']:.2f} crores\n")
        f.write(f"   - P/E Ratio: {row['Price to Earning']:.2f}\n")
        f.write(f"   - Buffett Score: {row['Buffett Score']:.2f}, Lynch Score: {row['Lynch Score']:.2f}\n")
        
        # Generate ranking reason dynamically instead of using the column
        reasons = []
        if row['Return on equity'] > 15:
            reasons.append(f"Strong ROE of {row['Return on equity']:.2f}%")
        if row['Return on capital employed'] > 15:
            reasons.append(f"Excellent ROCE of {row['Return on capital employed']:.2f}%")
        if row['Free cash flow last year'] > 0:
            reasons.append(f"Positive free cash flow of ₹{row['Free cash flow last year']:.2f} crores")
        if row['Price to Earning'] < ranked_df['Price to Earning'].median():
            reasons.append(f"Attractive P/E ratio of {row['Price to Earning']:.2f}")
        if row['Sales growth 5Years'] > 10:
            reasons.append(f"Strong 5-year sales growth of {row['Sales growth 5Years']:.2f}%")
        if row['Profit growth 5Years'] > 10:
            reasons.append(f"Impressive 5-year profit growth of {row['Profit growth 5Years']:.2f}%")
        if row['Debt to equity'] < 0.5:
            reasons.append(f"Low debt-to-equity ratio of {row['Debt to equity']:.2f}")
        if row['Buffett Score'] > 75:
            reasons.append("Aligns well with Warren Buffett's investment principles")
        if row['Lynch Score'] > 50:
            reasons.append("Matches Peter Lynch's growth at reasonable price criteria")
        
        reason_text = ". ".join(reasons) + "." if reasons else "Good overall financial performance across multiple metrics."
        f.write(f"   - **Why Ranked #{row['Rank']}**: {reason_text}\n\n")
    
    f.write("\n## Warren Buffett Investment Philosophy Matches\n\n")
    f.write("Warren Buffett's investment philosophy focuses on companies with consistent returns, low debt, strong profit margins, and sustainable competitive advantages.\n\n")
    if not buffett_companies.empty:
        for i, row in buffett_companies.head(5).iterrows():
            company_data = ranked_df[ranked_df['NSE Code'] == row['NSE Code']].iloc[0]
            f.write(f"- **{row['Name']}** (Buffett Score: {row['Buffett Score']:.2f})\n")
            f.write(f"  - ROE: {company_data['Return on equity']:.2f}%, ROCE: {company_data['Return on capital employed']:.2f}%\n")
            f.write(f"  - Debt to Equity: {company_data['Debt to equity']:.2f}\n")
            f.write(f"  - Operating Profit Margin: {company_data['OPM']:.2f}%\n")
            f.write(f"  - 5-Year Profit Growth: {company_data['Profit growth 5Years']:.2f}%\n\n")
    else:
        f.write("No companies strongly match Warren Buffett's investment criteria.\n\n")
    
    f.write("\n## Peter Lynch Investment Philosophy Matches\n\n")
    f.write("Peter Lynch's investment philosophy focuses on companies with reasonable P/E ratios relative to growth (PEG ratio), strong growth potential, and understandable business models.\n\n")
    if not lynch_companies.empty:
        for i, row in lynch_companies.head(5).iterrows():
            company_data = ranked_df[ranked_df['NSE Code'] == row['NSE Code']].iloc[0]
            f.write(f"- **{row['Name']}** (Lynch Score: {row['Lynch Score']:.2f})\n")
            f.write(f"  - PEG Ratio: {company_data['PEG Ratio']:.2f}\n")
            f.write(f"  - 5-Year Sales Growth: {company_data['Sales growth 5Years']:.2f}%\n")
            f.write(f"  - P/E Ratio: {company_data['Price to Earning']:.2f}\n")
            f.write(f"  - ROE: {company_data['Return on equity']:.2f}%\n\n")
    else:
        f.write("No companies strongly match Peter Lynch's investment criteria.\n\n")
    
    f.write("\n## Methodology\n\n")
    f.write("The ranking algorithm uses a weighted approach considering the following factors:\n\n")
    f.write("1. **Core Financial Metrics (50%)**: ROE, ROCE, Free Cash Flow, P/E Ratio\n")
    f.write("2. **Growth Metrics (20%)**: Sales Growth, Profit Growth\n")
    f.write("3. **Balance Sheet Strength (15%)**: Debt-to-Equity, Current Ratio, Interest Coverage\n")
    f.write("4. **Valuation Metrics (15%)**: P/B Ratio, P/S Ratio, Dividend Yield\n")
    f.write("5. **Investment Philosophy Alignment**: Additional weight given to companies matching Buffett and Lynch criteria\n\n")
    
    f.write("All metrics were normalized to ensure fair comparison across different scales, and the final ranking score represents a comprehensive evaluation of each company's financial health and investment potential.\n")

print("Analysis complete. Results saved to /home/ubuntu/financial_model/visualizations/")
