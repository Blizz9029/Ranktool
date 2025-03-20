import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the ranked companies data
ranked_df = pd.read_csv('/home/ubuntu/financial_model/ranked_companies.csv')
buffett_companies = pd.read_csv('/home/ubuntu/financial_model/buffett_companies.csv')
lynch_companies = pd.read_csv('/home/ubuntu/financial_model/lynch_companies.csv')

# Create a directory for investment philosophy analysis
os.makedirs('/home/ubuntu/financial_model/investment_philosophy', exist_ok=True)

# Define Warren Buffett's investment principles in more detail
buffett_principles = {
    'Consistent ROE': 'Return on equity consistently above 15% indicates a company with a sustainable competitive advantage',
    'Low Debt': 'Low debt-to-equity ratio (< 0.5) shows financial stability and reduced risk',
    'High Margins': 'High operating profit margins (> 20%) indicate pricing power and competitive advantage',
    'Earnings Growth': 'Consistent earnings growth over 5+ years shows business strength and management capability',
    'Understandable Business': 'Simple, understandable business models that are easier to evaluate',
    'Economic Moat': 'Companies with strong competitive advantages that protect market share and profitability'
}

# Define Peter Lynch's investment principles in more detail
lynch_principles = {
    'PEG Ratio': 'Price/Earnings to Growth ratio < 1 is excellent, < 2 is good - shows growth at reasonable price',
    'Growth Potential': 'Companies with strong growth potential in sales and earnings',
    'Reasonable P/E': 'P/E ratio should be reasonable relative to growth rate',
    'Strong Balance Sheet': 'Low debt and strong financial position',
    'Cash Flow': 'Strong and consistent cash flow generation',
    'Niche Market': 'Companies that dominate a niche market or have a unique product/service'
}

# 1. Detailed analysis of Buffett companies
print("Analyzing companies based on Warren Buffett's investment principles...")

# Get detailed data for Buffett companies
buffett_detailed = ranked_df[ranked_df['NSE Code'].isin(buffett_companies['NSE Code'])].copy()

# Create a more detailed analysis of each Buffett company
buffett_analysis = []
for _, company in buffett_detailed.iterrows():
    analysis = {
        'Name': company['Name'],
        'NSE Code': company['NSE Code'],
        'Buffett Score': company['Buffett Score'],
        'Overall Rank': company['Rank'],
        'ROE': company['Return on equity'],
        'ROCE': company['Return on capital employed'],
        'Debt to Equity': company['Debt to equity'],
        'OPM': company['OPM'],
        'Profit Growth 5Y': company['Profit growth 5Years'],
        'Sales Growth 5Y': company['Sales growth 5Years'],
        'P/E Ratio': company['Price to Earning'],
        'Current Ratio': company['Current ratio'],
        'Interest Coverage': company['Interest Coverage Ratio'],
        'Free Cash Flow': company['Free cash flow last year'],
        'Market Cap': company['Market Capitalization']
    }
    
    # Evaluate against Buffett principles
    principles_match = []
    if company['Return on equity'] > 15:
        principles_match.append('Consistent ROE')
    if company['Debt to equity'] < 0.5:
        principles_match.append('Low Debt')
    if company['OPM'] > 20:
        principles_match.append('High Margins')
    if company['Profit growth 5Years'] > 10:
        principles_match.append('Earnings Growth')
    if company['Interest Coverage Ratio'] > 10:
        principles_match.append('Economic Moat')
    
    analysis['Principles Matched'] = principles_match
    analysis['Match Percentage'] = len(principles_match) / 5 * 100
    
    buffett_analysis.append(analysis)

# Convert to DataFrame
buffett_df = pd.DataFrame(buffett_analysis)
buffett_df.to_csv('/home/ubuntu/financial_model/investment_philosophy/buffett_detailed_analysis.csv', index=False)

# Create a visualization of principles matched by each company
plt.figure(figsize=(14, 10))
companies = buffett_df['Name'].tolist()
principles = ['Consistent ROE', 'Low Debt', 'High Margins', 'Earnings Growth', 'Economic Moat']

# Create a matrix of 1s and 0s for heatmap
heatmap_data = []
for _, company in buffett_df.iterrows():
    row = []
    for principle in principles:
        if principle in company['Principles Matched']:
            row.append(1)
        else:
            row.append(0)
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=companies, columns=principles)
sns.heatmap(heatmap_df, cmap='Blues', cbar=False, linewidths=.5, annot=True, fmt='d')
plt.title("Warren Buffett Investment Principles Matched by Companies", fontsize=16)
plt.tight_layout()
plt.savefig('/home/ubuntu/financial_model/investment_philosophy/buffett_principles_heatmap.png')
plt.close()

# 2. Detailed analysis of Lynch companies
print("Analyzing companies based on Peter Lynch's investment principles...")

# Get detailed data for Lynch companies
lynch_detailed = ranked_df[ranked_df['NSE Code'].isin(lynch_companies['NSE Code'])].copy()

# Create a more detailed analysis of each Lynch company
lynch_analysis = []
for _, company in lynch_detailed.iterrows():
    analysis = {
        'Name': company['Name'],
        'NSE Code': company['NSE Code'],
        'Lynch Score': company['Lynch Score'],
        'Overall Rank': company['Rank'],
        'PEG Ratio': company['PEG Ratio'],
        'P/E Ratio': company['Price to Earning'],
        'Sales Growth 5Y': company['Sales growth 5Years'],
        'Profit Growth 5Y': company['Profit growth 5Years'],
        'ROE': company['Return on equity'],
        'Debt to Equity': company['Debt to equity'],
        'Free Cash Flow': company['Free cash flow last year'],
        'Market Cap': company['Market Capitalization']
    }
    
    # Evaluate against Lynch principles
    principles_match = []
    if company['PEG Ratio'] < 2:
        principles_match.append('PEG Ratio')
    if company['Sales growth 5Years'] > 15:
        principles_match.append('Growth Potential')
    if company['Price to Earning'] < ranked_df['Price to Earning'].median():
        principles_match.append('Reasonable P/E')
    if company['Debt to equity'] < 0.5:
        principles_match.append('Strong Balance Sheet')
    if company['Free cash flow last year'] > 0:
        principles_match.append('Cash Flow')
    
    analysis['Principles Matched'] = principles_match
    analysis['Match Percentage'] = len(principles_match) / 5 * 100
    
    lynch_analysis.append(analysis)

# Convert to DataFrame
lynch_df = pd.DataFrame(lynch_analysis)
lynch_df.to_csv('/home/ubuntu/financial_model/investment_philosophy/lynch_detailed_analysis.csv', index=False)

# If we have Lynch companies, create visualizations
if not lynch_df.empty:
    # Create a visualization of principles matched by each company
    plt.figure(figsize=(14, 10))
    companies = lynch_df['Name'].tolist()
    principles = ['PEG Ratio', 'Growth Potential', 'Reasonable P/E', 'Strong Balance Sheet', 'Cash Flow']
    
    # Create a matrix of 1s and 0s for heatmap
    heatmap_data = []
    for _, company in lynch_df.iterrows():
        row = []
        for principle in principles:
            if principle in company['Principles Matched']:
                row.append(1)
            else:
                row.append(0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=companies, columns=principles)
    sns.heatmap(heatmap_df, cmap='Greens', cbar=False, linewidths=.5, annot=True, fmt='d')
    plt.title("Peter Lynch Investment Principles Matched by Companies", fontsize=16)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/financial_model/investment_philosophy/lynch_principles_heatmap.png')
    plt.close()

# 3. Create a detailed report on investment philosophy analysis
with open('/home/ubuntu/financial_model/investment_philosophy/investment_philosophy_analysis.md', 'w') as f:
    f.write("# Investment Philosophy Analysis\n\n")
    
    # Warren Buffett section
    f.write("## Warren Buffett Investment Philosophy\n\n")
    f.write("Warren Buffett's investment approach focuses on identifying companies with strong fundamentals, sustainable competitive advantages, and excellent management. His philosophy emphasizes long-term value investing rather than short-term market fluctuations.\n\n")
    
    f.write("### Key Principles\n\n")
    for principle, description in buffett_principles.items():
        f.write(f"- **{principle}**: {description}\n")
    
    f.write("\n### Top Companies Matching Buffett's Criteria\n\n")
    for i, row in buffett_df.sort_values('Buffett Score', ascending=False).head(5).iterrows():
        f.write(f"#### {i+1}. {row['Name']} (NSE: {row['NSE Code']})\n")
        f.write(f"- **Buffett Score**: {row['Buffett Score']:.2f}\n")
        f.write(f"- **Overall Rank**: {row['Overall Rank']}\n")
        f.write(f"- **ROE**: {row['ROE']:.2f}%\n")
        f.write(f"- **Debt to Equity**: {row['Debt to Equity']:.2f}\n")
        f.write(f"- **Operating Profit Margin**: {row['OPM']:.2f}%\n")
        f.write(f"- **5-Year Profit Growth**: {row['Profit Growth 5Y']:.2f}%\n")
        f.write(f"- **Interest Coverage Ratio**: {row['Interest Coverage']:.2f}\n")
        f.write(f"- **Principles Matched**: {', '.join(row['Principles Matched'])}\n")
        f.write(f"- **Match Percentage**: {row['Match Percentage']:.2f}%\n\n")
        
        f.write(f"**Why This Company Matches Buffett's Philosophy**: ")
        reasons = []
        if 'Consistent ROE' in row['Principles Matched']:
            reasons.append(f"strong return on equity of {row['ROE']:.2f}%")
        if 'Low Debt' in row['Principles Matched']:
            reasons.append(f"low debt-to-equity ratio of {row['Debt to Equity']:.2f}")
        if 'High Margins' in row['Principles Matched']:
            reasons.append(f"high operating margin of {row['OPM']:.2f}%")
        if 'Earnings Growth' in row['Principles Matched']:
            reasons.append(f"consistent profit growth of {row['Profit Growth 5Y']:.2f}% over 5 years")
        if 'Economic Moat' in row['Principles Matched']:
            reasons.append(f"strong economic moat indicated by high interest coverage ratio of {row['Interest Coverage']:.2f}")
        
        f.write("This company demonstrates " + ", ".join(reasons) + ".\n\n")
    
    # Peter Lynch section
    f.write("## Peter Lynch Investment Philosophy\n\n")
    f.write("Peter Lynch's investment strategy focuses on finding companies with good growth at reasonable prices. He believes in investing in what you know and understand, and looks for companies with strong growth potential that are undervalued by the market.\n\n")
    
    f.write("### Key Principles\n\n")
    for principle, description in lynch_principles.items():
        f.write(f"- **{principle}**: {description}\n")
    
    f.write("\n### Top Companies Matching Lynch's Criteria\n\n")
    if not lynch_df.empty:
        for i, row in lynch_df.sort_values('Lynch Score', ascending=False).head(5).iterrows():
            f.write(f"#### {i+1}. {row['Name']} (NSE: {row['NSE Code']})\n")
            f.write(f"- **Lynch Score**: {row['Lynch Score']:.2f}\n")
            f.write(f"- **Overall Rank**: {row['Overall Rank']}\n")
            f.write(f"- **PEG Ratio**: {row['PEG Ratio']:.2f}\n")
            f.write(f"- **P/E Ratio**: {row['P/E Ratio']:.2f}\n")
            f.write(f"- **5-Year Sales Growth**: {row['Sales Growth 5Y']:.2f}%\n")
            f.write(f"- **5-Year Profit Growth**: {row['Profit Growth 5Y']:.2f}%\n")
            f.write(f"- **ROE**: {row['ROE']:.2f}%\n")
            f.write(f"- **Principles Matched**: {', '.join(row['Principles Matched'])}\n")
            f.write(f"- **Match Percentage**: {row['Match Percentage']:.2f}%\n\n")
            
            f.write(f"**Why This Company Matches Lynch's Philosophy**: ")
            reasons = []
            if 'PEG Ratio' in row['Principles Matched']:
                reasons.append(f"excellent PEG ratio of {row['PEG Ratio']:.2f}")
            if 'Growth Potential' in row['Principles Matched']:
                reasons.append(f"strong sales growth of {row['Sales Growth 5Y']:.2f}% over 5 years")
            if 'Reasonable P/E' in row['Principles Matched']:
                reasons.append(f"reasonable P/E ratio of {row['P/E Ratio']:.2f}")
            if 'Strong Balance Sheet' in row['Principles Matched']:
                reasons.append(f"strong balance sheet with debt-to-equity ratio of {row['Debt to Equity']:.2f}")
            if 'Cash Flow' in row['Principles Matched']:
                reasons.append(f"positive free cash flow of ₹{row['Free Cash Flow']:.2f} crores")
            
            f.write("This company demonstrates " + ", ".join(reasons) + ".\n\n")
    else:
        f.write("No companies strongly match Peter Lynch's investment criteria based on our analysis.\n\n")
    
    # Comparison section
    f.write("## Comparison of Investment Philosophies\n\n")
    f.write("While both Warren Buffett and Peter Lynch are value investors at heart, they have different approaches:\n\n")
    f.write("- **Warren Buffett** focuses on companies with strong competitive advantages, consistent returns, and excellent management. He prefers established companies with predictable earnings and is willing to pay a fair price for quality.\n\n")
    f.write("- **Peter Lynch** is more growth-oriented and looks for companies with good growth at reasonable prices. He's willing to invest in smaller, less established companies if they show strong growth potential.\n\n")
    
    f.write("### Companies That Match Both Philosophies\n\n")
    # Find companies that appear in both lists
    if not lynch_df.empty:
        both_philosophies = set(buffett_df['NSE Code']).intersection(set(lynch_df['NSE Code']))
        if both_philosophies:
            for code in both_philosophies:
                buffett_row = buffett_df[buffett_df['NSE Code'] == code].iloc[0]
                lynch_row = lynch_df[lynch_df['NSE Code'] == code].iloc[0]
                
                f.write(f"#### {buffett_row['Name']} (NSE: {code})\n")
                f.write(f"- **Buffett Score**: {buffett_row['Buffett Score']:.2f}\n")
                f.write(f"- **Lynch Score**: {lynch_row['Lynch Score']:.2f}\n")
                f.write(f"- **Overall Rank**: {buffett_row['Overall Rank']}\n")
                f.write(f"- **ROE**: {buffett_row['ROE']:.2f}%\n")
                f.write(f"- **PEG Ratio**: {lynch_row['PEG Ratio']:.2f}\n")
                f.write(f"- **5-Year Sales Growth**: {buffett_row['Sales Growth 5Y']:.2f}%\n")
                f.write(f"- **5-Year Profit Growth**: {buffett_row['Profit Growth 5Y']:.2f}%\n\n")
                
                f.write("This company combines the best of both investment philosophies, showing both strong fundamentals (Buffett) and good growth at a reasonable price (Lynch).\n\n")
        else:
            f.write("No companies strongly match both investment philosophies based on our analysis.\n\n")
    else:
        f.write("No companies strongly match both investment philosophies based on our analysis.\n\n")
    
    f.write("## Conclusion\n\n")
    f.write("Our analysis shows that the Indian IT sector has several companies that align well with Warren Buffett's investment philosophy, particularly those with strong returns on equity, low debt, and consistent profit margins. However, fewer companies match Peter Lynch's criteria for growth at a reasonable price, possibly due to the relatively high valuations in the sector.\n\n")
    f.write("Investors should consider their own investment philosophy and risk tolerance when selecting companies from this analysis. Those seeking stability and consistent returns might prefer companies matching Buffett's criteria, while those seeking growth might look to companies matching Lynch's approach.\n\n")

print("Investment philosophy analysis complete. Results saved to /home/ubuntu/financial_model/inv<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>