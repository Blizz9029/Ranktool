import pandas as pd
import json
import numpy as np

# Load the original CSV data
df = pd.read_csv('/home/ubuntu/financial_model/companies_data.csv')

# Load the Yahoo Finance data
with open('/home/ubuntu/financial_model/profiles_data.json', 'r') as f:
    profiles_data = json.load(f)

with open('/home/ubuntu/financial_model/insights_data.json', 'r') as f:
    insights_data = json.load(f)

# Create dictionaries for easy lookup
profiles_dict = {item['NSE Code']: item['Profile'] for item in profiles_data}
insights_dict = {item['NSE Code']: item['Insights'] for item in insights_data}

# Function to extract business summary from profile data
def get_business_summary(nse_code):
    if nse_code in profiles_dict:
        profile = profiles_dict[nse_code]
        try:
            if 'quoteSummary' in profile and 'result' in profile['quoteSummary'] and len(profile['quoteSummary']['result']) > 0:
                summary_profile = profile['quoteSummary']['result'][0].get('summaryProfile', {})
                return summary_profile.get('longBusinessSummary', '')
        except:
            pass
    return ''

# Function to extract industry and sector from profile data
def get_industry_sector(nse_code):
    industry = ''
    sector = ''
    if nse_code in profiles_dict:
        profile = profiles_dict[nse_code]
        try:
            if 'quoteSummary' in profile and 'result' in profile['quoteSummary'] and len(profile['quoteSummary']['result']) > 0:
                summary_profile = profile['quoteSummary']['result'][0].get('summaryProfile', {})
                industry = summary_profile.get('industry', '')
                sector = summary_profile.get('sector', '')
        except:
            pass
    return industry, sector

# Function to extract valuation metrics from insights data
def get_valuation_metrics(nse_code):
    valuation_desc = ''
    if nse_code in insights_dict:
        insights = insights_dict[nse_code]
        try:
            if 'finance' in insights and 'result' in insights['finance']:
                instrument_info = insights['finance']['result'].get('instrumentInfo', {})
                valuation = instrument_info.get('valuation', {})
                valuation_desc = valuation.get('description', '')
        except:
            pass
    return valuation_desc

# Add business summary, industry, and sector to the dataframe
df['Business Summary'] = df['NSE Code'].apply(get_business_summary)
df[['Industry from Yahoo', 'Sector']] = df.apply(lambda row: pd.Series(get_industry_sector(row['NSE Code'])), axis=1)
df['Valuation Description'] = df['NSE Code'].apply(get_valuation_metrics)

# Calculate additional financial metrics

# 1. Calculate Buffett criteria metrics
# Consistent ROE over time (using 5-year average)
df['Consistent ROE'] = df['Average return on equity 5Years'] > 15

# Debt to Equity ratio < 0.5 (conservative)
df['Low Debt'] = df['Debt to equity'] < 0.5

# High profit margins (using OPM - Operating Profit Margin)
df['High Margins'] = df['OPM'] > 20

# Consistent earnings growth
df['Earnings Growth'] = df['Profit growth 5Years'] > 10

# 2. Calculate Peter Lynch criteria metrics
# PEG Ratio < 1 is excellent, < 2 is good
df['Good PEG'] = df['PEG Ratio'] < 2

# Companies with high growth relative to P/E
df['Growth to PE'] = df['Sales growth 5Years'] / df['Price to Earning']

# 3. Calculate comprehensive financial health score
# Normalize key metrics to 0-100 scale for scoring

# Function to normalize a series to 0-100 scale
def normalize_series(series, higher_is_better=True):
    if higher_is_better:
        return 100 * (series - series.min()) / (series.max() - series.min())
    else:
        return 100 * (series.max() - series) / (series.max() - series.min())

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

# Calculate weighted financial health score
# Weights based on user's preferences
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

df['Financial Health Score'] = sum(df[col] * weight for col, weight in weights.items())

# 4. Calculate Buffett and Lynch scores
# Buffett score components
buffett_components = {
    'Consistent ROE': 0.25,
    'Low Debt': 0.25,
    'High Margins': 0.25,
    'Earnings Growth': 0.25
}

# Convert boolean columns to float for calculation
for col in buffett_components.keys():
    df[col] = df[col].astype(float)

df['Buffett Score'] = sum(df[col] * weight for col, weight in buffett_components.items()) * 100

# Lynch score components (more focused on growth at reasonable price)
lynch_components = {
    'Good PEG': 0.3,
    'Growth to PE': 0.3,
    'ROE Score': 0.2,
    'Sales Growth Score': 0.2
}

# Normalize Growth to PE
df['Growth to PE'] = normalize_series(df['Growth to PE'])
df['Growth to PE'] = df['Growth to PE'].fillna(0)

# Convert boolean columns to float for calculation
df['Good PEG'] = df['Good PEG'].astype(float)

df['Lynch Score'] = sum(df[col] * weight for col, weight in lynch_components.items())

# Save the enhanced dataframe with calculated metrics
df.to_csv('/home/ubuntu/financial_model/companies_with_metrics.csv', index=False)

print("Financial metrics calculation complete.")
print(f"Total companies analyzed: {len(df)}")
print(f"Companies with ROE > 15%: {len(df[df['Return on equity'] > 15])}")
print(f"Companies with ROCE > 15%: {len(df[df['Return on capital employed'] > 15])}")
print(f"Companies with positive FCF: {len(df[df['Free cash flow last year'] > 0])}")
print(f"Companies with Buffett Score > 75: {len(df[df['Buffett Score'] > 75])}")
print(f"Companies with Lynch Score > 75: {len(df[df['Lynch Score'] > 75])}")
print(f"Enhanced data saved to: /home/ubuntu/financial_model/companies_with_metrics.csv")
