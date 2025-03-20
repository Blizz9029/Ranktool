import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import time
import json

# Load the companies data
df = pd.read_csv('/home/ubuntu/financial_model/companies_data.csv')

# Create a client for the Yahoo Finance API
client = ApiClient()

# Function to get stock profile data
def get_stock_profile(symbol):
    try:
        result = client.call_api('YahooFinance/get_stock_profile', query={'symbol': symbol, 'region': 'IN'})
        return result
    except Exception as e:
        print(f"Error fetching profile for {symbol}: {str(e)}")
        return None

# Function to get stock insights
def get_stock_insights(symbol):
    try:
        result = client.call_api('YahooFinance/get_stock_insights', query={'symbol': symbol})
        return result
    except Exception as e:
        print(f"Error fetching insights for {symbol}: {str(e)}")
        return None

# Create empty lists to store the data
profiles = []
insights = []

# Process companies with NSE codes
companies_with_nse = df[df['NSE Code'].notna()]
total = len(companies_with_nse)
print(f"Fetching data for {total} companies...")

for i, (_, row) in enumerate(companies_with_nse.iterrows()):
    company_name = row['Name']
    nse_code = row['NSE Code']
    symbol = f"{nse_code}.NS"  # Append .NS for Indian stocks
    
    print(f"Processing {i+1}/{total}: {company_name} ({symbol})")
    
    # Get profile data
    profile_data = get_stock_profile(symbol)
    if profile_data:
        profile_entry = {
            'Name': company_name,
            'NSE Code': nse_code,
            'Symbol': symbol,
            'Profile': profile_data
        }
        profiles.append(profile_entry)
        print(f"  - Profile data fetched successfully")
    else:
        print(f"  - Failed to fetch profile data")
    
    # Get insights data
    insights_data = get_stock_insights(symbol)
    if insights_data:
        insight_entry = {
            'Name': company_name,
            'NSE Code': nse_code,
            'Symbol': symbol,
            'Insights': insights_data
        }
        insights.append(insight_entry)
        print(f"  - Insights data fetched successfully")
    else:
        print(f"  - Failed to fetch insights data")
    
    # Sleep to avoid rate limiting
    time.sleep(1)

# Save the data to files
with open('/home/ubuntu/financial_model/profiles_data.json', 'w') as f:
    json.dump(profiles, f, indent=2)

with open('/home/ubuntu/financial_model/insights_data.json', 'w') as f:
    json.dump(insights, f, indent=2)

print(f"\nData fetching complete.")
print(f"Profiles fetched: {len(profiles)}/{total}")
print(f"Insights fetched: {len(insights)}/{total}")
print(f"Data saved to /home/ubuntu/financial_model/profiles_data.json and /home/ubuntu/financial_model/insights_data.json")
