import numpy as np
import pandas as pd
import time
import datetime
import os.path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

default_profit_margin = 0.1
LIFETIME_ONE_DAY = 1000 * 60 * 60 * 24
df_header = ['Total Revenue','AVG Order Size', 'AVG Order Frequency', 'AVG Customer Value', 'AVG Customer Lifespan', 'CLV']

def to_timestamp(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()) * 1000

# Video for CLV explanation: https://www.youtube.com/watch?v=JR8_GNINmuQ
def compute_clv(df, purchasing_lifecycle = 30, first_purchasing_date = 0, last_purchasing_date = 0):
    results = []
    total_revenue = df['Total Revenue'].sum()
    dates_range = {'first': to_timestamp(first_purchasing_date), 'last': to_timestamp(last_purchasing_date) }

    # convert purchasing_lifecycle to purchasing_lifecycle_in_milisecs
    purchasing_lifecycle_in_milisecs = LIFETIME_ONE_DAY * purchasing_lifecycle

    # compute total_months as customer_lifetime
    total_months = (dates_range['last'] - dates_range['first']) / purchasing_lifecycle_in_milisecs
    if total_months == 0:
        total_months = 1

    avg_order_size = total_revenue / len(df)
    avg_order_frequency = len(df) / df['InvoiceDate'].nunique()
    avg_customer_value = avg_order_size * avg_order_frequency
    avg_customer_lifespan = total_months
    customer_lifetime_value = avg_customer_value * avg_customer_lifespan

    if customer_lifetime_value == 0:
        total_revenue *= 0.1

    results.append([
        total_revenue,
        avg_order_size,
        avg_order_frequency,
        avg_customer_value,
        avg_customer_lifespan,
        customer_lifetime_value
    ])    
    return pd.DataFrame(results, columns=df_header)

def processData(groupData):
    customer_id = ','.join(str(e) for e in groupData['Customer ID'].unique())
    # assumption: after 7 days,  a customer has a 100% chance of returning to your store
    purchasing_lifecycle = 7 
    # the price of each product mulitply with the number of quantity
    total_revenue = groupData['Price'].astype(float) * groupData['Quantity'].astype(float)
    groupData['Total Revenue'] = total_revenue * (1 - default_profit_margin)

    # time range
    first_purchasing_date = groupData['InvoiceDate'].min()
    last_purchasing_date = groupData['InvoiceDate'].max()
    
    print('\n ')
    # print(groupData)
    print('Customer ID = {0} Transaction Date from [{1}] to [{2}]'.format(customer_id, first_purchasing_date, last_purchasing_date))      

    clv_dataframe = compute_clv(groupData, purchasing_lifecycle, first_purchasing_date, last_purchasing_date)
    print(clv_dataframe.to_markdown())
    return 1

# the filename to load test data
csv_data_filename = './data/online_retail_II.csv'

# check to download test data
if not os.path.isfile(csv_data_filename):
    print(csv_data_filename + " not found !")
    exit(1)
    # test_data.donwload_online_retail_data(csv_data_filename)

data_reader = pd.read_csv(csv_data_filename)
dfx = pd.DataFrame(data_reader)
sample = dfx.head(200000) # just get a sample from 500,000 rows

# filter  customer Ids
filter_customer_ids = [18102.0, 15998.0, 15362.0, 15055, 13995]
df_result = sample.loc[sample['Customer ID'].isin(filter_customer_ids)].groupby(['Customer ID'], group_keys=True)
 
# loop in dataframe to compute CLV for each customer
total_customer = df_result.apply(processData).sum()
   
print('\n =>> Total {0} Customer'.format(total_customer))