
from collections import defaultdict
import time
import datetime
import pandas as pd
from datetime import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

CUSTOMER_ID = 0
DATE = 1
AMOUNT = 2
LIFETIME_ONE_DAY = 1000 * 60 * 60 * 24

def to_timestamp(s):
    return time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()) * 1000

# array of [customer ID, transaction-date, total-value-of-order]
orders = [
    [1, to_timestamp("01/03/2023"), 100],
    [1, to_timestamp("30/04/2023"), 200],
    [1, to_timestamp("05/07/2023"), 150],
    [2, to_timestamp("15/03/2023"), 300],
    [2, to_timestamp("15/06/2023"), 400],
    [2, to_timestamp("29/06/2023"), 300],
    [3, to_timestamp("20/01/2023"), 500],
    [3, to_timestamp("20/06/2023"), 200],
    [4, to_timestamp("30/06/2023"), 100]
]

df_header = ['Customer ID', 'AVG Order Size', 'AVG Order Frequency',
                  'AVG Customer Value', 'AVG Customer Lifespan', 'CLV']


# This code is about how to compute CLV https://en.wikipedia.org/wiki/Customer_lifetime_value
# Video about CLV: https://www.youtube.com/watch?v=JR8_GNINmuQ
# Sample data in Google Sheets: https://docs.google.com/spreadsheets/d/1Hb0jsAc22hXHeUBGlV2M-jcSz2M3aF5Jo05_cHdr0no/edit?usp=sharing

''' 
  Groups customers by their ID calculates customer lifetime value for each.
 
  @param {pandas.DataFrame} - DataFrame of customer ID, date, amount.
  @returns {pandas.DataFrame} - DataFrame, each row containing:
    - customer id,
    - average order size,
    - avgOrderFrequency,
    - avgCustomerValue,
    - avgCustomerLifespan,
    - customerLifetimeValue
'''
def lifetime_values(df, product_lifetime):
    result = []
    
    for id, group in df.groupby('customer_id'):
        total_revenue = group['amount'].sum()
        dates_range = {'first': group['date'].min(), 'last': group['date'].max()}

        # convert product_lifecycle to product_lifetime_in_milisecs
        product_lifetime_in_milisecs = LIFETIME_ONE_DAY * product_lifetime

        # compute total_months as customer_lifetime
        total_months = (dates_range['last'] - dates_range['first']) / product_lifetime_in_milisecs
        if total_months == 0:
            total_months = 1

        avg_order_size = total_revenue / len(group)
        avg_order_frequency = len(group) / df['customer_id'].nunique()
        avg_customer_value = avg_order_size * avg_order_frequency
        avg_customer_lifespan = total_months
        customer_lifetime_value = avg_customer_value * avg_customer_lifespan

        if customer_lifetime_value == 0:
            avg_customer_value *= 0.1

        result.append([
            id,
            avg_order_size,
            avg_order_frequency,
            avg_customer_value,
            avg_customer_lifespan,
            customer_lifetime_value
        ])
        
    return pd.DataFrame(result, columns=df_header)

# test

df = pd.DataFrame(orders, columns=['Customer ID', 'Order Date', 'Total Value'])
df['Order Date'] = df['Order Date'].map(lambda t: dt.fromtimestamp(t / 1000))
print('\n Data about orders')
print(df)

# after 30 days, the product could be purchased again
product_lifetime = 30
df_orders = pd.DataFrame(orders, columns=['customer_id', 'date', 'amount'])
dfResult = lifetime_values(df_orders, product_lifetime)
print('\n dfResult:')
print(dfResult)