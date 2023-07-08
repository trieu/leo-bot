import numpy as np
import pandas as pd
import time
import datetime
import dask.dataframe as dd

from faker import Faker

Faker.seed(1000)
fake = Faker("en_GB")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

LIFETIME_ONE_DAY = 1000 * 60 * 60 * 24
df_header = ['AVG Order Size', 'AVG Order Frequency',
                  'AVG Customer Value', 'AVG Customer Lifespan', 'CLV']
def to_timestamp(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()) * 1000

def compute_clv(df, product_lifetime = 30, first_purchasing_date = 0, last_purchasing_date = 0):
    results = []
    total_revenue = df['Total Price'].sum()
    dates_range = {'first': to_timestamp(first_purchasing_date), 'last': to_timestamp(last_purchasing_date) }

    # convert product_lifecycle to product_lifetime_in_milisecs
    product_lifetime_in_milisecs = LIFETIME_ONE_DAY * product_lifetime

    # compute total_months as customer_lifetime
    total_months = (dates_range['last'] - dates_range['first']) / product_lifetime_in_milisecs
    if total_months == 0:
        total_months = 1

    avg_order_size = total_revenue / len(df)
    avg_order_frequency = len(df) / df['InvoiceDate'].nunique()
    avg_customer_value = avg_order_size * avg_order_frequency
    avg_customer_lifespan = total_months
    customer_lifetime_value = avg_customer_value * avg_customer_lifespan

    if customer_lifetime_value == 0:
        avg_customer_value *= 0.1

    results.append([
        avg_order_size,
        avg_order_frequency,
        avg_customer_value,
        avg_customer_lifespan,
        customer_lifetime_value
    ])    
    return pd.DataFrame(results, columns=df_header)


# download file at https://drive.google.com/file/d/1g7LyuJFRzBFeWBbmrirGozxjCy61Cvyc/view
# excel_reader = pd.read_excel("online_retail_II.xlsx")
# excel_reader.to_csv (r'online_retail_II.csv', index = None, header=True)

data_reader = pd.read_csv('online_retail_II.csv')
dfx = pd.DataFrame(data_reader)
sample = dfx.head(200000)
ddf = dd.from_pandas(sample, npartitions=10)

# top10 = ddf.head(10)
# print("top10 DataFrame:")
# print(top10)

# select sample data


# filter  customer Ids
filter_customer_ids = [18102.0, 15998.0, 15362.0, 15055, 13995]
result = sample.loc[sample['Customer ID'].isin(filter_customer_ids)].groupby(['Customer ID'], group_keys=True)

# loop in dataframe to compute CLV for each customer
for groupKeys, groupData in result:
    total_price = groupData['Price'].astype(float) * groupData['Quantity'].astype(float)
    groupData['Total Price'] = total_price
    first_purchasing_date = groupData['InvoiceDate'].min()
    last_purchasing_date = groupData['InvoiceDate'].max()

    keys = ','.join(str(item) for item in groupKeys)
    
    print('\n => DataFrame:')
    # print(groupData)
    print(first_purchasing_date)
    print(last_purchasing_date)

    clv_dataframe = compute_clv(groupData, 30, first_purchasing_date, last_purchasing_date)
    print("=> Customer ID:  " + keys)
    print(clv_dataframe)
    
print('Total {0} Customer'.format(len(result)))    




def test():
    index = pd.date_range("2021-09-01", periods=2400, freq="1H")
    df = pd.DataFrame({"a": np.arange(2400), "b": list("abcaddbe" * 300)}, index=index)
    ddf = dd.from_pandas(df, npartitions=10)
    rs = ddf["2021-10-01": "2021-10-09 5:00"].compute()
    print(ddf)
    print(rs)
    print(ddf.a.mean().compute())