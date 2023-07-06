import numpy as np
import pandas as pd

import dask.dataframe as dd
import dask.array as da
import dask.bag as db

from faker import Faker

Faker.seed(1000)
fake = Faker("en_GB")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# download file at https://drive.google.com/file/d/1g7LyuJFRzBFeWBbmrirGozxjCy61Cvyc/view
dfx = pd.DataFrame(pd.read_excel("online_retail_II.xlsx"))
ddf = dd.from_pandas(dfx, npartitions=10)

# top10 = ddf.head(10)
# print("top10 DataFrame:")
# print(top10)

result = ddf.head(100).groupby(['Customer ID'], group_keys=True)
for groupKeys, groupData in result:
    total_price = groupData['Price'].astype(float) * groupData['Quantity'].astype(int)
    groupData['Total Price'] = total_price

    keys = ','.join(str(item) for item in groupKeys)
    print('\n => groupKeys: ' + keys)
    print('DataFrame:')
    print(groupData)
    print(fake.simple_profile())
    # print("DataFrame Customer ID:" + groupKeys[0] + " StockCode: " + groupKeys[1])
    

def test():
    index = pd.date_range("2021-09-01", periods=2400, freq="1H")
    df = pd.DataFrame({"a": np.arange(2400), "b": list("abcaddbe" * 300)}, index=index)
    ddf = dd.from_pandas(df, npartitions=10)
    rs = ddf["2021-10-01": "2021-10-09 5:00"].compute()
    print(ddf)
    print(rs)
    print(ddf.a.mean().compute())