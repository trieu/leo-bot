
from collections import defaultdict
import time
import datetime

CUSTOMER_ID = 0
DATE = 1
AMOUNT = 2
LIFETIME_ONE_DAY = 1000 * 60 * 60 * 24

# This code is about how to compute CLV https://en.wikipedia.org/wiki/Customer_lifetime_value
# Video about CLV: https://www.youtube.com/watch?v=JR8_GNINmuQ
# Sample data in Google Sheets: https://docs.google.com/spreadsheets/d/1Hb0jsAc22hXHeUBGlV2M-jcSz2M3aF5Jo05_cHdr0no/edit?usp=sharing

''' 
  Groups customers by their ID calculates customer lifetime value for each.
 
  @param {Any[][]} rows - array of customer ID, date, amount.
  @returns {Any[][]} - table, each row containing:
    - average order size
    - avgOrderFrequency,
    - avgCustomerValue,
    - avgCustomerLifespan,
    - customerLifetimeValue,
    - map of id:orders
'''


def lifetime_values(rows, product_lifetime):
    orders_per_customer = group_by_key(rows)

    result = []
    for id, orders in orders_per_customer.items():
        total_revenue = sum(order['amount'] for order in orders)

        dates_range = {'first': None, 'last': None}
        for order in orders:
            date = order['date']
            dates_range['first'] = min(dates_range['first'] or date, date)
            dates_range['last'] = max(dates_range['last'] or date, date)

        # convert product_lifecycle to product_lifetime_in_milisecs
        product_lifetime_in_milisecs = LIFETIME_ONE_DAY * product_lifetime
        # compute total_months as customer_lifetime
        total_months = (
            dates_range['last'] - dates_range['first']) / product_lifetime_in_milisecs
        if total_months == 0:
            total_months = 1

        avg_order_size = total_revenue / len(orders)
        avg_order_frequency = len(orders) / len(orders_per_customer)
        avg_customer_value = avg_order_size * avg_order_frequency
        avg_product_lifetime_in_milisecs = total_months
        customer_lifetime_value = avg_customer_value * avg_product_lifetime_in_milisecs

        if customer_lifetime_value == 0:
            avg_customer_value *= 0.1

        result.append([
            id,
            avg_order_size,
            avg_order_frequency,
            avg_customer_value,
            avg_product_lifetime_in_milisecs,
            customer_lifetime_value
        ])
    return result


def group_by_key(key_values):
    key_values_map = defaultdict(list)
    for key_value in key_values:
        key, value = key_value[CUSTOMER_ID], {
            'date': key_value[DATE], 'amount': key_value[AMOUNT]}
        key_values_map[key].append(value)
    return key_values_map


def to_timestamp(s):
    return time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()) * 1000

# test


# array of [customer ID, transaction-date, total-value-of-order]
orders = [
    [1, to_timestamp("01/06/2023"), 100],
    [1, to_timestamp("30/06/2023"), 200],
    [2, to_timestamp("15/01/2023"), 300],
    [2, to_timestamp("15/06/2023"), 400],
    [3, to_timestamp("20/05/2023"), 500],
    [3, to_timestamp("20/06/2023"), 200],
    [4, to_timestamp("30/06/2023"), 100]
]

# after 30 days, the product could be purchased again
product_lifetime = 30
result = lifetime_values(orders, product_lifetime)

for row in result:
    print(f'Customer ID: {row[0]}')
    print(f'Average Order Size: {row[1]}')
    print(f'Average Order Frequency: {row[2]}')
    print(f'Average Customer Value: {row[3]}')
    print(f'Average Customer Lifespan: {row[4]}')
    print(f'Customer Lifetime Value: {row[5]}\n')
