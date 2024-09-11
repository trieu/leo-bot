import numpy as np

"""
Customer Lifetime Value (CLV) is a key metric in understanding how valuable a customer is to a business over the long term. 
In the context of an Education MBA school, we can calculate CLV by considering factors like the average tuition fees, the probability of a student completing the program, any additional courses or certifications they might purchase, and the likelihood of future referrals.
Here is a Python code that can compute CLV based on key parameters such as average tuition, retention rate, and referral value:
"""


# Define a function to calculate CLV
def calculate_clv(average_tuition, retention_rate, average_additional_purchases, referral_value, discount_rate, program_duration_years):
    """
    Calculate the Customer Lifetime Value (CLV) for an MBA school.

    Parameters:
    - average_tuition: The average tuition fee per year per customer.
    - retention_rate: The probability that a customer will continue their program each year.
    - average_additional_purchases: The average additional revenue from purchases (certifications, workshops, etc.).
    - referral_value: The monetary value from referrals (if students refer others).
    - discount_rate: The discount rate (for net present value calculation).
    - program_duration_years: The duration of the MBA program in years.

    Returns:
    - clv: The calculated Customer Lifetime Value.
    """
    clv = 0

    for year in range(1, program_duration_years + 1):
        yearly_value = (average_tuition + average_additional_purchases) * retention_rate ** (year - 1)
        discounted_value = yearly_value / ((1 + discount_rate) ** year)
        clv += discounted_value

    # Adding the value from potential referrals (not discounted)
    clv += referral_value
    
    return clv

# Define inputs
average_tuition = 20000  # Average tuition fee per year
retention_rate = 0.85    # 85% probability of continuing each year
average_additional_purchases = 3000  # Average additional revenue from certifications, workshops
referral_value = 5000     # Value from referrals (if students refer others)
discount_rate = 0.10      # 10% discount rate
program_duration_years = 2 # Duration of the MBA program in years

# Calculate CLV
clv = calculate_clv(average_tuition, retention_rate, average_additional_purchases, referral_value, discount_rate, program_duration_years)

print(f"Customer Lifetime Value (CLV) for the MBA school is: ${clv:.3f}")
