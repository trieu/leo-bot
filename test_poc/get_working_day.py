import pandas as pd

holidays = [
    pd.to_datetime("2024-01-01"),
    pd.to_datetime("2024-01-02"),
    pd.to_datetime("2024-01-03")
]

def get_working_day(start_date, days):
    limit_days = days * 2
    all_dates = pd.date_range(start=start_date, periods=limit_days, freq="D")
    
    working_days = all_dates[
        (all_dates.weekday < 5) &  # Monday to Friday are working days
        (~all_dates.isin(holidays))
    ]
    
    next_working_day = working_days[days-1] if not working_days.empty else None
    return next_working_day


# Example usage
start_date = pd.to_datetime("2023-12-28")
days = 7
result = get_working_day(start_date, days)
print("\n Result: ",result)
