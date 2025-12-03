

import asyncio
import json
import sys

from leoai.data_scraper.windy_scraper import get_windy_forecast_raw_data
from tests.test_ai_prediction import extract_weather_info_from_text

async def process_weather_data(lat: float, lon: float):
    
    text_filename, screenshot_filename = await get_windy_forecast_raw_data(lat,lon)
    print(text_filename)
    
    data2 = extract_weather_info_from_text(f"./{text_filename}")

    #  Pretty-print
    # readable1 = json.dumps(data1, indent=2, ensure_ascii=False)
    # print(readable1)
    readable2 = json.dumps(data2, indent=2, ensure_ascii=False)
    print(readable2)
    print(screenshot_filename)
