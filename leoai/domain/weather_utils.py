
# ------------------------------
# Inject weather classification into JSON
# ------------------------------
import json

def build_weather_prompt(json_schema, cleaned_text):
    prompt_text = f"""
        You convert a Windy.com forecast-table HTML snippet into a structured weather JSON object.

        Follow these rules precisely:

        1) Treat the HTML as a strict table.  
        - Every <tr> is a row type.
        - Every <td> is a column aligned horizontally across all rows.
        - Columns NEVER shift order across rows.

        2) Day assignment:
        - A day header (e.g., "Saturday 29") applies to all subsequent columns
            until the next day header appears.
        - You MUST align each column index to the correct day based strictly
            on the order of day headers in the HTML.

        3) Row → variable mapping:
        - tr--hour: hour values in 24h integer form.
        - tr--icon: weather icon URL. Use the @2x srcset URL.
        - tr--temp: temperature in Celsius. Remove “°”.
        - tr--rain: rain in mm. Blank = 0. Do not invent values.
        - tr--wind: wind speed in knots.
        - tr--gust: wind gusts in knots.
        - tr--windDir: wind direction. Extract the numeric degrees from
            the CSS transform: rotate(Xdeg).

        4) Data integrity rules:
        - All arrays must preserve the exact left-to-right positional mapping
            across all rows.
        - Do not reorder, merge, or resample hours.
        - If a cell is blank, return 0 (for numeric fields) or null
            if allowed by the schema.

        5) Forbidden:
        - No hallucinated numbers.
        - No inferred data not explicitly shown.
        - No commentary or markdown.
        - No alternative interpretations.

        6) Required:
        - Produce ONLY valid JSON following this schema:
            {json_schema}

        The output MUST be valid JSON that parses without errors.

        Raw Windy HTML to extract from:
        ----------------
        {cleaned_text}
        ---
    """
    return prompt_text


def enrich_weather_forecast(text: str) -> dict:
    weather_info = json.loads(text)

    # Validate minimum expected structure
    if "forecast_days" not in weather_info:
        raise ValueError("Missing key: 'forecast_days' at root JSON level")

    # Loop through days & hours
    for day in weather_info["forecast_days"]:
        hours = day.get("hours", [])
        for hour in hours:

            kt = hour.get("wind_kt")
            gust_kt = hour.get("wind_gust_kt")

            if kt is not None:
                hour["wind_info"] = classify_knots(float(kt))

            if gust_kt is not None:
                hour["wind_gust_info"] = classify_knots(float(gust_kt))

    return weather_info


def knots_to_mps(knots: float) -> float:
    """
    Convert wind speed from knots (kt) to meters per second (m/s).
    1 knot = 0.514444 m/s
    """
    return knots * 0.514444


def wind_label_from_mps(mps: float) -> dict:
    """
    Return wind labels (EN + VI) based on m/s speed.
    """

    ranges = [
        (0, 0.2,  "Calm",             "Lặng gió"),
        (0.3, 1.6, "Light Air",       "Gió nhẹ"),
        (1.6, 3.3, "Light Breeze",    "Gió hiu hiu"),
        (3.3, 5.4, "Gentle Breeze",   "Gió nhẹ vừa"),
        (5.5, 7.9, "Moderate Breeze", "Gió trung bình"),
        (8.0, 10.7, "Fresh Breeze",    "Gió mạnh vừa"),
        (10.7, 13.8, "Strong Breeze",  "Gió mạnh"),
        (13.8, 17.1, "Near Gale",      "Gió gần cấp giật"),
        (17.1, 20.7, "Gale",           "Gió giật mạnh"),
        (20.7, 24.4, "Strong Gale",    "Gió giật rất mạnh"),
        (24.4, 28.4, "Storm",          "Bão"),
        (28.4, 32.6, "Violent Storm",  "Bão dữ dội"),
        (32.6, float("inf"), "Hurricane", "Cuồng phong")
    ]

    for low, high, en, vi in ranges:
        if low <= mps <= high:
            return {"en": en, "vi": vi}

    return {"en": "Unknown", "vi": "Không xác định"}


def classify_knots(knots: float) -> dict:
    """
    Convert kt → m/s and return bilingual wind classification.
    """
    mps = knots_to_mps(knots)
    label = wind_label_from_mps(mps)

    return {
        "knots": knots,
        "mps": round(mps, 2),
        "label_en": label["en"],
        "label_vi": label["vi"]
    }
