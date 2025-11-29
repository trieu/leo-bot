import logging
# Assuming this import structure for the Google Generative AI SDK Schema type
from google.genai.types import Schema

logger = logging.getLogger(__name__)


def build_weather_schema() -> Schema:
    """
    Build and return the Weather Forecast schema used for Gemini JSON output.
    This schema strictly matches the weather forecast JSON structure.
    """

    try:
        return Schema(
            type="object",
            properties={
                "location": Schema(
                    type="object",
                    properties={
                        # NOTE: While the original schema used string,
                        # latitude/longitude should ideally be numerical for downstream processing.
                        "name": Schema(type="string", description="The recognized name of the location."),
                        "city": Schema(type="string", description="The city or major locality."),
                        "country": Schema(type="string", description="The country of the location."),
                        "latitude": Schema(type="string", description="The latitude in decimal degrees (as a string)."),
                        "longitude": Schema(type="string", description="The longitude in decimal degrees (as a string)."),
                        "time_zone": Schema(type="string", description="The IANA time zone identifier (e.g., Asia/Ho_Chi_Minh).")
                    },
                    required=[
                        "name",
                        "city",
                        "country",
                        "latitude",
                        "longitude",
                        "time_zone"
                    ]
                ),

                "metadata": Schema(
                    type="object",
                    properties={
                        "data_source": Schema(type="string", description="The primary source of the forecast data."),
                        "model_source": Schema(type="string", description="The weather model used (e.g., ECMWF)."),
                        "updated_at": Schema(type="string", description="Timestamp of the last update.")
                    },
                    required=["data_source", "model_source", "updated_at"]
                ),

                "forecast_days": Schema(
                    type="array",
                    description="Daily forecast aggregated by 3-hour intervals.",
                    items=Schema(
                        type="object",
                        properties={
                            "date": Schema(type="string", description="Date of the forecast (e.g., '2025-11-28')."),
                            "hours": Schema(
                                type="array",
                                description="Hourly forecast details.",
                                items=Schema(
                                    type="object",
                                    properties={
                                        "time_hour": Schema(type="integer", description="Hour of the day (0-23)."),
                                        "temperature_c": Schema(type="number", description="Temperature in Celsius."),
                                        "rain_mm": Schema(type="number", description="Rainfall in millimeters."),
                                        "wind_ms": Schema(type="number", description="Wind speed in m/s.")
                                    },
                                    required=[
                                        "time_hour",
                                        "temperature_c",
                                        "rain_mm",
                                        "wind_ms"
                                    ]
                                )
                            )
                        },
                        required=["date", "hours"]
                    )
                )
            },
            required=["location", "metadata", "forecast_days"]
        )

    except Exception as e:
        logger.exception("Failed to build Weather Forecast schema.")
        raise e


def build_geolocation_schema() -> Schema:
    """
    Build and return the Geolocation schema, designed to extract location
    coordinates and descriptive details from an image or text input.
    """
    try:
        return Schema(
            type="object",
            properties={
                "latitude": Schema(
                    type="number",
                    description="The determined geographical latitude in decimal degrees. Must be a float."
                ),
                "longitude": Schema(
                    type="number",
                    description="The determined geographical longitude in decimal degrees. Must be a float."
                ),
                "location_name": Schema(
                    type="string",
                    description="The most specific descriptive name for the location (e.g., landmark, building name, or street)."
                ),
                "city": Schema(
                    type="string",
                    description="The city or major locality where the image was taken."
                ),
                "country": Schema(
                    type="string",
                    description="The country where the image was taken."
                ),
                "confidence_score": Schema(
                    type="number",
                    description="A numerical score (0.0 to 1.0) indicating the model's certainty about the extracted location. Use 0.9 or higher for high confidence."
                )
            },
            required=["latitude", "longitude", "city", "country"]
        )
    except Exception as e:
        logger.exception("Failed to build Geolocation schema.")
        raise e


# Exported schemas for direct import across your codebase
WEATHER_FORECAST_SCHEMA = build_weather_schema()
GEOLOCATION_SCHEMA = build_geolocation_schema()