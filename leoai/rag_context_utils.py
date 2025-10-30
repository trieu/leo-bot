# leoai/rag_context_utils.py
from datetime import datetime
from typing import Any, Dict

def get_date_time_now() -> str:
    """Return current timestamp as 'YYYY-MM-DD HH:MM' string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def get_base_context(request_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Return a base context model for chat summarization and RAG pipeline.
    Optionally deep-merge with provided request_data.
    """
    now_str = get_date_time_now()
    base = {
        "user_profile": {
            "first_name": None, "last_name": None,
            "primary_language": None, "primary_email": None, "primary_phone": None,
            "personal_interests": [], "personality_traits": [],
            "data_labels": [], "in_segments": [], "in_journey_maps": [],
            "product_interests": [], "content_interests": []
        },
        "user_context": {"location": None, "datetime": now_str},
        "context_summary": "",
        "context_keywords": [],
        "intent_label": None,
        "intent_confidence": 0.0
    }

    if request_data:
        def deep_merge(target: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in src.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
            return target

        base = deep_merge(base, request_data)

    return base
