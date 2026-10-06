from leoai.rag_agent import (
    format_place_picker,
    format_place_selection_confirmation,
    is_greeting_message,
    selected_place_index,
)


PLACES = [
    {"name": "Binh Tay Market", "distance_meters": 736.8},
    {"name": "Cho Lon Chinatown", "distance_meters": 1309.9},
    {"name": "Ong Bon Pagoda", "distance_meters": 1350.5},
]


def test_greeting_detection_is_narrow():
    assert is_greeting_message("hi")
    assert is_greeting_message("Xin chào!")
    assert not is_greeting_message("hi, where is the market?")


def test_place_selection_accepts_numbered_choices():
    assert selected_place_index("1", 3) == 0
    assert selected_place_index("option 2", 3) == 1
    assert selected_place_index("chọn 3.", 3) == 2
    assert selected_place_index("5", 3) is None


def test_place_picker_lists_nearby_places():
    answer = format_place_picker(PLACES, "Vietnamese")
    assert "Binh Tay Market" in answer
    assert "1." in answer
    assert "737 m" in answer


def test_selection_confirmation_names_selected_place():
    answer = format_place_selection_confirmation(PLACES[0], "English")
    assert "Binh Tay Market" in answer
