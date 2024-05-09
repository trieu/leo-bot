import json

from leoai.leo_chatbot import ask_question, translate_text, detect_language, extract_data_from_chat_message_by_ai, GOOGLE_GENAI_API_KEY
from leoai.leo_datamodel import Message, UpdateProfileEvent, ChatMessage

def test_extract_data_from_chat_message_by_ai():
    story = """
        Cho tôi đặt hàng gấp 10 áo sơ mi trắng của shop 
        Vui lòng giao hàng đến địa chỉ 123 Đường ABC, Quận 1, TP. HCM.
        Điện thoại của tôi là 0987654321, tên của tôi là Nguyễn Văn A.
    """
    msg = ChatMessage(content=story)
    extracted_data = extract_data_from_chat_message_by_ai(msg)

    # Print the pretty-printed JSON string
    json_str = json.dumps(extracted_data, indent=4, ensure_ascii=False)
    print(json_str)