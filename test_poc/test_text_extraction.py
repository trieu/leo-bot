import google.generativeai as genai
import textwrap
import os
import json

genai.configure(api_key=os.environ["GOOGLE_GENAI_API_KEY"])
model = genai.GenerativeModel(model_name='models/gemini-1.5-pro-latest')

story = """
Cho tui đặt hàng 1 áo sơ mi trắng của shop 
Vui lòng giao hàng đến địa chỉ 123 Đường ABC, Quận 1, TP. HCM.
Điện thoại của tôi là 0987654321. Tên của tôi là Nguyễn Văn A.

"""

response = model.generate_content(
  textwrap.dedent("""\
    Please return JSON describing the the people, places, things and relationships from this story using the following schema:

    {"people": list[PERSON], "places":list[PLACE], "things":list[THING], "relationships": list[RELATIONSHIP],"order_details": list[ORDER_DETAILS]}

    PERSON = {"name": str, "description": str, "phone_number": str, "email": str, "address": str, "start_place_name": str, "end_place_name": str}
    PLACE = {"name": str, "description": str}
    THING = {"name": str, "description": str, "start_place_name": str, "end_place_name": str}
    ORDER_DETAILS = {"product_name": str, quality: int}
    RELATIONSHIP = {"person_1_name": str, "person_2_name": str, "relationship": str}

    All fields are required.
    Important: Only return a single piece of valid JSON text.
    Here is the story:

    """) + story, generation_config={'response_mime_type':'application/json'}
)

# parse response into JSON
extracted_data = json.loads(response.text)

# Print the pretty-printed JSON string
json_str = json.dumps(extracted_data, indent=4, ensure_ascii=False)
print(json_str)