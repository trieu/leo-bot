
import uuid
from leoai.rag_agent import ask_question_rag

def run_test():
    # Simulated user session
    user_id = str(uuid.uuid4())  # Use a fixed UUID for consistency
    persona_id = "test-persona-001"
    touchpoint_id = "webchat-001"
    target_language = "English"
    keywords = ["billing", "payment"]

    # Example user question
    question = "How can I update my payment method?"

    print(f"🧪 Test RAG Agent for user: {user_id}")
    print(f"❓ Question: {question}\n")

    answer = ask_question_rag(
        user_id=user_id,
        question=question,
        persona_id=persona_id,
        touchpoint_id=touchpoint_id,
        target_language=target_language,
        keywords=keywords,
        answer_in_format="text",
    )

    print("🧠 Answer:\n" + "-" * 40)
    print(answer)

if __name__ == "__main__":
    run_test()
