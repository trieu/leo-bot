import asyncio
import json
import logging
import os

from leoai.ai_core import GeminiClient
from leoai.ai_index import ContentIndex
from leoai.email_sender import EmailSender
from leoai.rag_agent import get_base_context
from test_poc import test_extract_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test")

def pretty(obj):
    """Helper for beautiful JSON printing."""
    return json.dumps(obj, indent=2, ensure_ascii=False)

if __name__ == "__main__1":
    
    
    # Uncomment if you want to run extraction tests
    # test_extract_data.test_extract_data_from_chat_message_by_ai()
    # test_extract_data.test_extract_data_from_chat_message_by_human()

    gemini = GeminiClient()
    indexer = ContentIndex(gemini)

    # --- Sample Pages ---
    sample_pages = {
        1: """
        Chapter 1: Customer 360 Overview
        - Identity Resolution
        - Data Ingestion
        - Analytics & AI
        """,
        2: """
        Chapter 2: Marketing Automation
        - Campaign Orchestration
        - Personalization
        - A/B Testing
        """,
        3: """
        Chapter 3: Data Architecture
        - Data Lakes
        - Data Warehouses
        - Real-Time Streaming
        """,
        4: """
        Chapter 4: Machine Learning Use Cases
        - Recommendation Systems
        - Predictive Analytics
        - Natural Language Processing
        """,
        5: """
        Chapter 5: Governance & Compliance
        - Data Privacy
        - Security
        - Regulatory Frameworks
        """
    }

    # --- Index all pages ---
    for page_num, text in sample_pages.items():
        logger.info(f"\n=== Extract & Save Page {page_num} ===")
        toc = indexer.extract_toc(text)
        logger.info(pretty(toc))

        page_id = indexer.save_page(page_num, toc)
        logger.info(f"Saved PageIndex ID: {page_id}")

    # --- Query test ---
    queries = [
        "identity resolution",
        "marketing campaign",
        "data warehouse",
        "predictive analytics",
        "GDPR compliance"
    ]

    for q in queries:
        logger.info(f"\n=== Query: {q} ===")
        results = indexer.query_similar(q, top_k=3)
        logger.info(pretty(results))

# -------------------------------------------------------------------------
# Example usage (if run standalone)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        agent = EmailSender()

        context = get_base_context()
        context["user_profile"]["first_name"] = "Trieu"
        context["user_context"]["location"] = "Saigon"

        success = await agent.send(
            to_email="tantrieuf31.database@gmail.com",
            subject="AI Agent Test Email",
            template_name="welcome_email.html",
            context=context,
        )
        print("Email sent:", success)

    asyncio.run(main())