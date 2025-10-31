# leoai/rag_intent_analyzer.py
import json
import logging
import asyncio
import re

logger = logging.getLogger("IntentAnalyzer")

class IntentAnalyzer:
    """Analyzes a message for intent, confidence, and keywords."""

    def __init__(self, gemini_client):
        self.client = gemini_client

    async def analyze(self, message: str):
        PROMPT_TEMPLATE = """
        You are a data extractor. Please analyze the message below. Extract key information and return a single, valid JSON object enclosed in ```json ... ``` markdown block. Do not add any text before or after the JSON block.
        Your JSON object MUST have this exact structure:
        {{
            "keywords": ["list of keywords from the message"],
            "last_intent_label": "the label of intent from the message",
            "last_intent_confidence": "a probability score between 0 and 1"
        }}

        --- Message ---
        {message}
        """

        try:
            prompt = PROMPT_TEMPLATE.format(message=message)
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(None, lambda: self.client.generate_content(prompt))
            logger.info(f"🧩 Raw AI intent analysis response:\n{raw_response}")

            if not raw_response:
                logger.warning("⚠️ AI client returned empty response.")
                return {"last_intent_label": None, "last_intent_confidence": 0.0, "keywords": []}

            # Extract JSON from ```json ... ```
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
            if not match:
                logger.warning("⚠️ No valid JSON block in AI response.")
                return {"last_intent_label": None, "last_intent_confidence": 0.0, "keywords": []}

            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.exception("❌ JSON decoding failed in intent analysis.")
                return {"last_intent_label": None, "last_intent_confidence": 0.0, "keywords": []}

            return {
                "last_intent_label": data.get("last_intent_label"),
                "last_intent_confidence": float(data.get("last_intent_confidence", 0.0)),
                "keywords": data.get("keywords", []),
            }

        except Exception:
            logger.exception("❌ Unexpected error during intent analysis.")
            return {"last_intent_label": None, "last_intent_confidence": 0.0, "keywords": []}