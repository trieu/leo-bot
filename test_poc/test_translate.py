from google.cloud import translate_v2 as translate

# need export GOOGLE_APPLICATION_CREDENTIALS=
def translate_text( text: str, target: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client()
    
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result['translatedText']

print(translate_text("Hello", "vi"))
print(translate_text("iPhone 15 chả có gì mới, Apple Watch vẫn chán như mọi khi", "en"))