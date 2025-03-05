import os
from google import genai
from google.genai import types
from api_key import api_key

client = genai.Client(api_key=api_key)


def getImageDescription():
    with open("media/prueba.mp3", "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            "Describe este audio, por favor",
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="audio/mp3",
            ),
        ],
    )

    print(response.text)


getImageDescription()