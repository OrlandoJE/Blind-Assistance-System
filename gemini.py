import os
from google import genai
from google.genai import types

client = genai.Client()

myfile = client.files.upload(path='prueba.mp3')

response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents=[
    'Describe this audio clip',
    myfile,
  ]
)

print(response.text)