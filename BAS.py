from ollama import chat
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import imageio_ffmpeg as ffmpeg
import os

# Setting Up Llama 3.2 Vision


def getResponse(prompt, audio):
    responseLlama = ChatResponse = chat(
        model="llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image]
                # "raw": [audio]
            },
        ],
    )
    return responseLlama.message.content


def getResponseLittle(prompt):
    responseLlama = ChatResponse = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return responseLlama.message.content


def transcribeAudio(audioFilePath, pipe):
    result = pipe(audioFilePath)
    return result["text"]


# Setting Up Whisper


def whisperInit():
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v3-turbo",
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        device=("cuda:0" if torch.cuda.is_available() else "cpu"),
    )


pipe = whisperInit()

os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg.get_ffmpeg_exe()


# Test
image = "unnamed.jpg"
audio = "prueba.mp3"

print(getResponse("Descríbeme la imagen en español, por favor", image))


def getResponse(prompt, audio):
    responseLlama = ChatResponse = chat(
        model="llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image],
                # "raw": [audio]
            },
        ],
    )
    return responseLlama.message.content


# ola = 1

# while ola == 1:
#     ola = input("Ingrese 1 para continuar: ")
#     print(getResponse("Describe el archivo proporcionado", audio))
