from ollama import chat
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import imageio_ffmpeg as ffmpeg
import os

# Setting Up Llama 3.2 Vision


def getResponse(prompt, image):
    responseLlama = ChatResponse = chat(
        model="llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image],
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


def transcribeAudio(audioFilePath):
    result = pipe(audioFilePath)
    return result["text"]


# Setting Up Whisper

pipe = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-large-v3-turbo",
    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
    device=("cuda:0" if torch.cuda.is_available() else "cpu"),
)

os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg.get_ffmpeg_exe()


# Test
# image = "test.jpg"

print(getResponseLittle(transcribeAudio("prueba.mp3")))
