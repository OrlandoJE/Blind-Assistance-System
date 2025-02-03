from ollama import chat
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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


# Setting Up Whisper

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Test

prompt = "Qué hay en la imagen?"
image = "test.jpg"

print(getResponse(prompt, image))
