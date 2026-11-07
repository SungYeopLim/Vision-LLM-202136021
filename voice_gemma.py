import subprocess

from llama_cpp import Llama


GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

MIC_DEVICE = "plughw:3,0"
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 5

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150


def speech_to_text():
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(RECORD_SECONDS),
            AUDIO_FILE,
        ],
        check=True,
    )

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return result.stdout.strip()


def text_to_speech(text):
    subprocess.run(
        [
            PIPER_PYTHON,
            "-m",
            "piper",

            "-m",
            PIPER_MODEL,

            "-f",
            OUTPUT_FILE,

            "--",
            text,
        ],
        check=True,
    )

    subprocess.run(
        [
            "aplay",
            OUTPUT_FILE,
        ],
        check=True,
    )


llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


print("Enter : record")
print("q : quit")


while True:
    key = input("> ").strip()

    if key == "q":
        break

    try:
        print(f"Speak now ({RECORD_SECONDS}s)")
        question = speech_to_text()

        if question == "":
            print("FAIL!")
            continue

        print("\n[User]")
        print(question)

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """
                                Instruction:
                                Answer the user's question.

                                Constraint:
                                Do not use emoji, markdown, or special symbols.

                                Output Format:
                                Within two sentences in Korean.
                               """
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        answer = response["choices"][0]["message"]["content"].strip()

        print("\n[Gemma]")
        print(answer)

        text_to_speech(answer)

    except Exception:
        print("FAIL!")
