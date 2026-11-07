import json
import os
import subprocess
import time
from datetime import datetime

from llama_cpp import Llama


GEMMA_MODEL_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
EVENT_JSON_PATH = "../src/json/review_event.json"
REVIEW_DIR = "../src/output/reviews"

MIC_DEVICE = "plughw:3,0"
AUDIO_FILE = "../src/audio/input.wav"
WHISPER_PATH = "../whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "../whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 8

PIPER_PYTHON = "../.piper_venv/bin/python"
PIPER_MODEL = "../src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "../src/audio/response.wav"

CONTEXT_WINDOW = 2048
LABEL_TOKENS = 10     # 라벨, 별점 최대 토큰
REVIEW_TOKENS = 300   # 리뷰 문장 최대 토큰
CLAIM_RATING = 2      # 이 별점 이하면 클레임 후보

# 구매처별 리뷰 최소 글자 수 (예시값, 실제 기준은 각 쇼핑몰 안내 확인)
MIN_CHARS = {
    "coupang": 30,
    "naver": 30,
    "musinsa": 30,
    "kurly": 30,
}

# 카테고리별 첫 질문
QUESTIONS = {
    "food": "{name} 사진을 저장했어요. 맛이나 신선도, 포장 상태는 어떤가요?",
    "living": "{name} 사진을 저장했어요. 마감이나 색상이 상품 사진과 같은가요?",
    "electronics": "{name} 사진을 저장했어요. 외관이나 구성품은 괜찮은가요?",
    "fashion": "{name} 사진을 저장했어요. 색상이나 소재는 어떤가요?",
}

# 리뷰가 최소 글자 수보다 짧을 때 추가 질문
FOLLOW_UPS = {
    "food": "리뷰가 조금 짧아요. 맛, 양, 포장 중에 더 말씀하실 부분이 있나요?",
    "living": "리뷰가 조금 짧아요. 크기, 무게, 마감 중에 더 말씀하실 부분이 있나요?",
    "electronics": "리뷰가 조금 짧아요. 외관, 구성품, 처음 써 본 느낌 중에 더 말씀하실 부분이 있나요?",
    "fashion": "리뷰가 조금 짧아요. 사이즈, 소재, 색상 중에 더 말씀하실 부분이 있나요?",
}


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

    text = result.stdout.strip()

    # 무음일 때 출력되는 [BLANK_AUDIO] 등은 빈 문자열로 처리
    if text.startswith("["):
        return ""

    return text


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


# Gemma에 system / user 메시지를 전달하고 응답 문자열을 반환하는 함수
def ask_gemma(system_prompt, user_prompt, max_tokens, temperature):
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response["choices"][0]["message"]["content"].strip()


# 구매자 발화 → 리뷰 문장
def write_review(product, speech):
    return ask_gemma(
        """
        Instruction:
        구매자가 말한 내용을 구매자가 직접 작성한 상품 리뷰 문장으로 정리하시오.

        Constraint:
        구매자가 말한 내용과 상품명만 사용하시오.
        구매자가 말한 내용은 빠짐없이 포함하고, 줄이지 마시오.
        구매자가 말하지 않은 칭찬, 특징, 경험을 추가하지 마시오.
        이모지, 마크다운, 특수 기호를 사용하지 마시오.

        Output Format:
        존댓말 한국어 문장.
        """,
        f"""
        상품명: {product['name']}
        구매자 발화: {speech}
        """,
        REVIEW_TOKENS,
        0.7,
    )


# 구매자 발화 → 긍정 / 부정 / 없음
def classify_label(speech):
    answer = ask_gemma(
        """
        Instruction:
        구매자가 방금 받은 상품에 대해 말한 내용을 분류하시오.

        Constraint:
        반드시 다음 세 가지 중 하나만 대답하시오.
        긍정
        부정
        없음

        파손, 불량, 오배송, 구성품 누락, 상함, 불만을 말하면 부정으로 대답하시오.
        상품과 관계없는 내용이면 없음으로 대답하시오.
        다른 설명이나 문장을 추가하지 마시오.

        Output Format:
        한 단어.
        """,
        f"구매자 발화: {speech}",
        LABEL_TOKENS,
        0.0,
    )

    label = answer.strip(". ")

    if label not in ("긍정", "부정", "없음"):
        return "없음"

    return label


# 구매자 발화 → 별점 1 ~ 5
def rate_review(speech):
    answer = ask_gemma(
        """
        Instruction:
        구매자가 말한 내용을 바탕으로 만족도를 평가하시오.

        Constraint:
        반드시 1부터 5 사이의 숫자 하나만 대답하시오.
        5는 매우 만족, 1은 매우 불만족입니다.
        구매자의 표현만 보고 판단하시오.
        다른 설명이나 문장을 추가하지 마시오.

        Output Format:
        숫자 하나.
        """,
        f"구매자 발화: {speech}",
        LABEL_TOKENS,
        0.0,
    )

    answer = answer.strip(". ")

    if answer not in ("1", "2", "3", "4", "5"):
        return None

    return int(answer)


# 저장 상태와 클레임 후보 판정
def decide_status(label, rating, review, min_chars):
    if label == "없음" or review == "":
        return "photo_only", False

    claim_candidate = label == "부정" or (rating is not None and rating <= CLAIM_RATING)

    if len(review) < min_chars:
        return "draft_short", claim_candidate

    return "draft", claim_candidate


# 판정 결과에 맞는 TTS 안내 문장 선택
def select_message(status, rating, claim_candidate, min_chars):
    if status == "photo_only":
        return "답변을 인식하지 못해서 사진만 저장했어요."
    if claim_candidate:
        return "상태가 좋지 않다고 기록했어요. 사진과 함께 클레임 후보로 저장했어요."
    if status == "draft_short":
        return f"리뷰 초안을 저장했지만 {min_chars}자보다 짧아서 적립 기준에 못 미칠 수 있어요."
    if rating is None:
        return "리뷰 초안을 저장했어요."
    return f"별점 {rating}점으로 리뷰 초안을 저장했어요."


# 사진과 같은 이름으로 리뷰 JSON 저장
def save_review(event, speech, label, rating, review, status, claim_candidate, min_chars):
    product = event["product"]

    record = {
        "event_id": event["event_id"],
        "product_name": product["name"],
        "platform": product["platform"],
        "category": product["category"],
        "detected_class": event["class"],
        "confidence": event["confidence"],
        "image_path": event["image_path"],
        "stt_text": speech,
        "label": label,
        "rating": rating,
        "review": review,
        "review_length": len(review),
        "min_chars": min_chars,
        "status": status,
        "claim_candidate": claim_candidate,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    review_path = f"{REVIEW_DIR}/{event['event_id']}.json"

    with open(review_path, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=4)

    return review_path


for path in (GEMMA_MODEL_PATH, WHISPER_PATH, WHISPER_MODEL, PIPER_PYTHON, PIPER_MODEL):
    if not os.path.exists(path):
        print(f"파일을 찾을 수 없습니다: {path}")
        exit()

os.makedirs(REVIEW_DIR, exist_ok=True)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


print("촬영 이벤트 입력 대기 중...")


# 파일이 마지막으로 수정된 시간 저장
if os.path.exists(EVENT_JSON_PATH):
    last_modified = os.path.getmtime(EVENT_JSON_PATH)
else:
    last_modified = 0


while True:
    # JSON 파일 존재 확인
    if os.path.exists(EVENT_JSON_PATH):
        current_modified = os.path.getmtime(EVENT_JSON_PATH)

        # JSON 파일이 수정되었는지 확인
        if current_modified != last_modified:
            last_modified = current_modified

            try:
                # 1. 촬영 이벤트 읽기
                with open(EVENT_JSON_PATH, "r", encoding="utf-8") as file:
                    event = json.load(file)

                product = event["product"]
                category = product["category"]
                min_chars = MIN_CHARS[product["platform"]]

                print(f"\n[촬영 이벤트] {event['event_id']} → {product['name']} ({product['platform']}, 최소 {min_chars}자)")

                # 2. 카테고리별 질문 → 음성 입력
                text_to_speech(QUESTIONS[category].format(name=product["name"]))

                print(f"말씀해 주세요. ({RECORD_SECONDS}초)")
                speech = speech_to_text()
                print(f"\n[User] {speech}")

                label = "없음"
                rating = None
                review = ""

                if speech == "":
                    print("음성이 인식되지 않았습니다.")
                else:
                    # 3. 리뷰 작성
                    review = write_review(product, speech)
                    print(f"\n[Gemma 리뷰] {review}")

                    # 4. 최소 글자 수보다 짧으면 추가 질문 후 다시 작성
                    if len(review) < min_chars:
                        text_to_speech(FOLLOW_UPS[category])

                        print(f"말씀해 주세요. ({RECORD_SECONDS}초)")
                        more_speech = speech_to_text()
                        print(f"\n[User] {more_speech}")

                        if more_speech != "":
                            speech = speech + " " + more_speech
                            review = write_review(product, speech)
                            print(f"\n[Gemma 리뷰] {review}")

                    # 5. 라벨 분류, 별점
                    label = classify_label(speech)
                    rating = rate_review(speech)

                # 6. 코드 판정 → 리뷰 JSON 저장 → 결과 안내
                status, claim_candidate = decide_status(label, rating, review, min_chars)

                review_path = save_review(event, speech, label, rating, review, status, claim_candidate, min_chars)

                print("\n[판정]")
                print(f"label={label}, rating={rating}, length={len(review)}/{min_chars}, status={status}, claim_candidate={claim_candidate}")
                print(f"저장 완료: {review_path}")

                text_to_speech(select_message(status, rating, claim_candidate, min_chars))

            except Exception as error:
                print(f"오류 발생: {error}")


    # CPU를 불필요하게 계속 사용하지 않도록 잠시 대기
    time.sleep(0.1)
