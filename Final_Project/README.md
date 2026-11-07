# Final Project - 사진 리뷰 자동화 시스템 (가칭)

택배로 받은 상품을 카메라 앞 촬영 영역에 올려두면 상품 사진이 자동으로 저장되고, 음성으로 상품 상태를 말하면 리뷰 초안이 만들어지는 시스템입니다.
YOLO로 상품을 탐지하고, 촬영 영역 안에 2초 동안 있으면 촬영 이벤트가 발생합니다. 이후 TTS로 상품 상태를 묻고, STT로 받은 답변을 Gemma가 리뷰 문장, 라벨, 별점으로 정리하며, 저장 상태와 클레임 후보 판정은 코드가 합니다.

<br>

## 파일 구조

- **vision.py**: 카메라 입력, YOLO 탐지, 촬영 영역 및 유지 시간 판정, 상품 사진 저장, 촬영 이벤트 JSON 저장
- **assistant.py**: 촬영 이벤트 JSON 변경 감지, TTS 질문, STT 녹음 및 변환, Gemma 리뷰 작성·라벨 분류·별점, 코드 판정, 리뷰 JSON 저장, TTS 결과 안내
- **config/order_items.json**: YOLO 클래스와 주문 상품(상품명, 구매처, 카테고리) 연결 정보
- **config/platforms.json**: 구매처별 리뷰 최소 글자 수
- **config/tts_messages.json**: 카테고리별 질문, 추가 질문, 결과 안내 멘트
- **README.md**: 프로젝트 설명 문서

```
Vision-LLM-202136021/
├── src/
│   ├── models/
│   │   ├── YOLO/yolo11n_int8.engine
│   │   ├── Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
│   │   └── Piper/ko_KR-kss-medium.onnx
│   ├── audio/                   (녹음, TTS 임시 파일)
│   ├── json/                    (review_event.json, 실행 시 자동 생성)
│   └── output/reviews/          (결과 사진, 리뷰 JSON, 실행 시 자동 생성)
├── whisper.cpp/
├── .piper_venv/
└── Final_Project/
    ├── vision.py
    ├── assistant.py
    ├── config/
    │   ├── order_items.json
    │   ├── platforms.json
    │   └── tts_messages.json
    └── README.md
```

모든 경로는 `Final_Project` 폴더 기준 상대 경로(`../src/...`, `../whisper.cpp/...`, `../.piper_venv/...`)입니다. 따라서 `Final_Project` 폴더는 `src`, `whisper.cpp`, `.piper_venv`가 있는 폴더 바로 아래에 있어야 합니다.

<br>

## 동작 흐름

```
[vision.py]
카메라 → YOLO 탐지 → 주문 목록에 있는 클래스만 선택 → 촬영 영역 안에 2초 유지
→ 상품 사진 저장 (../src/output/reviews/{시각}_{클래스}.jpg)
→ 촬영 이벤트 저장 (../src/json/review_event.json, 임시 파일 작성 후 os.replace)

[assistant.py]
review_event.json 수정 시간 변경 감지 → 주문 목록에서 상품명/구매처/카테고리 조회
→ TTS 카테고리별 질문 → STT 녹음 → Gemma 리뷰 작성
→ (리뷰가 구매처 최소 글자 수보다 짧으면) TTS 추가 질문 → STT 녹음 → 두 답변을 합쳐 리뷰 다시 작성
→ Gemma 라벨 분류, 별점 → 코드 판정 → 리뷰 JSON 저장 (../src/output/reviews/{시각}_{클래스}.json) → TTS 결과 안내
```

두 프로그램은 독립된 프로세스로 실행되고, JSON 파일로만 정보를 주고받습니다. 영상 처리와 음성·LLM 처리를 분리해서, Gemma가 응답을 만드는 동안에도 카메라와 YOLO는 멈추지 않습니다. 파일은 임시 파일에 먼저 쓴 뒤 `os.replace`로 교체해, 쓰는 도중의 파일을 읽는 일이 없도록 했습니다.

### 트리거 조건 (vision.py)

| 조건 | 내용 |
|---|---|
| 대상 클래스 | `config/order_items.json`에 등록된 클래스만 대상 |
| 촬영 영역 | bbox 중심이 화면 중앙 사각형(`ZONE`) 안에 있음 |
| 유지 시간 | 같은 클래스가 촬영 영역 안에 `HOLD_SECONDS`(2초) 동안 있으면 촬영 |
| 탐지 끊김 | YOLO가 `MAX_LOST_FRAMES`(5프레임) 이하로 놓친 경우는 유지 시간을 이어서 계산 |
| 다음 촬영 | 촬영 후에는 촬영 영역이 `RESET_LOST_FRAMES`(30프레임, 약 1초)보다 오래 비어야 다음 촬영 가능 |
| 여러 물체 | 촬영 영역 안 대상 중 bbox 면적이 가장 큰 물체 하나만 판정 |

### Gemma 역할 (assistant.py)

Gemma E2B는 작은 모델이라 한 번에 여러 가지를 출력하게 하면 형식이 흔들릴 수 있습니다. 그래서 호출마다 한 가지 일만 맡겼습니다.

| 호출 | 출력 | `temperature` |
|---|---|---|
| 리뷰 작성 | 사용자 답변을 정리한 한국어 리뷰 문장 | 0.7 |
| 라벨 분류 | `positive`, `negative`, `none` 중 한 단어 | 0.0 |
| 별점 | `1`~`5` 중 숫자 하나 | 0.0 |

라벨과 별점은 정해진 값 중 하나만 출력하도록 제한하고 `temperature`를 0으로 두어, 코드에서 값 비교만으로 판정할 수 있게 했습니다.

리뷰 작성 프롬프트에는 "사용자가 말한 내용과 상품명만 사용하고, 말하지 않은 칭찬이나 특징을 추가하지 말 것"을 제약으로 넣었습니다. 리뷰 길이가 부족할 때 Gemma가 내용을 지어내서 늘리지 않도록, 길이는 사용자에게 추가 질문을 해서 채웁니다.

### 코드 판정 (assistant.py)

| 상황 | 코드 판정 | 저장 상태 (`status`) | TTS 안내 |
|---|---|---|---|
| STT 결과가 비어 있거나 `[BLANK_AUDIO]` 같은 표시 | Gemma를 호출하지 않음 | `photo_only` | 사진만 저장 |
| 라벨이 세 단어 밖이거나 `none` | `none` 처리 | `photo_only` | 사진만 저장 |
| 별점이 `1`~`5`가 아님 | 별점 `null` | 아래 기준대로 | 별점 없이 저장 |
| 라벨이 `negative` 또는 별점 2 이하 | `claim_candidate: true` | 아래 기준대로 | 클레임 후보로 저장 |
| 첫 답변으로 만든 리뷰가 최소 글자 수 미만 | 추가 질문 1번 | - | 카테고리별 추가 질문 |
| 추가 답변 후에도 최소 글자 수 미만 | 짧은 초안으로 저장 | `draft_short` | 적립 기준에 못 미칠 수 있다고 안내 |
| 그 외 | 리뷰 초안 저장 | `draft` | 별점과 함께 저장 |

<br>

## 실행 방식

### 실행 환경

| 항목 | 내용 |
|---|---|
| 보드 | Jetson Orin Nano (전원 모드 MAXN SUPER) |
| OS | JetPack 6.2 (L4T R36.5), CUDA 12.6, TensorRT 10.3, Python 3.10 |
| 장치 | USB 카메라, USB 마이크, 스피커 (3.5mm 또는 USB) |
| 시연 물건 | 음료 페트병(`bottle`), 스마트폰(`cell phone`), 마우스(`mouse`) 등 `config/order_items.json`에 등록된 물건 |

L4T R36.4.x에서는 llama.cpp의 CUDA 메모리 할당 오류가 발생할 수 있어 R36.5로 업데이트한 환경에서 실행했습니다. 현재 버전은 아래 명령으로 확인합니다.

```bash
cat /etc/nv_tegra_release | head -n 1
```

### 설치

모든 명령은 저장소 최상위 폴더(`Vision-LLM-202136021/`)에서 실행합니다. 모델 파일, 가상 환경, whisper.cpp는 용량이 크고 장치마다 새로 만들어야 하므로 저장소에 포함하지 않았습니다.

#### 1. 저장소 내려받기

```bash
git clone https://github.com/SungYeopLim/Vision-LLM-202136021.git
cd Vision-LLM-202136021
```

#### 2. 시스템 패키지

`arecord`, `aplay`(녹음, 재생), `pasuspender`(녹음 중 PulseAudio 일시 정지), `cmake`(whisper.cpp 빌드)를 설치합니다.

```bash
sudo apt update
sudo apt install -y python3-venv cmake alsa-utils pulseaudio-utils
```

#### 3. 메인 가상 환경 (`.venv`)

`--system-site-packages`는 JetPack에 기본 설치된 OpenCV(GStreamer 지원)와 TensorRT를 가상 환경에서도 그대로 쓰기 위한 옵션입니다.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

PyTorch (Jetson용 빌드):

```bash
pip install torch==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip install torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip install "sympy>=1.13.3" --no-deps
```

YOLO (Ultralytics)와 TensorRT 변환에 필요한 패키지:

```bash
pip install "numpy==1.23.0"
pip install --no-deps ultralytics
pip install --no-deps "typing_extensions>=4.10.0,<5"
pip install --no-deps "filelock>=3.16.1,<4"
pip install --no-deps onnx==1.16.0 onnxruntime==1.23.2 onnxslim==0.1.95 ml_dtypes==0.5.4
pip install --no-deps coloredlogs==15.0.1 humanfriendly==10.0 protobuf==3.20.3
pip install pyyaml
```

Gemma 실행용 `llama-cpp-python` (CUDA 빌드, 수십 분 걸릴 수 있음)과 모델 다운로드용 패키지:

```bash
export PATH=/usr/local/cuda/bin:$PATH
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
pip install --no-cache-dir llama-cpp-python
pip install huggingface_hub python-dotenv
```

#### 4. YOLO TensorRT INT8 엔진

TensorRT 엔진은 만든 장치에서만 동작하므로 실행할 Jetson에서 직접 만들어야 합니다.

1) YOLO11n 가중치를 내려받습니다.

```bash
mkdir -p src/models/YOLO
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O src/models/YOLO/yolo11n.pt
```

2) INT8 Calibration용 이미지를 준비합니다. 실제 사용 환경(책상 위 촬영 영역)을 찍은 카메라 이미지 약 500장을 `src/datasets/calibration/images/`에 저장하고, 같은 이름의 빈 `.txt` 파일을 `src/datasets/calibration/labels/`에 만듭니다. 그 다음 `src/datasets/calibration/calibration.yaml`을 아래 형식으로 작성합니다. (`path`는 절대 경로, `names`는 COCO 80개 클래스)

```yaml
path: /home/<사용자>/Vision-LLM-202136021/src/datasets/calibration
train: images
val: images
names:
  0: person
  1: bicycle
  ...
  79: toothbrush
```

`names`는 아래 명령으로 출력해서 붙여 넣을 수 있습니다.

```bash
python3 -c "from ultralytics import YOLO; import yaml; print(yaml.safe_dump({'names': YOLO('src/models/YOLO/yolo11n.pt').names}, allow_unicode=True))"
```

3) INT8 엔진으로 변환합니다. (수 분 걸림)

```bash
python3 -c "
from pathlib import Path
from ultralytics import YOLO
path = YOLO('src/models/YOLO/yolo11n.pt').export(format='engine', imgsz=640, quantize=8, data='src/datasets/calibration/calibration.yaml', batch=1, dynamic=False, device=0, nms=False)
Path(path).replace('src/models/YOLO/yolo11n_int8.engine')
"
```

#### 5. Gemma 모델

Hugging Face Access Token을 저장소 최상위 폴더의 `.env` 파일에 저장합니다. (`.env`는 저장소에 올리지 않습니다)

```bash
echo "HF_TOKEN=<본인 토큰>" > .env
```

```bash
python3 -c "
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
load_dotenv()
hf_hub_download(repo_id='bartowski/google_gemma-4-E2B-it-GGUF', filename='google_gemma-4-E2B-it-Q4_K_M.gguf', token=os.getenv('HF_TOKEN'), local_dir='src/models/Gemma4')
"
```

#### 6. STT (whisper.cpp)

저장소 최상위 폴더에 whisper.cpp를 내려받아 CPU 버전으로 빌드하고, base 모델을 받습니다.

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build-cpu
cmake --build build-cpu --target whisper-cli -j4 --config Release
sh ./models/download-ggml-model.sh base
cd ..
```

#### 7. TTS (Piper)

Piper는 메인 가상 환경과 패키지 충돌을 피하기 위해 별도 가상 환경(`.piper_venv`)에 설치합니다.

```bash
deactivate
python3 -m venv .piper_venv
source .piper_venv/bin/activate
pip install piper-tts
mkdir -p src/models/Piper
python3 -m piper.download_voices ko_KR-kss-medium --data-dir src/models/Piper
deactivate
```

설치가 끝나면 저장소 최상위 폴더는 아래 구조가 됩니다.

```
Vision-LLM-202136021/
├── .venv/
├── .piper_venv/
├── .env
├── whisper.cpp/
│   ├── build-cpu/bin/whisper-cli
│   └── models/ggml-base.bin
├── src/
│   ├── models/YOLO/yolo11n_int8.engine
│   ├── models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
│   └── models/Piper/ko_KR-kss-medium.onnx
└── Final_Project/
```

#### 8. 장치 확인

마이크 card 번호를 확인하고 `assistant.py`의 `MIC_DEVICE`를 맞춥니다. (기본 `plughw:3,0`, `card 3, device 0`이면 그대로 사용)

```bash
arecord -l
```

카메라 장치 번호를 확인하고 `vision.py`의 `CAMERA_DEVICE`를 맞춥니다. (기본 `/dev/video0`)

```bash
ls /dev/video*
```

### 실행

YOLO와 Gemma를 함께 GPU에 올리므로, 다른 Python 프로그램이나 Jupyter 커널이 모델을 잡고 있지 않은 상태에서 터미널 2개로 실행합니다.

1. 터미널 1에서 assistant를 먼저 실행합니다. Gemma 로딩이 끝나고 `Waiting for capture event...`가 출력될 때까지 기다립니다.

```bash
cd Final_Project
source ../.venv/bin/activate
python3 assistant.py
```

2. 터미널 2에서 vision을 실행합니다.

```bash
cd Final_Project
source ../.venv/bin/activate
python3 vision.py
```

3. 물건을 화면 중앙 사각형 안에 올려두고 2초 기다립니다. 질문이 나오면 `RECORD_SECONDS`(8초) 안에 상품 상태를 말합니다.

### 조작 방법

| 입력 | 동작 |
|---|---|
| 물건을 촬영 영역에 2초 유지 | 자동 촬영 (영역 아래 노란 막대가 차면 촬영) |
| 촬영 후 물건 치우기 | 화면의 `Saved. Clear the zone for next item`이 사라지면 다음 물건 촬영 가능 |
| `l` 키 (vision 창) | 촬영 영역 안 물체를 즉시 촬영 |
| `q` 키 (vision 창) | vision 종료 |
| `Ctrl + C` (assistant 터미널) | assistant 종료 |

### 조정 파라미터 (코드 상단)

| 파일 | 변수 | 의미 | 조정 시점 |
|---|---|---|---|
| vision.py | `CAMERA_DEVICE` | 카메라 장치 | 카메라가 열리지 않을 때 |
| vision.py | `CONFIDENCE` | YOLO confidence 기준 (기본 0.4) | 물건이 안 잡히면 내리고, 오탐이 많으면 올림 |
| vision.py | `ZONE` | 촬영 영역 (x1, y1, x2, y2) | 카메라 위치에 맞게 |
| vision.py | `HOLD_SECONDS` | 촬영까지 유지 시간 | 너무 느리거나 들고 있는 중에 촬영될 때 |
| vision.py | `MAX_LOST_FRAMES` | 탐지가 끊겨도 유지 시간을 이어갈 프레임 수 | 막대가 자꾸 처음부터 다시 찰 때 |
| vision.py | `RESET_LOST_FRAMES` | 다음 촬영을 허용할 빈 영역 프레임 수 | 물건을 치우기 전에 다시 촬영될 때 |
| assistant.py | `MIC_DEVICE` | 마이크 장치 | 녹음이 안 될 때 |
| assistant.py | `RECORD_SECONDS` | 녹음 시간 (기본 8초) | 답변이 잘리거나 대기가 길 때 |
| assistant.py | `CLAIM_RATING` | 클레임 후보로 볼 별점 기준 (기본 2 이하) | 판정 기준 변경 시 |
| assistant.py | `REVIEW_TOKENS` | 리뷰 문장 최대 토큰 | 리뷰가 중간에 끊길 때 |

### 설정 파일 수정

시연할 물건은 `config/order_items.json`에서 바꿉니다. 키는 YOLO(COCO) 클래스 이름이고, `platform`은 `platforms.json`의 키, `category`는 `tts_messages.json`의 질문 종류(`food`, `living`, `electronics`, `fashion`)와 맞춥니다.

```json
"bottle": {
    "name": "탄산음료 500ml",
    "platform": "coupang",
    "category": "food"
}
```

구매처별 최소 글자 수는 `config/platforms.json`에서 바꿉니다. 현재 값(30자)은 예시이며, 실제 기준은 각 쇼핑몰의 리뷰 작성 화면이나 적립금 안내에서 확인해 맞춰야 합니다.

```json
"coupang": {
    "min_chars": 30
}
```

<br>

## 결과 예시

`../src/output/reviews/`에 사진과 리뷰 JSON이 같은 이름으로 한 쌍씩 저장됩니다.

```json
{
    "event_id": "20261114_103512_bottle",
    "product_name": "탄산음료 500ml",
    "platform": "coupang",
    "category": "food",
    "detected_class": "bottle",
    "confidence": 0.87,
    "image_path": "../src/output/reviews/20261114_103512_bottle.jpg",
    "stt_text": "탄산이 세고 시원해요 근데 병이 조금 찌그러져 왔어요 양은 딱 한 번 마시기 좋아요",
    "label": "negative",
    "rating": 3,
    "review": "탄산이 세고 시원했습니다. 다만 병이 조금 찌그러진 상태로 배송되었습니다. 양은 한 번에 마시기 좋았습니다.",
    "review_length": 61,
    "min_chars": 30,
    "status": "draft",
    "claim_candidate": true,
    "created_at": "2026-11-14 10:35:31"
}
```

- `stt_text`: 사용자가 실제로 말한 원문 (추가 질문에 답한 경우 두 답변을 합친 것). 리뷰가 이 내용을 정리한 것인지 확인할 수 있도록 함께 저장합니다.
- `status`: `draft`(초안), `draft_short`(최소 글자 수 미달 초안), `photo_only`(사진만 저장). 바로 업로드하지 않고 사용자가 확인한 뒤 올리는 초안이라는 의미입니다.

<br>

## 창의성 및 확장성

### 아이디어 설명

부모님이 식재료를 원룸으로 배송시켜 주시면서 사진 리뷰를 남기라고 하시는데, 상자에서 하나씩 꺼내고, 휴대폰으로 찍고, 앱에서 리뷰를 타이핑하는 과정이 번거로웠습니다. 사진 리뷰에 적립금을 주는 쇼핑몰이 많지만(네이버, 쿠팡, 무신사, 컬리 등), 이 과정이 귀찮고 최소 글자 수 같은 조건도 있어서 놓치는 경우가 많습니다.

그래서 상자를 풀면서 물건을 카메라 앞에 올려두기만 하면 사진이 저장되고, 손이 바쁜 상태에서도 말로 리뷰를 남길 수 있는 구조를 만들었습니다. 여러 쇼핑몰에서 산 물건을 한 곳에서 처리하고, 결과는 주문 1건당 리뷰 1건으로 해당 구매처 정보와 함께 저장합니다. 구매처마다 다른 최소 글자 수를 코드가 확인하고, 부족하면 내용을 지어내지 않고 사용자에게 한 번 더 물어봅니다.

### 기존 서비스와 차이점

| 기존 서비스 | 내용 | 차이점 |
|---|---|---|
| [Spokk](https://apps.shopify.com/spokk-voice-reviews), [Parlata](https://apps.shopify.com/parlata) (Shopify 앱) | 고객이 음성으로 말하면 AI가 텍스트 리뷰로 정리 | 휴대폰으로 앱을 열고 직접 녹음, 촬영해야 함 |
| 당근 '말로 쓰기' | AI 질문에 음성으로 답하면 후기로 정리 | 특정 서비스 안에서만 동작 |
| 쿠팡, 네이버 AI 리뷰 요약 | 작성된 리뷰를 요약 | 리뷰 작성이 아니라 읽는 쪽 기능 |
| [Vocalizer](https://arxiv.org/html/2412.05445v2) (2024 논문) | Whisper + GPT-4로 음성 식당 리뷰 정리 | 사용자들이 진정성 저하를 우려함 → 이 프로젝트는 말한 내용만 정리하고 STT 원문을 함께 저장 |
| [Albertsons Intelligent Quality Control](https://www.freshfruitportal.com/news/2026/05/20/albertsons-ia-tool/) | 물류센터에서 비전 AI로 신선식품 품질 검수 | 유통사 내부용. 이 프로젝트는 소비자가 받는 시점의 상태를 기록 |

이 프로젝트의 차이점은 아래 네 가지입니다.

1. 휴대폰을 들지 않고 카메라 트리거로 촬영합니다.
2. 여러 쇼핑몰의 주문을 한 곳에서 처리하고, 구매처별 최소 글자 수를 코드가 확인합니다.
3. 같은 답변에서 긍정/부정을 나눠, 부정이면 사진과 함께 클레임 후보로 표시합니다.
4. Jetson에서 모든 처리를 하므로 영상과 음성이 외부 서버로 나가지 않습니다.

### 구현한 내용

| 구성 | 구현 내용 |
|---|---|
| 프로세스 구조 | vision과 assistant를 독립 프로세스로 분리, 촬영 이벤트를 JSON 파일로 전달하고 파일 수정 시간으로 새 이벤트 감지 |
| 상품 탐지 | YOLO11n TensorRT INT8 엔진으로 실시간 탐지, 주문 목록에 있는 클래스만 판정 대상 |
| 촬영 트리거 | 촬영 영역 안 좌표 판정, 같은 물건 2초 유지 시 1회 촬영, 탐지 끊김 허용(5프레임), 영역이 비면 다음 촬영 허용(30프레임) |
| 사진 저장 | 선택한 물체의 bbox 영역을 잘라 이벤트 ID 이름으로 저장 |
| 주문 매핑 | 탐지 클래스를 상품명, 구매처, 카테고리로 변환 |
| 음성 질문 | 카테고리별 질문(식품: 신선도·포장, 생활: 마감·색상, 전자기기: 외관·구성품, 패션: 색상·소재) |
| STT / TTS | whisper.cpp로 한국어 음성 인식, Piper로 한국어 음성 출력 |
| 리뷰 생성 | Gemma가 사용자 답변만으로 리뷰 작성, 라벨과 별점은 정해진 값 하나로만 출력 |
| 최소 글자 수 | 구매처 기준과 리뷰 길이를 비교해, 부족하면 추가 질문 1번 후 다시 작성 |
| 판정 | 라벨·별점 범위 확인, 클레임 후보 판정, 저장 상태(`draft`, `draft_short`, `photo_only`) 결정 |
| 결과 저장 | 사진과 리뷰 JSON을 같은 이름으로 저장, STT 원문을 함께 기록 |

### 시연 대체 대응표

수업에서 쓰는 YOLO 모델은 COCO 80개 클래스만 인식하므로, 실제 상품은 비슷한 클래스로 대체해 시연합니다. 트리거와 판정 로직은 동일하고 `order_items.json`의 연결만 다릅니다.

| 실제 상품 | 시연 클래스 | 비고 |
|---|---|---|
| 음료, 우유 | `bottle` | |
| 컵, 텀블러, 그릇류 | `cup` | |
| 과일 | `orange`, `apple`, `banana` | 대체 없이 인식 가능 |
| 전자기기 | `mouse`, `keyboard`, `cell phone` | 대체 없이 인식 가능 |
| 가방, 잡화 | `backpack`, `handbag`, `umbrella` | 대체 없이 인식 가능 |
| 의류, 신발, 화장품 등 | (COCO에 없음) | 커스텀 학습 필요 → 확장성 |

### 확장성

아래 표는 현재 구현의 범위와 한계, 실제 서비스로 확장할 때의 방법입니다.

| 구분 | 현재 코드 | 한계 | 확장 방법 |
|---|---|---|---|
| 촬영 판정 | 같은 클래스가 영역 안에 2초 있으면 촬영 | 들고 흔드는 중에 찍혀 사진이 흐릴 수 있음 | bbox 이동량 확인, 이미지 선명도(Laplacian 분산) 검사로 흔들린 사진 제외 |
| 여러 상품 | 영역 안에서 가장 큰 물체 1개, 촬영 후 영역을 비워야 다음 촬영 | 여러 상품을 한 번에 올리면 1개만 처리 | YOLO 객체 추적(track)으로 물체마다 ID를 붙여 상품별로 처리 |
| 상품 매칭 | `order_items.json`에 클래스 1개 = 상품 1개로 직접 작성 | 같은 클래스 상품 2개(예: 음료 2종)를 구분 못 함 | 쇼핑몰 주문 내역 연동, 송장·상품 바코드 인식 |
| 인식 범위 | COCO 80개 클래스 | 의류, 화장품 등 인식 불가 | 커스텀 데이터셋으로 YOLO 학습 후 int8 engine으로 교체 (`order_items.json` 키만 변경) |
| 사진 | bbox crop 1장 | 쇼핑몰이 요구하는 장수나 구도(예: 착용 사진)와 다를 수 있음 | 여러 각도 연속 촬영, 쇼핑몰별 사진 규격에 맞춰 저장 |
| 상태 판단 | 사용자 말로만 판단 | 사용자가 못 본 파손은 기록 안 됨 | 저장한 사진을 이미지 입력(mmproj)으로 Gemma에 전달해 파손, 변색 판단 보조 |
| 녹음 | 8초 고정, 추가 질문 1번 | 답변이 길면 잘림 | 무음 감지(VAD)로 말이 끝날 때까지 녹음 |
| 최소 글자 수 | `platforms.json`에 직접 입력 | 쇼핑몰 정책이 바뀌면 직접 수정 | 쇼핑몰 API에서 리뷰 정책(글자 수, 사진 조건, 적립금)을 받아와 적용 |
| 이벤트 처리 | `review_event.json` 1개 파일, 가장 최근 촬영만 처리 | 대화 중 여러 번 촬영되면 중간 건은 리뷰 없이 사진만 남음 | 이벤트 파일을 촬영마다 따로 만들고 순서대로 처리하는 대기열 |
| 리뷰 업로드 | 로컬 JSON 초안 저장 | 쇼핑몰에 직접 올라가지 않음 | 쇼핑몰 API 연동, 사용자가 초안을 확인한 뒤 업로드 (`draft → uploaded`) |
| 클레임 | `claim_candidate: true` 표시만 | 교환, 환불 요청은 직접 해야 함 | 표시된 건을 사진과 함께 교환, 환불, 재배송 요청으로 자동 접수 |
| 리뷰 기한 | 없음 | 적립금 기한을 놓칠 수 있음 | 쇼핑몰별 리뷰 작성 기한, 적립금 조건을 모아 기한 전에 알림 |
| 2차 리뷰 | 받는 순간의 리뷰만 | 맛, 내구성, 착용감 등 사용 후 내용 없음 | 일정 기간 뒤 다시 질문해 사용 후기 추가 |

실제 서비스로 확장할 때 바뀌는 부분과 그대로인 부분은 아래와 같습니다.

| 바뀌는 부분 | 그대로인 부분 |
|---|---|
| YOLO 모델 파일, `order_items.json`(주문 정보), `platforms.json`(쇼핑몰 정책), 결과 전달 대상(쇼핑몰 API) | 프로세스 분리 구조, 촬영 트리거 판정, STT → Gemma → 코드 판정 → TTS 흐름, 리뷰 JSON 형식 |

<br>

## 기타 참고 사항

- 상품명, 쇼핑몰 정책, TTS 멘트는 `config` 폴더의 JSON 파일로 분리했습니다. 시연 물건이나 멘트를 바꿀 때 코드를 수정하지 않아도 됩니다.
- Gemma 프롬프트는 영어로 작성하고, 리뷰 문장만 한국어로 출력하도록 했습니다. 화면 표시(`cv2.putText`)는 한글을 그릴 수 없어 영어로 표시합니다.
- Gemma가 라벨을 `Positive.`처럼 대문자나 마침표를 붙여 출력하는 경우가 있어, 앞뒤 마침표와 공백을 지우고 소문자로 바꾼 뒤 비교합니다.
- whisper는 무음일 때 `[BLANK_AUDIO]`를 출력하므로, `[`로 시작하는 결과는 답변 없음으로 처리합니다.
- Gemma 호출이 최대 4번(리뷰 작성 2번, 라벨, 별점)이라 답변 후 결과 안내까지 몇 초 걸릴 수 있습니다. 라벨과 별점은 출력 토큰을 5개로 제한했습니다.
- 받는 순간의 리뷰이므로 사용 후에야 알 수 있는 내용(맛, 내구성, 착용감 등)은 담기 어렵습니다.
