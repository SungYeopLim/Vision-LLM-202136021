# Vision-LLM-202136021

**Physical AI: Vision-LLM 멀티모달 시스템 실습** — 202136021 임성엽

Jetson Orin Nano에서 Computer Vision, 딥러닝, YOLO 객체 탐지, LLM(Gemma)을 차례로 실습하고, 마지막에 이를 하나로 묶은 온디바이스 멀티모달 시스템을 만든 저장소입니다.

<br>

## 최종 프로젝트: 사진 리뷰 자동화 시스템

택배로 받은 상품을 카메라 앞 촬영 영역에 올려두면 YOLO가 상품을 인식해 사진을 저장하고, 음성으로 상품 상태를 말하면 Gemma가 답변과 사진을 바탕으로 리뷰 초안·라벨·별점을 만듭니다. 영상, 음성, LLM 처리를 모두 Jetson 안에서 처리해 외부 서버로 데이터가 나가지 않습니다.

```
카메라 → YOLO11n (TensorRT INT8) → 촬영 영역 2초 유지 → 사진 저장 + 이벤트 JSON
      → TTS 질문 (Piper) → STT 녹음 (whisper.cpp) → Gemma E2B (답변 + 사진) → 리뷰 JSON 저장
```

- 설명, 설치, 실행 방법: [`Final_Project/README.md`](Final_Project/README.md)
- 발표 자료: [`Final_Project/Final_Project_presentation.pdf`](Final_Project/Final_Project_presentation.pdf)
- 시연 영상: [`Final_Project/시연영상_임성엽.mp4`](Final_Project/시연영상_임성엽.mp4)

<br>

## 저장소 구성

```
Vision-LLM-202136021/
├── 00 ~ 06 *.ipynb      실습 노트북
├── 261010/              Assignment 01
├── 261017/              Assignment 02
├── 261025/              Assignment 03
├── Project01/           Section Project (손가락 + 탁구공 계산기)
├── Final_Project/       최종 프로젝트 (사진 리뷰 자동화 시스템)
├── experiments/         최종 프로젝트 전 단계에서 만든 개별 기능 테스트 스크립트
└── src/                 실습용 이미지, 모델·데이터셋 폴더 (모델과 데이터셋은 저장소에 포함하지 않음)
```

### 실습 노트북

| 파일 | 내용 |
|---|---|
| `00_Initial-Setup.ipynb` | Jetson 초기 설정 (한글, VS Code, 가상 환경, Git) |
| `01_Linux-and-Python.ipynb` | Linux CLI, Python 기초 문법 (`맹구_개인정보.json/.yaml`은 이 노트북의 파일 입출력 실습 결과) |
| `02_Computer-Vision.ipynb` | 이미지 조정, 색공간 변환, 필터링 등 OpenCV 기초, MediaPipe |
| `03_DL-and-GPU.ipynb` | 퍼셉트론, 다층 신경망, 활성화 함수, GPU 가속 학습 |
| `04_DL-Object-Detection.ipynb` | Bounding Box, IoU, NMS, YOLO 탐지와 TensorRT 변환 |
| `05-LLM_and_Gemma.ipynb` | Hugging Face, Access Token 관리, Gemma 텍스트 생성 |
| `06-Vision_LLM_Multimodal_Systems.ipynb` | YOLO 탐지 결과 → JSON → 자연어 → Gemma로 잇는 Vision-LLM 파이프라인 |
| `[ANSWER]_06-Vision_LLM_Multimodal_Systems.ipynb` | 06 노트북의 실습 문제를 직접 채운 버전 |

### 과제와 프로젝트

| 폴더 | 내용 |
|---|---|
| [`261010`](261010/README.md) | Assignment 01: Python 기초 (형 변환, List Comprehension, NumPy) |
| [`261017`](261017/README.md) | Assignment 02: 농구공 이미지 LAB color segmentation, CLAHE, Morphological operation |
| [`261025`](261025/README.md) | Assignment 03: NumPy/CuPy CPU·GPU 비교, MNIST·CIFAR-10 학습, IoU·NMS 구현 |
| [`Project01`](Project01/README.md) | Section Project: MediaPipe 손가락 개수 + Kalman Filter 탁구공 추적으로 만든 실시간 계산기 |
| [`Final_Project`](Final_Project/README.md) | 최종 프로젝트: 사진 리뷰 자동화 시스템 |

### experiments/

최종 프로젝트를 만들기 전에 기능을 하나씩 확인한 스크립트입니다. 경로가 저장소 최상위 폴더 기준이므로 최상위 폴더에서 `python3 experiments/<파일>.py`로 실행합니다.

| 파일 | 내용 |
|---|---|
| `kalman_filter_test.py` | Kalman Filter 물체 추적 테스트 |
| `mediapipe_hand_test.py` | MediaPipe 손 인식 테스트 |
| `yolo_pt_test.py` | YOLO11n `.pt` 모델 탐지 속도 테스트 |
| `yolo_int8_engine_test.py` | YOLO11n TensorRT INT8 엔진 탐지 속도 테스트 |
| `stt_tts.py` | whisper.cpp STT, Piper TTS 테스트 |
| `voice_gemma.py` | 음성 입력 → Gemma 응답 → 음성 출력 테스트 |

<br>

## 실행 환경

| 항목 | 내용 |
|---|---|
| 보드 | Jetson Orin Nano (MAXN SUPER) |
| OS | JetPack 6.2 (L4T R36.5), CUDA 12.6, TensorRT 10.3, Python 3.10 |
| 주요 라이브러리 | OpenCV, MediaPipe, PyTorch, Ultralytics YOLO, llama-cpp-python, whisper.cpp, Piper |

과제 01은 Windows(Python 3.14)에서 실행했고, 나머지는 Jetson에서 실행했습니다. 폴더별 실행 방법은 각 폴더의 README에 있습니다.

### 저장소에 포함하지 않은 파일

모델과 데이터셋은 용량이 크고 다시 받을 수 있어 저장소에 넣지 않았습니다 (`.gitignore`).

| 파일 | 받는 방법 |
|---|---|
| MNIST, CIFAR-10 데이터셋 | 노트북, `261025/assignment03.py` 실행 시 `src/datasets/`에 자동 다운로드 |
| MediaPipe 손 인식 모델 | `wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O src/models/MediaPipe/hand_landmarker.task` |
| MediaPipe 자세 인식 모델 | `wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -O src/models/MediaPipe/pose_landmarker_full.task` |
| YOLO, Gemma, Piper, whisper.cpp | [`Final_Project/README.md`의 설치 과정](Final_Project/README.md#설치) 참고 |
