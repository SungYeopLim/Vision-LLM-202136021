# Section Project

손가락 개수로 숫자를 입력하고, 초록 탁구공을 화면 위 가상 버튼에 올려 연산자를 입력하는 실시간 계산기입니다.
키보드나 계산기 없이 손과 공만으로 입력 순서대로 계산합니다.

<br>

## 파일 구조

- **project01.py**: 카메라 입력, 초록 탁구공 검출 및 Kalman Filter 추적, MediaPipe 손 인식과 손가락 개수 계산, 가상 버튼 판정, 계산 로직, 화면 출력을 하나로 묶은 실행 파일
- **README.md**: 프로젝트 설명 문서

```
Vision-LLM-202136021/
└── Project01/
    ├── project01.py
    └── README.md
```

<br>

## 실행 방식

### 준비물

- 초록색 탁구공
- Jetson + USB 카메라(`/dev/video0`)
- 손 인식 모델 파일 `hand_landmarker.task` (아래 단계에서 다운로드)

### Jetson

1. 저장소를 내려받고 저장소 폴더로 이동합니다.

```bash
git clone https://github.com/SungYeopLim/Vision-LLM-202136021.git
cd Vision-LLM-202136021
```

2. 가상 환경을 만들고 활성화합니다. `--system-site-packages`는 JetPack에 기본 설치된 OpenCV(GStreamer 지원)를 가상 환경에서도 그대로 쓰기 위한 옵션입니다.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

3. MediaPipe를 설치합니다. `--no-deps`는 JetPack의 OpenCV와 NumPy를 덮어쓰지 않기 위한 옵션이며, 필요한 의존성은 두 번째 줄에서 따로 설치합니다.

```bash
pip install --no-deps mediapipe
pip install absl-py flatbuffers sounddevice
```

4. `Project01` 폴더로 이동한 뒤 손 인식 모델을 `src/models/MediaPipe/` 폴더에 다운로드합니다.

```bash
cd Project01
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O ../src/models/MediaPipe/hand_landmarker.task
```

5. 모델 경로가 `../src/...` 상대 경로이므로, 가상 환경을 활성화한 상태로 `Project01` 폴더 위치에서 아래 명령을 입력합니다.

```bash
python3 project01.py
```

### 조작 방법

| 입력 | 방법 |
|---|---|
| 숫자 0~9 | 양손의 펼친 손가락 개수 합을 1초 유지 (하단 노란 막대가 차면 입력) |
| 같은 숫자 연속 입력 (예: 11) | 1 입력 → 손을 화면 밖으로 내리거나 다른 개수로 바꿈 → 다시 1 |
| 연산자 `+ - * / sqrt x^2 <- C =` | 초록 공을 화면 위쪽 버튼 위에 1초 유지 (버튼 아래 막대가 차면 입력) |
| `m` 키 | 공 Binary Mask 창 켜기/끄기 (LAB 범위 조정용) |
| `e` 키 | CLAHE(Image Enhancement) 켜기/끄기 (화면 하단 `CLAHE: ON/OFF` 표시) |
| `q` 키 | 종료 |

- 공이 버튼 영역에 있거나 빠르게 움직이는 중에는 숫자 입력이 멈춥니다 (`Fingers: PAUSED`). 공을 옮기는 손의 손가락이 숫자로 잘못 입력되는 것을 막기 위함입니다.
- 숫자를 입력할 때는 공을 책상 위에 내려놓습니다.
- 0으로 나누기, 음수의 제곱근은 `FAIL!`이 표시되며, 다음 입력 시 초기화됩니다.

### 조정 파라미터 (코드 상단)

| 변수 | 의미 | 조정 시점 |
|---|---|---|
| `green_lower`, `green_upper` | 공 LAB 범위 | `m`으로 마스크를 보며 공만 흰색이 되도록 |
| `min_ball_area` | 공으로 인정할 최소 면적 | 작은 노이즈가 버튼을 누를 때 |
| `angle_threshold` | 손가락 펼침 판단 각도 (기본 160) | 손가락 개수가 자주 틀릴 때 |
| `thumb_ratio` | 엄지 펼침 판단 거리 비율 (기본 1.3) | 주먹에서 1이 나오면 올리고, 엄지를 폈는데 안 세어지면 내림 |
| `hold_seconds` | 입력 확정 시간 | 너무 느리거나 오입력이 많을 때 |
| `speed_threshold` | 숫자 입력을 멈출 공 속도 (px/frame) | 공을 옮길 때 숫자가 입력될 때 |
| `flip_code` | `1` 좌우 반전(거울), `0` 상하 반전, `None` 반전 없음 | 카메라 방향에 맞게 |
| `use_clahe` | 시작 시 CLAHE 적용 여부 (기본 `False`, 실행 중 `e` 키로 전환) | 조명이 어둡거나 고르지 않아 공 마스크가 끊길 때 |

<br>

## 창의성 및 확장성

### 아이디어 설명

손으로는 숫자를, 물체(공)로는 명령을 입력하는 두 입력 채널 인터페이스입니다.

손가락만으로 모든 입력을 하면 숫자와 연산자 제스처가 서로 헷갈리기 쉬워서, 연산자는 화면 위 가상 버튼과 실제 물체(탁구공)의 위치 관계로 입력하도록 분리했습니다. 결과적으로 신체(손)와 객체(공)를 동시에 인식하고, 객체의 위치와 속도로 신체 입력을 제어하는 구조가 됩니다.

### 구현한 내용

실습 파일(`02_Computer-Vision.ipynb`)에서 다룬 기법을 순서대로 연결했습니다.

| 단계 | 사용 기법 | 실습 출처 |
|---|---|---|
| 이미지 조정 | `cv2.flip` | 이미지 조정 |
| 색공간 변환 | BGR→LAB (공), BGR→RGB (MediaPipe) | 색공간 변환 |
| 필터링 | `cv2.GaussianBlur` (7x7) | 필터링 |
| Image Enhancement | LAB L 채널에 CLAHE 적용 (clipLimit 3.0, 8x8), `e` 키로 켜기/끄기 | Image Enhancement |
| Color Segmentation | LAB `cv2.inRange`로 초록 공 Binary Mask | Color Segmentation (농구공, 탁구공) |
| Morphological Operation | Opening 2회 → Closing 1회 | 오토바이·농구공 마스크 실습 |
| Contour 검출 | `findContours` → 가장 큰 contour → `minEnclosingCircle`, moments 무게중심 | 농구공 외접원/무게중심 실습 |
| 객체 추적 | `cv2.KalmanFilter` (상태 `[x, y, vx, vy]`) | 실시간 객체 추적 |
| 신체 인식 | MediaPipe HandLandmarker + 관절 각도 규칙으로 손가락 개수 (엄지는 거리 조건 추가) |  MediaPipe, Rule-based 손가락 개수 |

새로 작성한 부분은 아래 규칙 로직입니다.

- **버튼 판정**: Kalman Filter로 추정한 공 좌표가 버튼 사각형 안에 있는지 비교
- **1초 유지 확정**: 같은 값이 유지된 시간을 `time.time()`으로 재서 1초가 되면 한 번만 입력 (입력 후 잠금, 값이 바뀌면 해제)
- **숫자 입력 일시정지**: 공이 버튼 영역에 있거나, Kalman Filter 상태의 속도 `(vx, vy)` 크기가 기준보다 크면 숫자 입력 중지
- **가려짐 처리**: 공이 손에 잠깐 가려져도 Kalman Filter의 예측값으로 커서 유지. 단, 5프레임 넘게 안 보이면 버튼 입력에 사용하지 않고 30프레임이 넘으면 추적을 초기화 (예측값이 표류해서 버튼이 잘못 눌리는 것을 방지)
- **계산 로직**: 입력 순서대로 계산하는 방식으로 (`2 + 3 * 4 =` → 20), `=` 반복 시 마지막 연산 반복, `sqrt`, `x^2`는 현재 숫자에 즉시 적용

### 확장성

- 버튼 리스트(`button_labels`)에 항목과 `calculate()` 분기만 추가하면 연산자를 늘릴 수 있습니다.
- 공 대신 다른 색의 물체를 쓰려면 LAB 범위만 바꾸면 되고, 여러 색 물체를 쓰면 물체마다 다른 기능을 줄 수 있습니다.

<br>

## 기타 참고 사항

- 공 LAB 범위는 수업 중 검증한 탁구공 실시간 검출 코드의 값을 그대로 사용했습니다. 강의실 조명에 따라 조정이 필요할 수 있습니다.
- CLAHE는 L 채널(밝기)을 바꾸기 때문에, CLAHE 없이 맞춘 LAB 범위와 결과가 달라질 수 있습니다. 그래서 기본값은 꺼 두고, 조명이 어두울 때 `e` 키로 켜서 `m` 키의 마스크 창으로 두 결과를 비교한 뒤 더 깨끗한 쪽을 사용합니다.
- 엄지는 주먹을 쥐어도 관절이 거의 굽지 않아 각도 규칙만으로는 펴진 것으로 잘못 세어졌습니다. 그래서 엄지에는 각도 조건에 더해, 엄지 끝(4번)과 새끼 뿌리(17번) 사이 거리가 검지 뿌리(5번)와 17번 사이 거리의 `thumb_ratio`배(기본 1.3)보다 클 때만 펴진 것으로 보는 거리 조건을 추가했습니다. 손 크기 대비 비율로 비교하므로 손과 카메라 사이 거리가 달라도 같은 기준이 적용됩니다.
- 양손이 겹치면 손 인식이 불안정해지므로 양손을 떨어뜨려 입력합니다.
- 함수(`def`)는 실습에 있던 `calculate_angle` 외에, 같은 사칙연산 코드가 두 곳에서 필요해 `calculate`와 숫자 표시용 `format_number`, 엄지 판정용 `calculate_distance`와 `is_thumb_open`만 추가했습니다. 나머지는 실습 코드처럼 위에서 아래로 이어지는 구조입니다.
- AI 활용 vs. 본인 기여도
  - AI 활용: Claude를 사용해 예제 코드들을 정리하고 하나로 합쳤으며, 공의 위치를 이용해 사칙연산을 수행하는 아이디어에 도움을 받았습니다.
  - 본인 기여: 프로젝트 아이디어를 생성하고, 코드 가독성을 점검했습니다.