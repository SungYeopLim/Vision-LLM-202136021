# Assignment 03

## 파일 구조

- **assignment03.py**: `03_DL-and-GPU.ipynb`의 MNIST CNN 학습 코드에 이어서 Epoch별 Loss/Accuracy 그래프와 학습된 모델의 Feature Map을 출력하고, NumPy(CPU)와 CuPy(GPU)로 이미지 잔상 효과를 만들어 처리 시간을 비교하는 코드
- **README.md**: 과제 설명 문서

```
Vision-LLM-202136021/
├── src/
│   ├── datasets/
│   │   └── MNIST/    (최초 실행 시 자동 다운로드)
│   └── images/
│       └── seagull.jpg
└── 261025/
    ├── assignment03.py
    └── README.md
```

<br>

## 실행 방식

CuPy와 CUDA를 사용하므로 Jetson에서 실행합니다. (JetPack 6.2.1, CUDA 12.6, Python 3.10.12 기준)

1. 저장소 폴더로 이동한 뒤 가상 환경을 만들고 활성화합니다. `--system-site-packages`는 JetPack에 설치된 OpenCV와 NumPy를 가상 환경에서도 그대로 쓰기 위한 옵션입니다.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

2. 실습 파일에 나온 순서대로 PyTorch와 Torchvision을 설치합니다.

```bash
pip install torch==2.8.0 torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip uninstall torchvision
pip install torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip install "sympy>=1.13.3" --no-deps
```

3. CuPy를 설치합니다.

```bash
python -m pip download --no-deps --only-binary=:all: "cupy-cuda12x==12.3.0" -d ~/Downloads
python -m pip install --no-deps "fastrlock==0.8.3"
python -m pip install --no-deps --only-binary=:all: ~/Downloads/cupy_cuda12x-12.3.0-*.whl
```

4. 가상 환경을 활성화한 상태로 `261025` 폴더 위치에서 아래 명령을 입력합니다.

```bash
cd 261025
python3 assignment03.py
```

5. 그래프 창이 뜨면 창을 닫아야 다음 단계로 넘어갑니다.

<br>

## 기타 참고 사항

### 파일 구성

- 실습 파일은 저장소 최상위 폴더에서 실행하기 때문에 경로가 `src/...`로 되어 있는데, 과제 파일은 `261025` 폴더에서 실행하므로 `../src/...`로만 바꿨습니다.

### 16) 모델 성능 확인

- x축은 `range(1, epochs + 1)`로 Epoch 번호를 1부터 표시했습니다.
- Accuracy는 0~1 값으로 저장되어 있어서 `np.array(...) * 100`으로 백분율로 바꿔 그렸습니다.

### 17) 학습된 모델의 Feature Map 확인

- 테스트 데이터의 첫 번째 이미지(`test_dataset[0]`)를 선택하고, `torch.no_grad()` 안에서 `return_features=True`로 순전파해 각 층의 Feature Map을 받았습니다.
- Conv1은 8개, Conv2는 16개 전체를 `show_feature_maps()`에 `cmap="viridis"`로 넣어 출력했습니다.
- 6개 Layer 비교는 각 Layer의 0~3번 채널을 한 줄씩 배치해서 6행 4열로 출력했습니다.

### 잔상 효과 함수 (CPU, GPU)

- **CPU**: 출력 픽셀 `(output_y, output_x)`마다 `input_x = output_x - shift`로 왼쪽 픽셀을 가져와 가중치를 곱해 누적했습니다. `input_x`가 0보다 작으면 이미지 밖이므로 더하지 않았고, 실습 설명의 예시처럼 앞쪽이 0으로 채워진 것과 같은 결과가 나옵니다.
- **GPU**: 이동 거리마다 `image[:, :width - shift]`를 `output[:, shift:]` 위치에 맞춰 잘라 `output_region += input_region * weights[shift]`로 한 번에 더했습니다. `output_region`은 `output`의 일부를 가리키기 때문에 따로 저장하지 않아도 `output`에 바로 반영됩니다.
- 두 함수 모두 `shift < len(weights)` 조건을 넣었습니다. 반복문은 `TRAIL_LENGTH`(32)번 도는데 GPU 워밍업 셀에서는 가중치를 4개만 넘기기 때문에, 조건이 없으면 `weights[4]`에서 IndexError가 납니다.

### 실행 시간

- CPU로 MNIST를 3 Epoch 학습하는 부분과 CPU 잔상 효과 함수(1280x720 픽셀마다 32번 반복)는 시간이 오래 걸립니다. 실습 시간에 Jetson에서 CPU 학습은 Epoch당 약 200초 정도 걸렸습니다.
