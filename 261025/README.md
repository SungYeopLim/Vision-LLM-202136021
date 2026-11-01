# Assignment 03

## 파일 구조

- **assignment03.py**: `03_DL-and-GPU.ipynb`와 `04_DL-Object-Detection.ipynb`의 과제 범위 코드를 실습 파일 순서대로 모은 파일
  1. 이미지 잔상 효과, Convolution 연산을 NumPy(CPU)와 CuPy(GPU)로 처리하고 시간과 결과 비교
  2. MNIST 데이터셋을 `main_cpu()`, `main_gpu()`로 학습해 CPU/GPU 학습 시간 비교
  3. CIFAR-10 데이터셋을 GPU로 학습, 평가하고 모델을 저장한 뒤 다시 불러와 확인
  4. IoU 계산과 NMS 구현
- **CIFAR10_CNN.pth**: `assignment03.py`로 5 Epoch 학습한 CIFAR-10 모델 파일
- **README.md**: 과제 설명 문서

```
Vision-LLM-202136021/
├── src/
│   ├── datasets/      (MNIST, CIFAR-10은 없으면 자동 다운로드)
│   └── images/
│       ├── seagull.jpg
│       └── two_apples1~3.png
└── 261025/
    ├── assignment03.py
    ├── CIFAR10_CNN.pth
    └── README.md
```

<br>

## 실행 방식

### 의존성

- `numpy`, `opencv-python`, `matplotlib`, `scikit-image`
- `cupy-cuda12x`, `torch`, `torchvision`
- CuPy와 PyTorch GPU 연산을 사용하므로 CUDA GPU가 필요합니다.

### 실행

데이터셋과 이미지 경로가 `../src/...` 상대 경로이므로 `261025` 폴더 위치에서 실행합니다.

```bash
cd 261025
python3 assignment03.py
```

- 그래프 창이 뜨면 창을 닫아야 다음 코드로 넘어갑니다.
- CIFAR-10 학습이 끝나면 `CIFAR10_CNN.pth`를 `261025` 폴더에 새로 저장하고, 저장한 파일을 다시 불러와 결과를 출력합니다.

<br>

## 기타 참고 사항

### 파일 구성

- CIFAR-10 학습에 쓰는 `train_one_epoch()`, `evaluate()`는 MNIST 부분에서 정의한 함수를 그대로 사용합니다.
- 실습 파일은 저장소 최상위 폴더 기준이라 경로가 `src/...`로 되어 있어서, 과제 파일에서는 `../src/...`로만 바꿨습니다.
- 모델 저장 경로는 실습 파일의 `src/models/CIFAR10/`이 `.gitignore`에 포함되어 저장소에 올라가지 않기 때문에, 과제 폴더(`261025`)에 저장하도록 바꿨습니다.

### 잔상 효과 (CPU)

원본 이미지를 오른쪽으로 `shift`칸 민 이미지에 `weights[shift]`를 곱해서 모두 더하는 방식입니다.

출력 픽셀 하나를 기준으로 보면, 오른쪽으로 `shift`칸 밀린 이미지에서 `output_x` 위치에 오는 값은 원본의 `output_x - shift` 위치 값입니다. 그래서 출력 픽셀마다 아래 계산을 합니다.

```
output[y, x] = image[y, x]      * weights[0]
             + image[y, x - 1]  * weights[1]
             + image[y, x - 2]  * weights[2]
             + ...
             + image[y, x - 31] * weights[31]
```

1. `height, width = image.shape`로 크기를 구하고, 입력과 같은 크기와 dtype(float32)의 0 배열을 `np.zeros_like(image)`로 만들어 출력 배열로 사용했습니다.
2. `output_y`, `output_x`를 `range(height)`, `range(width)`로 돌며 출력 픽셀을 하나씩 계산합니다.
3. 각 픽셀에서 `shift`를 0부터 `TRAIL_LENGTH - 1`까지 돌며 `input_x = output_x - shift`로 가져올 원본 x 좌표를 구합니다.
4. `input_x`가 0보다 작으면 이미지 왼쪽 밖이므로 더하지 않습니다. 실습 설명 예시에서 밀려난 앞부분이 0으로 채워지는 것(`[0, 10, 20, 30, 40]`)과 같은 결과입니다.
5. 범위 안이면 `image[output_y, input_x] * weights[shift]`를 `output[output_y, output_x]`에 `+=`로 누적해서, 누적과 저장을 한 줄로 처리했습니다.

1280x720 이미지에서 픽셀 921,600개마다 32번씩 Python 반복문으로 계산하기 때문에 시간이 오래 걸립니다.

### 잔상 효과 (GPU)

CPU 코드가 픽셀을 하나씩 계산하는 것과 달리, `shift` 반복만 Python에서 돌리고 한 번의 반복에서는 이미지 전체를 한 번에 계산합니다.

오른쪽으로 `shift`칸 미는 것은 배열 slicing으로 표현했습니다.

```python
input_region = image[:, :width - shift]   # 원본에서 오른쪽 끝 shift칸을 뺀 부분
output_region = output[:, shift:]         # 출력에서 왼쪽 앞 shift칸을 뺀 부분
```

두 영역은 크기가 같고, `input_region`의 0번 열이 `output_region`의 0번 열(= 출력의 `shift`번 열)에 대응됩니다. 실습 예시로 보면 `shift = 1`일 때 아래와 같습니다.

```
image          : [10, 20, 30, 40, 50]
input_region   : [10, 20, 30, 40]          (image[:, :5 - 1])
output_region  : 출력의 1, 2, 3, 4번 위치   (output[:, 1:])
```

그래서 `output_region += input_region * weights[shift]` 한 줄이 CPU 코드의 `output[y, x] += image[y, x - shift] * weights[shift]`를 모든 픽셀에 대해 한 번에 한 것과 같습니다. 출력의 앞 `shift`칸에는 더해지지 않으므로 CPU 코드에서 `input_x < 0`을 건너뛴 것과 같은 결과가 나옵니다.

- `output_region`은 새 배열이 아니라 `output`의 일부를 가리키는 view이기 때문에, `+=`로 더하면 따로 저장하지 않아도 `output`에 바로 반영됩니다.
- `width`는 slicing에만 필요해서 `image.shape[1]`만 가져왔고, 출력 배열은 `cp.zeros_like(image)`로 GPU 메모리에 만들었습니다.
- 곱셈과 덧셈이 CuPy 배열 연산이라 이미지 전체 픽셀 계산이 GPU에서 병렬로 처리되고, Python 반복은 32번만 돕니다.

### 잔상 효과 공통 사항

- 두 함수 모두 `shift < len(weights)` 조건을 넣었습니다. 반복문은 `TRAIL_LENGTH`(32)번 도는데 GPU 워밍업 셀에서는 가중치를 4개(`weights_np[:4]`)만 넘기기 때문에, 조건이 없으면 `weights[4]`에서 IndexError가 납니다.
- CuPy 배열 준비는 `cp.asarray()`로 이미지와 가중치를 GPU 메모리로 복사했고, 결과는 `cp.asnumpy()`로 CPU 메모리로 가져왔습니다. 복사 시간은 `upload_time`, `download_time`으로 따로 측정해서, 계산 시간만 비교한 속도와 전송 시간까지 포함한 속도를 둘 다 출력합니다.

### Convolution 연산 (GPU)

`convolution2d_cp()`는 잔상 효과 GPU 함수를 2차원으로 확장한 방식입니다.

- padding과 출력 크기 계산은 CPU의 `convolution2d()`와 같고, `np` 대신 `cp`를 사용했습니다.
- 출력 픽셀을 하나씩 도는 대신 커널 원소 `(kernel_y, kernel_x)` 개수만큼(3x3 필터는 9번)만 반복합니다. 각 반복에서 모든 출력 위치에 대응되는 입력 픽셀을 `padded_image[kernel_y::stride, kernel_x::stride]` 형태로 잘라 `kernel[kernel_y, kernel_x]`를 곱하고 출력 배열 전체에 누적합니다.
- slicing 끝 값을 세로는 `kernel_y + (output_height - 1) * stride + 1`, 가로는 `kernel_x + (output_width - 1) * stride + 1`로 지정해서, 잘라낸 영역이 출력 크기와 항상 같도록 했습니다.
- 마지막에 `bias`를 더합니다.

CPU 처리는 기존 `convolution2d()`에 실습에서 사용한 세로/가로 edge 필터와 편향(`dx_edge_filter`, `dy_edge_filter`, `dx_bias`, `dy_bias`, padding 1)을 그대로 사용했고, ReLU는 `relu()` 함수로 적용했습니다.

GPU 처리는 입력 이미지와 필터를 `cp.asarray()`로 변환한 뒤, 64x64 0 배열로 워밍업하고 시간을 측정했습니다. ReLU는 `cp.maximum(0, ...)`로 GPU에서 적용한 뒤 `cp.asnumpy()`로 가져왔습니다.

결과 출력에서는 CPU/GPU 처리 시간, 속도 차이, 두 결과의 최대 오차를 출력합니다.

### MNIST `main_gpu()`

`main_cpu()`와 같은 구조에서 장치를 `torch.device("cuda")`로 바꾸고, GPU 작업이 끝난 시점에 시간을 재기 위해 학습 시작 전, 각 Epoch 시작과 끝, 전체 학습 끝에 `torch.cuda.synchronize()`를 넣었습니다.

`if __name__ == "__main__":` 부분에서 `main_cpu()`와 `main_gpu()`를 차례로 실행하고 두 학습 시간의 차이와 속도 비율을 출력합니다.

### CIFAR-10 학습

MNIST의 `main_gpu()`와 같은 방식으로 5 Epoch 동안 `train_one_epoch()`, `evaluate()`를 실행하고 결과를 기록했습니다. Epoch 시간과 전체 학습 시간은 `torch.cuda.synchronize()` 후 `time.perf_counter()`로 측정했습니다.

### IoU

- `convert_bbox_coord()`: `cv2.boundingRect()`가 주는 `(x, y, w, h)`를 `(x1, y1, x2, y2)`로 바꿉니다.
- `calculate_iou()`: 교집합 좌표를 `max`, `min`으로 구하고 너비와 높이가 음수면 0으로 처리한 뒤, 교집합 넓이 / 합집합 넓이를 계산합니다. 합집합 넓이가 0 이하면 0을 반환해서 ZeroDivisionError를 막았습니다.

### NMS

- `prune_low_conf()`: Confidence Score가 `conf_threshold` 이상인 BBox만 남깁니다.
- `apply_nms()`:
  1. `np.argsort(scores)[::-1]`로 Confidence Score가 높은 순서의 인덱스를 구합니다.
  2. 가장 앞의 BBox를 결과에 추가합니다.
  3. 남은 BBox 중 방금 선택한 BBox와의 IoU가 `iou_threshold`보다 큰 것은 제거하고, 나머지만 다음 순서로 남깁니다.
  4. 남은 BBox가 없을 때까지 2~3번을 반복합니다.
- OpenCV `cv2.dnn.NMSBoxes()`도 IoU가 임계값보다 클 때 제거하기 때문에 같은 기준(`iou <= iou_threshold`이면 유지)으로 맞췄습니다.
