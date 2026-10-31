import time
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
import cupy as cp
import skimage.data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def relu(x):
    return np.maximum(0, x)

def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지, shape = (height, width)
    kernel : np.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : np.ndarray
        편향까지 적용된 convolution 결과
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 커널을 적용할 이미지 영역의 좌측상단
            start_y = output_y * stride
            start_x = output_x * stride

            # 커널을 적용할 이미지 영역을 crop
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 필터를 적용하여 가중합 계산
            weighted_sum = np.sum(image_region * kernel)

            # 편향 적용
            output[output_y, output_x] = weighted_sum + bias

    return output

# x축 방향 edge (세로 선) 검출 필터
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# y축 방향 edge (가로 선) 검출 필터
dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

# 너무 약한 특징 (0.2 이하)은 무시
dx_bias = -0.2
dy_bias = -0.2

# ----------------------------------------------------------------

IMAGE_PATH = "../src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

plt.imshow(input_np, cmap="gray")
plt.axis("off");

# TODO: NumPy를 사용하여 잔상 효과 함수 정의

def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = np.zeros_like(image)

    for output_y in range(height):
        for output_x in range(width):

            for shift in range(TRAIL_LENGTH):
                # TODO: x값 shift
                input_x = output_x - shift

                # TODO: shift한 픽셀을 누적
                # TODO: 출력 배열에 누적 픽셀 저장
                if input_x >= 0 and shift < len(weights):
                    output[output_y, output_x] += image[output_y, input_x] * weights[shift]

    return output

# TODO: CuPy를 사용하여 잔상 효과 함수 정의

def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    width = image.shape[1]
    output = cp.zeros_like(image)

    for shift in range(TRAIL_LENGTH):
        # TODO: 배열 slicing
        input_region = image[:, :width - shift]
        output_region = output[:, shift:]

        # TODO: GPU가 병렬 연산할 output_region 계산 코드
        if shift < len(weights):
            output_region += input_region * weights[shift]

    return output

# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

# 전체 가중치의 합을 1로 정규화
weights_np /= np.sum(weights_np)

cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

# TODO: NumPy 배열을 CuPy 배열로 변환

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start

warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

cp.cuda.Stream.null.synchronize()

cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start

# TODO: output_cp를 CPU 메모리로 복사

download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")

plt.figure(figsize=(18,12))

plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.figure(figsize=(12,8))
plt.imshow(output_cp_np, cmap="gray")
plt.axis("off");

# ----------------------------------------------------------------

plt.imshow(input_np, cmap="gray")
plt.axis("off");

# TODO: CuPy를 사용하여 2차원 convolution 함수 정의

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            input_region = padded_image[
                kernel_y:kernel_y + (output_height - 1) * stride + 1:stride,
                kernel_x:kernel_x + (output_width - 1) * stride + 1:stride
            ]

            output += input_region * kernel[kernel_y, kernel_x]

    output += bias

    return output

# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능

cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np,
    kernel=dx_edge_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)
output_dy_np = convolution2d(
    image=input_np,
    kernel=dy_edge_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)
cpu_time = time.perf_counter() - cpu_start
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)

# TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
# TODO: 변수 재사용 가능

input_cp = cp.asarray(input_np)
dx_edge_filter_cp = cp.asarray(dx_edge_filter)
dy_edge_filter_cp = cp.asarray(dy_edge_filter)

warmup_image = cp.zeros((64, 64), dtype=cp.float32)

_ = convolution2d_cp(
    image=warmup_image,
    kernel=dx_edge_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=dx_edge_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)
output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=dy_edge_filter_cp,
    bias=dy_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()

gpu_time = time.perf_counter() - gpu_start
relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))

# TODO: CPU/GPU 결과 비교 및 출력
compute_speedup = cpu_time / gpu_time

max_difference_dx = np.max(np.abs(relu_dx_np - relu_dx_cp_np))
max_difference_dy = np.max(np.abs(relu_dy_np - relu_dy_cp_np))

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"GPU processing: {gpu_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Max difference (vertical): {max_difference_dx:.6f}")
print(f"Max difference (horizontal): {max_difference_dy:.6f}")

plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# ----------------------------------------------------------------

SEED = 42
DATA_ROOT = "../src/datasets"

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            10,
        )

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }

            return logits, features

        return logits


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 모델을 학습 모드(training mode)로 변경
    model.train()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 학습 데이터를 DataLoader(train_loader)의 Batch 단위로 불러와 학습 반복
    for images, labels in data_loader:
        # 장치(CPU/GPU)로 이동
        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화 (이전 Batch에서 계산한 Gradient를 제거)
        optimizer.zero_grad()

        # 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
        logits = model(images)

        # Loss 계산
        loss = criterion(logits, labels)
        # 역전파를 통한 Gradient 계산
        loss.backward()
        # Gradient를 사용하여 파라미터 수정
        optimizer.step()

        # Batch 한 개의 이미지 개수
        batch_size = images.size(0)

        # 결과 누적
        total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
        predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
        correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
        sample_count += batch_size                             # 전체 샘플 수 누적

    # 평균 Loss 및 Accuracy 계산
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    # 모델을 평가 모드(evaluation mode)로 변경
    model.eval()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # Gradient 계산을 비활성화 (파라미터 수정 비활성화)
    with torch.no_grad():
        # 평가 데이터를 DataLoader(test_loader)의 Batch 단위로 불러와 평가 반복
        for images, labels in data_loader:
            # 장치(CPU/GPU)로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
            logits = model(images)

            # Loss 계산 (역전파는 수행하지 않음)
            loss = criterion(logits, labels)

            # Batch 한 개의 이미지 개수
            batch_size = images.size(0)

            # 결과 누적
            total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
            predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
            correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
            sample_count += batch_size                             # 전체 샘플 수 누적

        # 평균 Loss 및 Accuracy 계산
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy


def main_cpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="../src/datasets",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cpu")

    model = SimpleCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total CPU training time: {total_training_time:.2f}s")

    return total_training_time

# TODO: GPU 기반 MNIST 데이터셋 학습 코드 작성

def main_gpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="../src/datasets",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda")

    model = SimpleCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    torch.cuda.synchronize()

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return total_training_time

if __name__ == "__main__":
    print("=== CPU Training ===")
    cpu_training_time = main_cpu()

    print("\n=== GPU Training ===")
    gpu_training_time = main_gpu()

    speedup = cpu_training_time / gpu_training_time
    time_difference = cpu_training_time - gpu_training_time

    print("\n=== CPU vs. GPU Comparison ===")
    print(f"CPU training time: {cpu_training_time:.2f}s")
    print(f"GPU training time: {gpu_training_time:.2f}s")
    print(f"Time difference: {time_difference:.2f}s")
    print(f"GPU speedup: {speedup:.2f}x")

# ----------------------------------------------------------------

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed);

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

train_dataset = datasets.CIFAR10(
    root="../src/datasets",
    train=True,
    transform=train_transform,
    download=True,
)

test_dataset = datasets.CIFAR10(
    root="../src/datasets",
    train=False,
    transform=test_transform,
    download=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

class ConvBlock(nn.Module):
    """
    Convolution을 이용해 특징을 추출하고 Max Pooling을 적용하는 재사용 가능한 CNN 블록.

    기본 구조:
    Conv2d → BatchNorm2d → ReLU
    → Conv2d → BatchNorm2d → ReLU
    → MaxPool2d
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

         # Block을 구성할 Layer들을 순서대로 저장
        layers = [

            # 첫 번째 Convolution (특징 추출)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            # 각 출력 채널의 값 분포를 조정하여 학습 안정화
            nn.BatchNorm2d(out_channels),

            # ReLU 활성화 함수
            nn.ReLU(),

            # 두 번째 Convolution (특징 추가적으로 추출)
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            # 두 번째 Convolution의 출력을 정규화
            nn.BatchNorm2d(out_channels),

            # 두 번째 활성화 함수를 적용
            nn.ReLU(),

            # Max Pooling 적용하여 다운스케일
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
            
        ]

        # 리스트에 저장한 Layer들을 순서대로 실행하는 하나의 Sequential Module로 구성
        self.block = nn.Sequential(*layers)

    # 입력 x를 self.block의 Layer에 순서대로 전달
    def forward(self, x):
        return self.block(x)

class CIFAR10CNN(nn.Module):
    """
    CIFAR-10의 32x32 RGB 이미지를 10개 클래스로 분류하는 CNN 모델.

    전체 구조:
    ConvBlock 1
    → ConvBlock 2
    → ConvBlock 3
    → Global Average Pooling
    → Dropout
    → Linear
    """

    def __init__(self):
        super().__init__()

        # 이미지에서 특징을 추출하는 부분 (ConvBlock 1 → ConvBlock 2 → ConvBlock 3)
        self.features = nn.Sequential(

            
            ConvBlock(                 # 출력: [Batch, 64, 16, 16]
                in_channels=3,
                out_channels=64,
            ),

            
            ConvBlock(                 # 출력: [Batch, 128, 8, 8]
                in_channels=64,
                out_channels=128,
            ),

            
            ConvBlock(                 # 출력: [Batch, 256, 4, 4]
                in_channels=128,
                out_channels=256,
            ),

        )

        # 각 4×4 Feature Map의 평균을 계산하여 공간 크기를 1×1로 압축
        self.global_average_pool = (
            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)     # 출력: [Batch, 256, 1, 1]
                )
        )

        # 추출된 특징을 이용해 최종 클래스를 예측
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),         # 출력: 256

            nn.Linear(                 # 출력: 10
                in_features=256,
                out_features=10,
            ),
        )

    # 입력 이미지: [Batch, 3, 32, 32]
    def forward(self, x):
        # 세 개의 ConvBlock을 통과하며 특징을 추출
        # 출력: [Batch, 256, 4, 4]
        x = self.features(x)

        # 각 채널의 4×4 값을 평균내어 1×1로 압축
        # 출력: [Batch, 256, 1, 1]
        x = self.global_average_pool(x)

        # Tensor를 2차원으로 펼침
        # [Batch, 256, 1, 1] → [Batch, 256]
        x = torch.flatten(x, start_dim=1)

        # 각 클래스에 대한 점수인 Logit을 계산
        # 출력: [Batch, 10]
        logits = self.classifier(x)

        return logits

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU를 사용할 수 없습니다."
    )

device = torch.device("cuda")
model = CIFAR10CNN().to(device)

print("Device:", device)
print(model)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", parameter_count)
print("Trainable parameters:", trainable_parameter_count)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
)

warmup_input = torch.zeros(64, 3, 32, 32, device=device)

model.eval()

with torch.no_grad():
    for _ in range(3):
        _ = model(warmup_input)

torch.cuda.synchronize()

# TODO: CUDA GPU 장치에서 CIFAR-10 데이터셋을 학습과 평가
# TODO: 각 Epoch 및 전체 학습 시간을 출력

EPOCHS = 5

# 학습 기록 저장 공간 준비
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# 전체 학습 시간 측정 시작
torch.cuda.synchronize()
training_start_time = time.perf_counter()

# TODO: 5번의 Epoch 동안 학습 및 평가 진행
for epoch in range(EPOCHS):
    torch.cuda.synchronize()
    epoch_start_time = time.perf_counter()

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_start_time

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# TODO: 전체 학습 시간 계산 및 출력
torch.cuda.synchronize()
total_training_time = time.perf_counter() - training_start_time
print(f"Total GPU training time: {total_training_time:.2f}s")

# 학습된 모델로 이미지 하나 테스트
image, true_label = test_dataset[0]          # 테스트할 이미지(image)와 정답(label)
input_batch = image.unsqueeze(0).to(device)  # Batch 차원 추가(unsqueeze)하여 입력 Batch로 저장 (이미지 1장)

# 모델을 평가 모드(evaluation mode)로 변경
model.eval()

# Gradient 계산을 비활성화 (파라미터 수정 비활성화)
with torch.no_grad():
    logits = model(input_batch)                           # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
    probabilities = torch.softmax(logits, dim=1)          # 원시 점수에 softmax 함수 적용 (클래스별 확률 계산)
    predicted_label = probabilities.argmax(dim=1).item()  # 정답 예측

# 결과 출력 (정답, 예측, confidence)
print("True label:", true_label)
print("Predicted label:", predicted_label)
print("Confidence:", probabilities[0, predicted_label].item())

# 결과 이미지 출력
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

image_denormalized = image.squeeze(0) * std + mean
image_rgb = image_denormalized.permute(1, 2, 0).clamp(0, 1)
image_rgb_cpu = image_rgb.cpu()

plt.figure(figsize=(4,4))
plt.imshow(image_rgb_cpu)
plt.title(f"True: {CIFAR10_CLASSES[true_label]}, " f"Prediction: {CIFAR10_CLASSES[predicted_label]}")
plt.axis("off")
plt.show()

plt.figure(figsize=(15,10))

num_images = 40

model.eval()

with torch.no_grad():
    for i in range(num_images):
        image, true_label = test_dataset[i]
        input_batch = image.unsqueeze(0).to(device)

        logits = model(input_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()

        image_denormalized = image.squeeze(0) * std + mean
        image_rgb = image_denormalized.permute(1, 2, 0).clamp(0, 1)
        image_rgb_cpu = image_rgb.cpu()

        title_color = "red" if predicted_label != true_label else "black"

        plt.subplot(4, 10, i+1)
        plt.imshow(image_rgb_cpu)
        plt.title(
            f"True: {CIFAR10_CLASSES[true_label]}\n"
            f"Pred: {CIFAR10_CLASSES[predicted_label]}\n"
            f"Conf: {confidence*100:.1f}%",
            color=title_color
        )
        plt.axis("off")

plt.tight_layout()
plt.show()

epoch_axis = range(1, EPOCHS + 1)

plt.plot(
    epoch_axis,
    train_loss_history,
    marker="o",
    label="Train loss",
)

plt.plot(
    epoch_axis,
    test_loss_history,
    marker="o",
    label="Test loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.grid()
plt.legend()
plt.show()

plt.plot(
    epoch_axis,
    np.array(train_accuracy_history) * 100,
    marker="o",
    label="Train accuracy",
)

plt.plot(
    epoch_axis,
    np.array(test_accuracy_history) * 100,
    marker="o",
    label="Test accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.grid()
plt.legend()
plt.show()

save_data = {
    "model_name": "CIFAR10CNN",
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "test_accuracy": test_accuracy_history[-1],
    "epochs": EPOCHS,
    "class_names": CIFAR10_CLASSES,
    "mean": CIFAR10_MEAN,
    "std": CIFAR10_STD,
}

torch.save(
    save_data,
    "CIFAR10_CNN.pth",
)

loaded_model = CIFAR10CNN()

state_dict = torch.load(
    "CIFAR10_CNN.pth",
    map_location="cpu",
    weights_only=True,
)

loaded_model.load_state_dict(state_dict["model_state_dict"])
loaded_model.to(device)
loaded_model.eval()

    # "model_name": "CIFAR10CNN",
    # "model_state_dict": model.state_dict(),
    # "optimizer_state_dict": optimizer.state_dict(),
    # "test_accuracy": test_accuracy_history[-1],
    # "epochs": EPOCHS,
    # "class_names": CIFAR10_CLASSES,
    # "mean": CIFAR10_MEAN,
    # "std": CIFAR10_STD,
print("모델 이름:", state_dict["model_name"])
print("Epoch 체크포인트:", state_dict["epochs"])
print(f"마지막 테스트 정확도: {state_dict['test_accuracy'] * 100:.2f}%")

plt.figure(figsize=(15,10))

num_images = 40

loaded_model.eval()

with torch.no_grad():
    for i in range(num_images):
        image, true_label = test_dataset[i+40]
        input_batch = image.unsqueeze(0).to(device)

        logits = loaded_model(input_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()

        image_denormalized = image.squeeze(0) * std + mean
        image_rgb = image_denormalized.permute(1, 2, 0).clamp(0, 1)
        image_rgb_cpu = image_rgb.cpu()

        title_color = "red" if predicted_label != true_label else "black"

        plt.subplot(4, 10, i+1)
        plt.imshow(image_rgb_cpu)
        plt.title(
            f"True: {CIFAR10_CLASSES[true_label]}\n"
            f"Pred: {CIFAR10_CLASSES[predicted_label]}\n"
            f"Conf: {confidence*100:.1f}%",
            color=title_color
        )
        plt.axis("off")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------

def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x+w, y+h

def calculate_iou(bbox_a, bbox_b):

    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)

    intersection_area = w * h

    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0

    # TODO: ZeroDivisionError 주의
    return intersection_area / union_area

apples1 = cv2.imread("../src/images/two_apples1.png")
apples2 = cv2.imread("../src/images/two_apples2.png")
apples3 = cv2.imread("../src/images/two_apples3.png")
apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,6))

plt.subplot(1,3,1), plt.imshow(apples1_rgb)
plt.subplot(1,3,2), plt.imshow(apples2_rgb)
plt.subplot(1,3,3), plt.imshow(apples3_rgb)

for ax in plt.gcf().axes:
    ax.axis("off")

def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn

apples1 = cv2.imread("../src/images/two_apples1.png")
apples2 = cv2.imread("../src/images/two_apples2.png")
apples3 = cv2.imread("../src/images/two_apples3.png")

apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)

cntr_red1, cntr_grn1 = get_contour(apples1)
cntr_red2, cntr_grn2 = get_contour(apples2)
cntr_red3, cntr_grn3 = get_contour(apples3)

bbox1 = cv2.boundingRect(cntr_red1)
bbox2 = cv2.boundingRect(cntr_grn1)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples1_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples1_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples1_rgb);

iou1 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou1)

bbox1 = cv2.boundingRect(cntr_red2)
bbox2 = cv2.boundingRect(cntr_grn2)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples2_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples2_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples2_rgb);

iou2 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou2)

bbox1 = cv2.boundingRect(cntr_red3)
bbox2 = cv2.boundingRect(cntr_grn3)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples3_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples3_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples3_rgb);

iou3 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou3)

plt.figure(figsize=(16,6))

plt.subplot(1,3,1), plt.imshow(apples1_rgb), plt.title(f"IoU: {iou1:.4f}")
plt.subplot(1,3,2), plt.imshow(apples2_rgb), plt.title(f"IoU: {iou2:.4f}")
plt.subplot(1,3,3), plt.imshow(apples3_rgb), plt.title(f"IoU: {iou3:.4f}")

for ax in plt.gcf().axes:
    ax.axis("off")

# ----------------------------------------------------------------

from random import randint, gauss

def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))

hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """

    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)
    
    return pruned_boxes, pruned_scores

# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """

    if len(boxes) == 0:
        return [], []

    order = np.argsort(scores)[::-1]

    nms_boxes = []
    nms_scores = []

    while len(order) > 0:
        best_index = order[0]

        nms_boxes.append(boxes[best_index])
        nms_scores.append(scores[best_index])

        remaining_order = []

        for index in order[1:]:
            iou = calculate_iou(boxes[best_index], boxes[index])

            if iou <= iou_threshold:
                remaining_order.append(index)

        order = remaining_order

    return nms_boxes, nms_scores

print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(boxes, scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """
    OpenCV의 cv2.dnn.NMSBoxes()를 이용해 NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록
        각 박스는 기존과 동일한 (x1, y1, x2, y2) 형식

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score
    """

    if len(boxes) == 0:
        return [], []

    # cv2.dnn.NMSBoxes()는 (x, y, width, height) 형식을 사용
    boxes_xywh = []

    for box in boxes:
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        boxes_xywh.append([int(x1), int(y1), int(width), int(height)])

    # OpenCV NMS 적용하여 BBox 제거 후 남은 BBox의 인덱스
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    # 1차원 리스트로 변환
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    # 원래 xyxy 형식의 박스에서 선택
    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

print(len(boxes), len(scores))

cv2_new_boxes, cv2_new_scores = apply_nms_cv2(boxes, scores, iou_threshold=0.2)

print(len(cv2_new_boxes), len(cv2_new_boxes))

hubble_cv2_nms = skimage.data.hubble_deep_field()

for i, box in enumerate(cv2_new_boxes):
    draw_bbox(
        hubble_cv2_nms,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble_cv2_nms)
plt.axis("off");

plt.figure(figsize=(16,10))

plt.subplot(1,2,1), plt.imshow(hubble), plt.title("Custom NMS Result")
plt.subplot(1,2,2), plt.imshow(hubble_cv2_nms), plt.title("CV2 NMS Result")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()
