import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

image = skimage.data.astronaut()

# ----------------------------------------------------------------

# TODO: astronaut 이미지를 흑백으로 변환하여 렌더링

image = image.astype(np.float32) / 255.

image_gray_wrong = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1), plt.imshow(image_gray, cmap="gray"), plt.title("RGB2GRAY (correct)")
plt.subplot(1,2,2), plt.imshow(image_gray_wrong, cmap="gray"), plt.title("BGR2GRAY (wrong)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# ----------------------------------------------------------------

# TODO: astronaut 이미지를 HSV 색공간으로 변환
# TODO: HSV 각 채널을 분리한 뒤 subplot에 시각화

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(image_hsv)

plt.figure(figsize=(15,8))

plt.subplot(1, 3, 1), plt.imshow(h, cmap="gray"), plt.title('H Channel (Hue)')
plt.subplot(1, 3, 2), plt.imshow(s, cmap="gray"), plt.title('S Channel (Saturation)')
plt.subplot(1, 3, 3), plt.imshow(v, cmap="gray"), plt.title('V Channel (Value)')

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# ----------------------------------------------------------------

image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# TODO: HSV 이미지의 각 채널 분리

h, s, v = cv2.split(image_hsv)

# TODO: H 각도를 증가시켜 색상 변경 (360도 초과하지 않도록 유의)

shift = 45
h = (h + shift) % 360

# TODO: 각 채널을 cv2.merge([h, s, v])로 결합

image_merge = cv2.merge([h, s, v])

# TODO: 색공간을 다시 변환하여 이미지 출력

image_color = cv2.cvtColor(image_merge, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(12,8))

plt.subplot(1, 2, 1), plt.imshow(image), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(image_color), plt.title('H Increased')

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# ----------------------------------------------------------------

image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# TODO: HSV 이미지의 각 채널 분리

h, s, v = cv2.split(image_hsv)

# TODO: S 값에 배율을 적용하여 채도 감쇠

scale = 0.3
s = s * scale

# TODO: np.clip(s, 0, 1)을 사용하여 최소/최대값 설정

s = np.clip(s, 0, 1)

# TODO: 각 채널을 cv2.merge([h, s, v])로 결합

image_merge = cv2.merge([h, s, v])

# TODO: 색공간을 다시 변환하여 이미지 출력

image_color = cv2.cvtColor(image_merge, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(12,8))

plt.subplot(1, 2, 1), plt.imshow(image), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(image_color), plt.title('S Decreased')

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# ----------------------------------------------------------------

image = skimage.data.astronaut()
image = image.astype(np.float32) / 255.

# TODO: 이미지를 HSV 색공간으로 변환

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# TODO: HSV 이미지의 각 채널 분리

h, s, v = cv2.split(image_hsv)

# TODO: V 값에 배율을 적용하여 명도 증폭

scale = 1.8
v = v * scale

# TODO: np.clip(v, 0, 1)을 사용하여 최소/최대값 설정

v = np.clip(v, 0, 1)

# TODO: 각 채널을 cv2.merge([h, s, v])로 결합

image_merge = cv2.merge([h, s, v])

# TODO: 색공간을 다시 변환하여 이미지 출력

image_color = cv2.cvtColor(image_merge, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(12,8))

plt.subplot(1, 2, 1), plt.imshow(image), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(image_color), plt.title('V Increased')

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# ----------------------------------------------------------------

# TODO: 기존 sharpening kernel 포함하여 총 3가지 kernel을 적용한 결과를 원본 이미지와 함께 비교하는 subplot 생성

sharp_kernel1 = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)

sharp_kernel2 = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]], dtype=np.float32)

sharp_kernel3 = np.array([[-1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1],
                          [-1, -1, 25, -1, -1],
                          [-1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1]], dtype=np.float32)

sharpened1 = cv2.filter2D(image, cv2.CV_32F, sharp_kernel1)
sharpened1 = np.clip(sharpened1, 0, 1)
sharpened2 = cv2.filter2D(image, cv2.CV_32F, sharp_kernel2)
sharpened2 = np.clip(sharpened2, 0, 1)
sharpened3 = cv2.filter2D(image, cv2.CV_32F, sharp_kernel3)
sharpened3 = np.clip(sharpened3, 0, 1)

plt.figure(figsize=(16,8))
plt.subplot(1,4,1), plt.imshow(image), plt.title("Original")
plt.subplot(1,4,2), plt.imshow(sharpened1), plt.title("Kernel 1 (3x3, center 5)")
plt.subplot(1,4,3), plt.imshow(sharpened2), plt.title("Kernel 2 (3x3, center 9)")
plt.subplot(1,4,4), plt.imshow(sharpened3), plt.title("Kernel 3 (5x5, center 25)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()

# ----------------------------------------------------------------

# TODO: 각 borderType을 적용한 결과 이미지와 제목이 적혀있는 subplot 생성

kernel_size = (101, 101)

border_types = [
    (cv2.BORDER_CONSTANT, "CONSTANT"),
    (cv2.BORDER_REFLECT, "REFLECT"),
    (cv2.BORDER_REFLECT_101, "REFLECT_101"),
    (cv2.BORDER_REPLICATE, "REPLICATE"),
]

plt.figure(figsize=(18,8))

for idx, (border, name) in enumerate(border_types):
    result = cv2.blur(image, kernel_size, borderType=border)

    plt.subplot(1, 4, idx+1)
    plt.imshow(result)
    plt.title(name)
    plt.axis("off")

# TODO: enumerate과 for loop을 사용하여 효율적으로 subplot 생성

plt.show()

# ----------------------------------------------------------------

img_path = 'src/images/sky.png'
sky_img = cv2.imread(img_path)

plt.figure(figsize=(5,10))

# TODO: 원본 이미지 확인

sky_img = cv2.cvtColor(sky_img, cv2.COLOR_BGR2RGB)
plt.imshow(sky_img)
plt.axis("off")
plt.show()

# ----------------------------------------------------------------

# TODO: 이미지를 grayscale로 변환
sky_img_gray = cv2.cvtColor(sky_img, cv2.COLOR_RGB2GRAY)

# TODO: 이미지를 float32로 변환
sky_img_gray = sky_img_gray.astype(np.float32) / 255.

# TODO: Gaussian Filter 적용 (Kernel 사이즈는 55x55)
sky_blur = cv2.GaussianBlur(sky_img_gray, (55,55), 0)

# TODO: cv2.subtract를 사용하여 원본에서 저주파 성분 차감 (High-pass Filter)
sky_hf = cv2.subtract(sky_img_gray, sky_blur)

# TODO: 고주파 이미지 확인 (cmap 설정)
plt.figure(figsize=(5,10))
plt.imshow(sky_hf, cmap="gray")
plt.axis("off")

plt.show()

# ----------------------------------------------------------------

# TODO: cv2.add를 사용하여 원본과 고주파 이미지를 합산
unsharp_mask_applied = cv2.add(sky_img_gray, sky_hf)

# TODO: subplot 3개 (원본 흑백 이미지, 고주파 이미지, unsharp mask 적용된 이미지) 생성
plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(sky_img_gray, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(sky_hf, cmap="gray"), plt.title("High Frequency")
plt.subplot(1,3,3), plt.imshow(unsharp_mask_applied, cmap="gray"), plt.title("Unsharp Mask Applied")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()