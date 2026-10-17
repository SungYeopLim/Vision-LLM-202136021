import cv2
import numpy as np
import matplotlib.pyplot as plt

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

basketball = cv2.imread('../src/images/basketball_crop.jpg')

basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(basketball_lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)

# ----------------------------------------------------------------

# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ball_mask_opened_x2 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
final_ball_mask = cv2.morphologyEx(ball_mask_opened_x2, cv2.MORPH_CLOSE, kernel)

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(final_ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# TODO: 최종 농구공 영역 및 외곽선 선택
final_img_enhanced = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)

    largest_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 0] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

plt.figure(figsize=(16,8))
plt.subplot(1,3,1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Enhanced Mask")
plt.subplot(1,3,2), plt.imshow(final_ball_mask, cmap="gray"), plt.title("Opening x2 & Closing x1")
plt.subplot(1,3,3), plt.imshow(final_img_enhanced), plt.title(f"Largest Contour ({len(contours)} contours found)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()