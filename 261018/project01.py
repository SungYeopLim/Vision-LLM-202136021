import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


flip_code = 1

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)
min_ball_area = 300
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

use_clahe = False
# CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

angle_threshold = 160  # 손가락 펼쳐짐 여부 판단 각도 임계값
finger_tips = (4, 8, 12, 16, 20)  # 손가락 끝 landmark의 index
# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = (
    (1, 2, 3),  # 엄지: 2번 중심
    (5, 6, 7),  # 검지: 6번 중심
    (9, 10, 11),  # 중지: 10번 중심
    (13, 14, 15),  # 약지: 14번 중심
    (17, 18, 19),  # 소지: 18번 중심
)
thumb_ratio = 1.3

hold_seconds = 1.0
max_lost_frames = 5
reset_lost_frames = 30
speed_threshold = 8.0

button_labels = ["+", "-", "*", "/", "sqrt", "x^2", "<-", "C", "="]
button_w = 64
button_h = 60
button_gap = 6
button_top = 10
button_area_bottom = button_top + button_h + 20

buttons = []
for i, label in enumerate(button_labels):
    x1 = 8 + i * (button_w + button_gap)
    buttons.append((label, x1, button_top, x1 + button_w, button_top + button_h))


# 중간 점 기준으로 각도 계산하는 함수
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def calculate_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]))


def is_thumb_open(hand):
    return calculate_distance(hand[4], hand[17]) > calculate_distance(hand[5], hand[17]) * thumb_ratio


def calculate(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            return None
        return a / b
    return None


def format_number(value):
    return f"{value:.8g}"


base_option = python.BaseOptions(model_asset_path="../src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)  # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)  # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS  # 각 landmark를 잇는 연결선 정보

# --------------- OpenCV Kalman Filter 설정 ---------------
# 상태 벡터: [x, y, vx, vy]
# 측정 벡터: [x, y]
kalman = cv2.KalmanFilter(4, 2)

# Transition Matrix (상태 전이 행렬)
kalman.transitionMatrix = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

# Measurement Matrix (측정 행렬)
# 측정값으로부터 x, y만 관측
kalman.measurementMatrix = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=np.float32,
)

# Process Noise Covariance Matrix (프로세스 노이즈 공분산)
# 값이 클수록 모델 예측보다 측정값 변화에 더 유연하게 반응
kalman.processNoiseCov = np.array(
    [
        [1e-2, 0, 0, 0],
        [0, 1e-2, 0, 0],
        [0, 0, 5e-2, 0],
        [0, 0, 0, 5e-2],
    ],
    dtype=np.float32,
)

# Measurement Noise Covariance Matrix (측정 노이즈 공분산)
# 값이 클수록 측정값을 덜 신뢰하고 예측값을 더 신뢰
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

# Posteriori Error Covariance Matrix (초기 추정 오차 공분산)
kalman.errorCovPost = np.eye(4, dtype=np.float32)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("FAIL!")
    exit()

# 객체 최초 검출 여부 확인용 boolean
found = False
lost_count = 0
x_bel, y_bel = 0, 0
radius = 0

hover_label = None
hover_start = 0
hover_locked = False

digit_label = None
digit_start = 0
digit_locked = False

current_text = "0"
stored = None
pending_op = None
last_op = None
last_operand = None
new_entry = True
has_operand = False
just_evaluated = False
error = False

show_mask = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("FAIL!")
        break

    frame = cv2.resize(frame, (640, 480))
    # 이미지 좌우 반전 (전처리)
    if flip_code is not None:
        frame = cv2.flip(frame, flip_code)
    h, w = frame.shape[:2]

    # Gaussian blur 적용
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    # LAB 색공간으로 변환
    lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

    if use_clahe:
        # 각 채널 분리
        l, a, b = cv2.split(lab)
        # L 채널에 CLAHE 적용 후 강화된 이미지 생성
        l_clahe_enhanced = clahe.apply(l)
        lab = cv2.merge((l_clahe_enhanced, a, b))

    # 초록색 추출하여 binary mask 생성
    mask = cv2.inRange(lab, green_lower, green_upper)

    # Morphological operation을 적용하여 최종 binary mask 생성
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    final_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # 컨투어 추출 (두 번째 반환값인 hierarchy는 바깥쪽 테두리만 검출하는 현재 방식상 무의미하기에 저장하지 않음)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_ball_area]

    if len(contours) > 0:
        # 가장 큰 contour 선택
        largest_contour = max(contours, key=cv2.contourArea)

        # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        # Moments를 통해 무게중심 (Centroid) 산출
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (int(x), int(y))

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        # 객체 최초 검출 (최초 a priori state 저장)
        if not found:
            kalman.statePre = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            kalman.statePost = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            found = True

        lost_count = 0
        # 측정값 사용 (visible) -> Prediction & Update
        z = np.array([[center[0]], [center[1]]], dtype=np.float32)
        predicted_state = kalman.predict()
        corrected_state = kalman.correct(z)
        x_bel = corrected_state[0, 0]
        y_bel = corrected_state[1, 0]

    # 측정값 미사용 (occluded) -> Prediction
    elif found:
        lost_count += 1
        predicted_state = kalman.predict()
        x_bel = predicted_state[0, 0]
        y_bel = predicted_state[1, 0]

        if lost_count > reset_lost_frames:
            found = False

    ball_speed = 0.0
    if found:
        ball_speed = np.hypot(kalman.statePost[2, 0], kalman.statePost[3, 0])

    cursor_valid = found and lost_count <= max_lost_frames

    # 예측한 객체 위치에 노란 원 overlay
    if cursor_valid:
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (0, 255, 255), -1)

    # RGB로 색공간 변환 (전처리)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 프레임 내 손 탐지
    hand_result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    # 감지된 모든 손에서 펼쳐진 손가락 개수 계산
    finger_count = None
    if len(hand_result.hand_landmarks) > 0:
        finger_count = 0
        for hand in hand_result.hand_landmarks:
            for i, (point1_idx, point2_idx, point3_idx) in enumerate(finger_angle_points):
                angle = calculate_angle(hand[point1_idx], hand[point2_idx], hand[point3_idx])
                if angle >= angle_threshold:
                    if i == 0 and not is_thumb_open(hand):
                        continue
                    finger_count += 1

    # 탐지 결과의 각 손마다 선과 점 그리기
    for hand in hand_result.hand_landmarks:
        points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표

        # landmark를 연결하는 선 (skeleton) 그리기
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)

        # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
        for i, point in enumerate(points):
            color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)

    pressed = None

    hovered = None
    if cursor_valid:
        for label, x1, y1, x2, y2 in buttons:
            if x1 < x_bel < x2 and y1 < y_bel < y2:
                hovered = label

    if hovered != hover_label:
        hover_label = hovered
        hover_start = time.time()
        hover_locked = False

    hover_progress = 0.0
    if hover_label is not None and not hover_locked:
        hover_progress = min((time.time() - hover_start) / hold_seconds, 1.0)
        if hover_progress >= 1.0:
            pressed = hover_label
            hover_locked = True

    digit_paused = (cursor_valid and y_bel < button_area_bottom) or ball_speed > speed_threshold

    digit_value = None
    if finger_count is not None and finger_count <= 9 and not digit_paused:
        digit_value = finger_count

    if digit_value != digit_label:
        digit_label = digit_value
        digit_start = time.time()
        digit_locked = False

    digit_progress = 0.0
    if digit_label is not None and not digit_locked and pressed is None:
        digit_progress = min((time.time() - digit_start) / hold_seconds, 1.0)
        if digit_progress >= 1.0:
            pressed = str(digit_label)
            digit_locked = True

    if pressed is not None:
        if error or pressed == "C":
            current_text = "0"
            stored = None
            pending_op = None
            last_op = None
            last_operand = None
            new_entry = True
            has_operand = False
            just_evaluated = False
            error = False

        if pressed.isdigit():
            if just_evaluated:
                stored = None
                last_op = None
                last_operand = None
                just_evaluated = False
            if new_entry or current_text == "0":
                current_text = pressed
                new_entry = False
            elif len(current_text) < 10:
                current_text += pressed
            has_operand = True

        elif pressed in ("+", "-", "*", "/"):
            value = float(current_text)
            if pending_op is not None and has_operand:
                stored = calculate(stored, pending_op, value)
            elif pending_op is None:
                stored = value
            if stored is None:
                error = True
            else:
                current_text = format_number(stored)
            pending_op = pressed
            new_entry = True
            has_operand = False
            just_evaluated = False

        elif pressed == "=":
            value = float(current_text)
            if pending_op is not None:
                last_op = pending_op
                last_operand = value
                stored = calculate(stored, pending_op, value)
                pending_op = None
            elif last_op is not None:
                stored = calculate(value, last_op, last_operand)
            else:
                stored = value
            if stored is None:
                error = True
            else:
                current_text = format_number(stored)
            new_entry = True
            has_operand = False
            just_evaluated = True

        elif pressed in ("sqrt", "x^2"):
            value = float(current_text)
            if pressed == "sqrt":
                if value < 0:
                    unary_result = None
                else:
                    unary_result = value ** 0.5
            else:
                unary_result = value * value
            if unary_result is None:
                error = True
            else:
                current_text = format_number(unary_result)
                if just_evaluated:
                    stored = unary_result
                new_entry = True
                has_operand = True

        elif pressed == "<-":
            if not new_entry:
                current_text = current_text[:-1]
                if current_text == "" or current_text == "-":
                    current_text = "0"

    for label, x1, y1, x2, y2 in buttons:
        if label == hover_label and not hover_locked:
            fill_x = x1 + int((x2 - x1) * hover_progress)
            cv2.rectangle(frame, (x1, y2 - 10), (fill_x, y2), (0, 200, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        font_scale = 0.9 if len(label) == 1 else 0.6
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x1 + (button_w - text_size[0]) // 2
        text_y = y1 + (button_h + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, 390), (w, h), (0, 0, 0), -1)

    expression = ""
    if pending_op is not None and stored is not None:
        expression = f"{format_number(stored)} {pending_op}"
    cv2.putText(frame, expression, (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    display_text = "FAIL!" if error else current_text
    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.putText(frame, display_text, (w - text_size[0] - 15, 465), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    if digit_paused:
        status_text = "Fingers: PAUSED"
    elif finger_count is None:
        status_text = "Fingers: -"
    else:
        status_text = f"Fingers: {finger_count}"
    cv2.putText(frame, status_text, (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    clahe_text = "CLAHE: ON" if use_clahe else "CLAHE: OFF"
    cv2.putText(frame, clahe_text, (w - 160, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if digit_progress > 0:
        cv2.rectangle(frame, (10, 465), (10 + int(200 * digit_progress), 475), (0, 255, 255), -1)

    cv2.imshow("Gesture Calculator", frame)
    if show_mask:
        cv2.imshow("Binary Mask", final_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow("Binary Mask")
    if key == ord("e"):
        use_clahe = not use_clahe

hand_detector.close()
cap.release()
cv2.destroyAllWindows()
