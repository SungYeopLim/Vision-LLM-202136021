import cv2
import json
import os
import time
from datetime import datetime

from ultralytics import YOLO


YOLO_MODEL_PATH = "../src/models/YOLO/yolo11n_int8.engine"
EVENT_JSON_PATH = "../src/json/review_event.json"
TEMP_EVENT_JSON_PATH = "../src/json/review_event_tmp.json"
REVIEW_DIR = "../src/output/reviews"

CAMERA_DEVICE = "/dev/video0"
CONFIDENCE = 0.4
ZONE = (160, 60, 480, 420)   # 촬영 영역 (x1, y1, x2, y2)
HOLD_SECONDS = 2.0           # 촬영 영역 안에 머물러야 하는 시간
MAX_LOST_FRAMES = 5          # 탐지가 끊겨도 유지 시간을 이어갈 프레임 수
RESET_LOST_FRAMES = 30       # 촬영 후 다음 촬영을 허용할 빈 영역 프레임 수

# 주문 목록: YOLO 클래스 → 상품명, 구매처, 카테고리
ORDER_ITEMS = {
    "bottle": {"name": "탄산음료 500ml", "platform": "coupang", "category": "food"},
    "cell phone": {"name": "스마트폰", "platform": "naver", "category": "electronics"},
    "mouse": {"name": "무선 마우스", "platform": "coupang", "category": "electronics"},
    "backpack": {"name": "데일리 백팩", "platform": "musinsa", "category": "fashion"},
    "book": {"name": "파이썬 입문서", "platform": "naver", "category": "living"},
    "orange": {"name": "제주 감귤 1kg", "platform": "kurly", "category": "food"},
}


pipeline = (
    f"v4l2src device={CAMERA_DEVICE} ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


# 선택한 객체의 ROI 이미지와 촬영 이벤트 JSON을 저장하는 함수
def save_capture(frame, selected):
    x1, y1, x2, y2 = selected["bbox"]

    # 이미지 크기에 맞게 좌표 보정
    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # ROI 이미지 추출
    object_image = frame[y1:y2, x1:x2].copy()

    if object_image.size == 0:
        print("객체 이미지를 추출할 수 없습니다.")
        return None

    event_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + selected["class"].replace(" ", "_")
    image_path = f"{REVIEW_DIR}/{event_id}.jpg"
    temp_image_path = f"{REVIEW_DIR}/{event_id}_temp.jpg"

    # 임시 파일 저장
    success = cv2.imwrite(temp_image_path, object_image)

    if not success:
        print("이미지 저장에 실패했습니다.")
        return None

    # 기존 이미지 교체
    os.replace(temp_image_path, image_path)

    event = {
        "event_id": event_id,
        "class": selected["class"],
        "confidence": selected["confidence"],
        "image_path": image_path,
        "product": ORDER_ITEMS[selected["class"]],
    }

    # Dictionary → JSON 파일 저장
    with open(TEMP_EVENT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(event, file, ensure_ascii=False, indent=4)

    # 임시 파일 작성 완료 후 실제 파일로 교체
    os.replace(TEMP_EVENT_JSON_PATH, EVENT_JSON_PATH)

    return event_id


os.makedirs(os.path.dirname(EVENT_JSON_PATH), exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

yolo = YOLO(YOLO_MODEL_PATH)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("촬영 대상:", ", ".join(ORDER_ITEMS.keys()))
print("q : 종료")
print("l : 촬영 영역 안 객체 바로 저장")


hold_class = None
hold_start = 0.0
lost_frames = 0
ready = True
last_saved = ""


while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break


    # 1. 현재 프레임 YOLO 탐지
    result = yolo.predict(
        source=frame,     # source image
        conf=CONFIDENCE,  # Confidence Threshold
        iou=0.5,          # IoU Threshold
        verbose=False,    # no output prints
    )[0]


    # 2. 주문 목록에 있는 객체 중 촬영 영역 안에 있는 객체만 추출
    zone_objects = []

    for box in result.boxes:
        class_name = result.names[int(box.cls[0].item())]

        if class_name not in ORDER_ITEMS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if ZONE[0] <= center_x <= ZONE[2] and ZONE[1] <= center_y <= ZONE[3]:
            zone_objects.append(
                {
                    "class": class_name,
                    "confidence": round(float(box.conf[0].item()), 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "area": (x2 - x1) * (y2 - y1),
                }
            )


    # 3. 촬영 영역 안 객체 중 가장 큰 객체 선택, 같은 객체가 머문 시간 계산
    selected = None

    if zone_objects:
        selected = max(zone_objects, key=lambda obj: obj["area"])
        lost_frames = 0

        # 다른 객체로 바뀌면 유지 시간 다시 측정
        if selected["class"] != hold_class:
            hold_class = selected["class"]
            hold_start = time.time()
    else:
        lost_frames += 1

        # 잠깐 탐지가 끊긴 경우는 유지 시간을 이어서 계산
        if lost_frames > MAX_LOST_FRAMES:
            hold_class = None

        # 촬영 영역이 충분히 비면 다음 촬영 허용
        if lost_frames > RESET_LOST_FRAMES:
            ready = True

    hold_time = time.time() - hold_start if hold_class is not None else 0.0


    # 4. 유지 시간이 지나거나 l 키 입력 시 ROI 이미지 저장
    capture = False

    if selected is not None:
        if key == ord("l"):
            capture = True
        elif ready and hold_time >= HOLD_SECONDS:
            capture = True

    if capture:
        event_id = save_capture(frame, selected)

        if event_id is not None:
            ready = False
            last_saved = event_id
            print(f"\n[이미지 저장 완료] {event_id}")


    # 5. YOLO 결과 화면 출력 (촬영 영역, 상태, 유지 시간 막대)
    output_frame = result.plot()

    zone_color = (0, 255, 0) if selected is not None else (255, 255, 255)
    cv2.rectangle(output_frame, (ZONE[0], ZONE[1]), (ZONE[2], ZONE[3]), zone_color, 2)

    if not ready:
        status_text = "Saved. Clear the zone for next item"
    elif hold_class is None:
        status_text = "Place an item in the zone"
    else:
        status_text = f"{hold_class}: hold {hold_time:.1f}/{HOLD_SECONDS:.1f}s"

        bar_width = int((ZONE[2] - ZONE[0]) * min(hold_time / HOLD_SECONDS, 1.0))
        cv2.rectangle(output_frame, (ZONE[0], ZONE[3] + 5), (ZONE[0] + bar_width, ZONE[3] + 15), (0, 255, 255), -1)

    cv2.putText(output_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if last_saved:
        cv2.putText(output_frame, f"Last: {last_saved}", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Photo Review - Vision", output_frame)


cap.release()
cv2.destroyAllWindows()
