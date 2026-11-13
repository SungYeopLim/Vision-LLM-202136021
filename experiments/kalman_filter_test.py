import cv2
import numpy as np
from numpy.linalg import inv

# Kalman filter 변수 정의
# A_t = np.array([[1, 1], [0, 1]])
# G = np.array([[0.5], [1]])
# R_t = G.dot(G.transpose())
# C_t = np.array([[1, 0]])
# Q_t = np.array([[1]])
# mu_t = np.array([[0, 0], [0, 0]])
# sigma_t = np.array([[0, 0], [0, 0]])

# def KalmanFilter(mu_prev, sigma_prev, z):
#     mu_bar = A_t.dot(mu_prev)
#     sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
#     if z is None:
#         return mu_bar, sigma_bar
#     else:
#         K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
#         mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
#         sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
#         return mu, sigma
    
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

kalman = cv2.KalmanFilter(4, 2)

kalman.transitionMatrix = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

kalman.measurementMatrix = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=np.float32,
)

kalman.processNoiseCov = np.array(
    [
        [1e-2, 0, 0, 0],
        [0, 1e-2, 0, 0],
        [0, 0, 5e-2, 0],
        [0, 0, 0, 5e-2],
    ],
    dtype=np.float32,
)

kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

kalman.errorCovPost = np.eye(4, dtype=np.float32)

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

found = False
x_bel, y_bel = 0, 0
radius = 0

while True:
    ret, frame = cap.read()
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0) # Gaussin blur 적용
    lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, green_lower, green_upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        largest_mask = np.zeros_like(mask)

        mask_color = np.zeros_like(frame)
        mask_color[:, :, 0] = largest_mask
        alpha = 0.15
        frame = cv2.addWeighted(frame, 1.0, mask_color, alpha, 0)
 
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(frame, center, radius, (255,0,0), 3)
        cv2.circle(frame, center, 5, (255,0,0), -1)

        if not found:
            kalman.statePre = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            kalman.statePost = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            found = True
    
    if found and (len(contours) > 0):
            z = np.array([[center[0]], [center[1]]],dtype=np.float32)
            predicted_state = kalman.predict()   
            corrected_state = kalman.correct(z)  
            x_bel = corrected_state[0, 0]
            y_bel = corrected_state[1, 0]

    elif found and (len(contours) <= 0):
            predicted_state = kalman.predict()   
            x_bel = predicted_state[0, 0]
            y_bel = predicted_state[1, 0]

    cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius),(0, 255, 255), 3)
    cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (0, 255, 255), -1)

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("VideoCapture with GStreamer", frame)

cap.release()
cv2.destroyAllWindows()