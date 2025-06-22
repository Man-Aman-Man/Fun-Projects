import cv2
import numpy as np
import mediapipe as mp

# Init MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
canvas = None

draw_color = (0, 255, 0)
mode = "Draw"  # Default mode
prev_x, prev_y = 0, 0

def fingers_up(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    landmarks = hand_landmarks.landmark

    # Thumb
    fingers.append(1 if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x else 0)
    # Other fingers
    for id in range(1, 5):
        fingers.append(1 if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y else 0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        lmList = handLms.landmark
        fingers = fingers_up(handLms)

        x = int(lmList[8].x * w)
        y = int(lmList[8].y * h)

        # Detect gestures
        if fingers == [0, 1, 0, 0, 0]:
            mode = "Draw"
        elif fingers == [0, 1, 1, 0, 0]:
            mode = "Erase"
        elif fingers == [1, 1, 1, 1, 1]:
            mode = "Clear"

        if mode == "Draw":
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5)
            prev_x, prev_y = x, y
        elif mode == "Erase":
            cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
            prev_x, prev_y = 0, 0
        elif mode == "Clear":
            canvas = np.zeros_like(frame)
            prev_x, prev_y = 0, 0
        else:
            prev_x, prev_y = 0, 0

        # Show fingertip circle
        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

    else:
        prev_x, prev_y = 0, 0

    # Overlay canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(frame_bg, canvas_fg)

    # Show mode on screen
    cv2.putText(frame, f"Mode: {mode}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Air Drawing", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
