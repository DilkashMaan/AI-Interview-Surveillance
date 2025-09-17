# import cv2

# for i in range(5):  # check first 5 indexes
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"Camera index {i} is available")
#         cap.release()

import cv2
import time

for i in range(3):  # you have 0, 1, 2
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        continue
    
    print(f"Showing camera index {i}...")
    start = time.time()
    while time.time() - start < 5:  # show for 5 seconds
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {i}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()