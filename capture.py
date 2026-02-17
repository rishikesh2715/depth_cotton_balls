import cv2, os, time

OUT_DIR = "captures"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
assert cap.isOpened()

distance_in = 6  # start here; change with keys

print("Keys: [s]=save frame, [=]=+6in, [-]=-6in, [q]=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    disp = frame.copy()
    cv2.putText(disp, f"Distance: {distance_in} in", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Capture", disp)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('='):   # plus
        distance_in = min(72, distance_in + 6)
    elif k == ord('-'):   # minus
        distance_in = max(6, distance_in - 6)
    elif k == ord('s'):
        ts = int(time.time() * 1000)
        fn = os.path.join(OUT_DIR, f"d{distance_in:02d}_{ts}.jpg")
        cv2.imwrite(fn, frame)
        print("saved", fn)

cap.release()
cv2.destroyAllWindows()