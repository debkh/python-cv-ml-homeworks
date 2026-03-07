import cv2
from pathlib import Path

VIDEO_PATH = Path("../../data/lesson_10/traffic.mp4")
CASCADE_PATH = Path("../../data/lesson_10/cars.xml")
RESULTS_DIR = Path("../../data/lesson_10/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REDETECT_FRAMES_PERIOD = 10
FRAME_LIMIT = 15
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

def detect_largest_car(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
  
    if len(cars) == 0:
        return None

    print(f"Detected {len(cars)} cars in the current frame.")
    largest_car = max(cars, key=lambda rect: rect[2] * rect[3])
    return tuple(largest_car)

# Main function to run the tracking (tracker_type can be "KCF" or "CSRT")
def main(tracker_type: str = "KCF"):
    if tracker_type not in ["KCF", "CSRT"]:
        raise ValueError("Invalid tracker_type. Must be 'KCF' or 'CSRT'.")

    video_capture = cv2.VideoCapture(str(VIDEO_PATH))
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

    if not video_capture.isOpened():
        raise RuntimeError("Error opening video stream or file")
    if cascade.empty():
        raise RuntimeError("Error loading cars.xml cascade")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
    delay_ms = max(1, int(frame_duration * 1000))

    ret, first_frame = video_capture.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    first_bbox = detect_largest_car(first_frame, cascade)
    if first_bbox is None:
        raise RuntimeError("No cars detected in the first frame")

    if tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    tracker.init(first_frame, first_bbox)

    frame_index = 0
    detected_count = 0
    redetected_count = 0

    while frame_index < FRAME_LIMIT:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        success, bbox = tracker.update(frame)

        if not success or (frame_index % REDETECT_FRAMES_PERIOD == 0):
            redetect_bbox = detect_largest_car(frame, cascade)
            if redetect_bbox is not None:
                if tracker_type == "KCF":
                    tracker = cv2.TrackerKCF_create()
                elif tracker_type == "CSRT":
                    tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, redetect_bbox)
                bbox = redetect_bbox
                success = True
                redetected_count += 1

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)
            cv2.putText(frame, f"Frame: {frame_index}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_GREEN, 2)
            detected_count += 1
        else:
            cv2.putText(frame, f"Frame: {frame_index} - Tracking failure", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_RED, 2)

        cv2.putText(frame, f"Detections: {detected_count}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_GREEN, 2)
        cv2.putText(frame, f"Redetections: {redetected_count}", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_GREEN, 2)
        tracker_type_path = tracker_type.lower()
        tracker_dir = RESULTS_DIR / tracker_type_path
        tracker_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tracker_dir / f"output_frame_{frame_index}.jpg"), frame)  # Save the frame for debugging
        cv2.imshow("Car Tracking", frame)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

        frame_index += 1

    video_capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main(tracker_type="CSRT")  # Change to "CSRT" to use the CSRT tracker