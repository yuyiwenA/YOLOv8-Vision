import cv2
import pandas as pd
from ultralytics import YOLO

def main(video_path, output_video="result.mp4", output_csv="trajectory.csv"):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    trajectory_records = []
    frame_id = 0
    print(f"Processing video, total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                trajectory_records.append({
                    "frame_id": frame_id,
                    "class": cls_name,
                    "confidence": conf,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
        out.write(frame)
        frame_id += 1
        if frame_id % 30 == 0:
            print(f"Processed {frame_id}/{total_frames} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(trajectory_records)
    df.to_csv(output_csv, index=False)
    print(f"Done! Result video saved to {output_video}")
    print(f"Trajectory data saved to {output_csv}")

if __name__ == "__main__":
    VIDEO_PATH = "test_video.mp4"
    main(VIDEO_PATH)
