import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image, ImageTk

# Loading YOLO models
vehicle_model = YOLO("yolov8n.pt")  # Detect vehicles
plate_model = YOLO("license_plate.pt")  # Detect license plates
helmet_model = YOLO("helmet.pt")  # Detect helmets
traffic_model = YOLO("meta_yolo.pt")  # Detect traffic signals


# Initializeing the DeepSORT tracker
tracker = DeepSort(max_age=30)

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Global video path
video_path = None
cap = None

# GUI Function to Select Video
def select_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if video_path:
        lbl_selected_video.config(text=f"Selected: {video_path}", fg="green")

# Function to Start Processing the Selected Video
def start_processing(source):
    global cap
    cap = cv2.VideoCapture(source)  # Open video file or webcam
    
    if not cap.isOpened():
        lbl_selected_video.config(text=" Error: Can't open video source!", fg="red")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:  # Check if frame capture was successful
            print("Error: Empty frame received! Exiting loop.")
            continue  # Skip this frame instead of crashing

        frame = process_frame(frame)
        if frame is not None and frame.size > 0:  # Ensure frame has data
            cv2.imshow("AI Traffic Management System", frame)
        else:
            print("Warning: Processed frame is empty!")

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    colors = {}

    def get_color(track_id):
        track_id = int(track_id)
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, size=3).tolist())

    def recognize_plate(plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        plate_text = pytesseract.image_to_string(thresh, config="--psm 7")
        return plate_text.strip()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        vehicle_results = vehicle_model(frame)
        detections = []
        for result in vehicle_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, class_id, conf in zip(boxes, class_ids, confidences):
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    label = vehicle_model.names[int(class_id)]
                    
                    if label in ["car", "truck", "bus", "motorbike"]:
                        detections.append(([x1, y1, x2, y2], conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = str(track.track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            if track_id not in colors:
                colors[track_id] = get_color(track_id)
            color = colors[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue

            plate_results = plate_model(frame)  # Detect the number plates in full frame
            for plate_result in plate_results:
                plate_boxes = plate_result.boxes.xyxy.cpu().numpy()
                for plate_box in plate_boxes:
                    px1, py1, px2, py2 = map(int, plate_box)

                    # Crop the license plate
                    plate_img = frame[py1:py2, px1:px2]
                    plate_number = recognize_plate(plate_img)

                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Plate: {plate_number}', (px1, py1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Helmet Detection
        helmet_results = helmet_model(frame)
        for helmet_result in helmet_results:
            helmet_boxes = helmet_result.boxes.xyxy.cpu().numpy()
            helmet_class_ids = helmet_result.boxes.cls.cpu().numpy()
            helmet_confidences = helmet_result.boxes.conf.cpu().numpy()

            for box, class_id, conf in zip(helmet_boxes, helmet_class_ids, helmet_confidences):
                if conf > 0.5:
                    hx1, hy1, hx2, hy2 = map(int, box)
                    helmet_label = helmet_model.names[int(class_id)]
                    helmet_color = (0, 0, 255) if helmet_label == "no_helmet" else (0, 255, 0)
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), helmet_color, 2)
                    cv2.putText(frame, helmet_label, (hx1, hy1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, helmet_color, 2)

        # Traffic Violation Detection
        traffic_results = traffic_model(frame)
        for traffic_result in traffic_results:
            traffic_boxes = traffic_result.boxes.xyxy.cpu().numpy()
            traffic_class_ids = traffic_result.boxes.cls.cpu().numpy()
            traffic_confidences = traffic_result.boxes.conf.cpu().numpy()

            for box, class_id, conf in zip(traffic_boxes, traffic_class_ids, traffic_confidences):
                if conf > 0.5:
                    tx1, ty1, tx2, ty2 = map(int, box)
                    traffic_label = traffic_model.names[int(class_id)]
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
                    cv2.putText(frame, traffic_label, (tx1, ty1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("AI Traffic Management System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
root = tk.Tk()
root.title("AI Traffic Management System")
root.geometry("1920x1080")
root.configure(bg="black")

# to add empty space on top
tk.Label(root, text="", bg="black", height=8).pack()

icon_image = Image.open("trafficlight.png").resize((300, 300))
icon_photo = ImageTk.PhotoImage(icon_image)
icon_label = tk.Label(root, image=icon_photo, bg="black")
icon_label.pack(pady=(50,10))

lbl_title = tk.Label(root, text="AI Traffic Management System", font=("Arial", 34, "bold"), fg="white", bg="black")
lbl_title.pack(pady=10)

btn_select = tk.Button(root, text="📁 Select Video", command=select_video, font=("Arial", 30), bg="#ffcc00", fg="black", width=20)
btn_select.pack(pady=5)

lbl_selected_video = tk.Label(root, text="No video selected :/", font=("Arial", 26), fg="gray", bg="black")
lbl_selected_video.pack(pady=5)

btn_start = tk.Button(root, text="▶ Start Processing", command=lambda: start_processing(video_path if video_path else ""), font=("Arial", 30), bg="lime", fg="black", width=20)
btn_start.pack(pady=5)
#if video_path else "" is used to ensure if video_path is none or empty, it doesn't crash

btn_webcam = tk.Button(root, text="📷 Use Camera", command=lambda: start_processing(0), font=("Arial", 30), bg="blue", fg="white", width=20)
btn_webcam.pack(pady=5)
#If the system is not having a webcam, it will display an error instead of crashing.

btn_quit = tk.Button(root, text="❌ Quit", command=root.quit, font=("Arial", 30), bg="red", fg="white", width=20)
btn_quit.pack(pady=5)

lbl_note = tk.Text(root, font=("Arial", 20), fg="pink", bg="black", height=1, width=55, borderwidth=0)
lbl_note.insert(tk.END, "Note: ", "yellow")
lbl_note.insert(tk.END, "Press ")
lbl_note.insert(tk.END, "Esc key", "red")  # Add 'Esc' with a red tag
lbl_note.insert(tk.END, " on the keyboard to terminate the video processing midway")

lbl_note.tag_configure("red", foreground="red")
lbl_note.tag_configure("yellow", foreground="yellow")
lbl_note.config(state=tk.DISABLED)
lbl_note.pack(pady=10)

root.mainloop()
