import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import threading
import queue
import os


class AIObjectDetectionApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Object Detection Application")
        self.root.geometry("1100x750")
        self.root.configure(bg="#101827")
        self.root.minsize(850, 600)

        # Application variables
        self.model = None
        self.cap = None
        self.worker = None
        self.stop_event = threading.Event()

        self.frame_queue = queue.Queue(maxsize=1)
        self.current_frame = None
        self.detecting = False

        self.output_folder = "output"
        os.makedirs(self.output_folder, exist_ok=True)

        self.setup_ui()

        # Load YOLO model
        self.root.after(200, self.load_model)

        # Process frames on the Tkinter main thread
        self.root.after(30, self.update_display)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    # ---------------------------------
    # CREATE USER INTERFACE
    # ---------------------------------

    def setup_ui(self):

        title = tk.Label(
            self.root,
            text="AI OBJECT DETECTION",
            font=("Arial", 25, "bold"),
            bg="#101827",
            fg="#00e5ff"
        )
        title.pack(pady=(18, 5))

        subtitle = tk.Label(
            self.root,
            text="Real-Time Object Detection Using YOLO AI",
            font=("Arial", 11),
            bg="#101827",
            fg="white"
        )
        subtitle.pack(pady=(0, 15))

        # Video display area
        self.video_frame = tk.Frame(
            self.root,
            bg="#000000",
            highlightbackground="#00e5ff",
            highlightthickness=2
        )
        self.video_frame.pack(
            padx=25,
            pady=10,
            fill="both",
            expand=True
        )

        self.video_label = tk.Label(
            self.video_frame,
            text="AI OBJECT DETECTION\n\n"
                 "Select Webcam, Image or Video to Start",
            font=("Arial", 17, "bold"),
            bg="black",
            fg="#00e5ff"
        )
        self.video_label.pack(
            fill="both",
            expand=True
        )

        # Buttons
        button_frame = tk.Frame(
            self.root,
            bg="#101827"
        )
        button_frame.pack(pady=15)

        self.create_button(
            button_frame,
            "Open Webcam",
            self.start_webcam,
            "#00c853"
        ).grid(row=0, column=0, padx=6, pady=5)

        self.create_button(
            button_frame,
            "Upload Image",
            self.open_image,
            "#2196f3"
        ).grid(row=0, column=1, padx=6, pady=5)

        self.create_button(
            button_frame,
            "Upload Video",
            self.open_video,
            "#9c27b0"
        ).grid(row=0, column=2, padx=6, pady=5)

        self.create_button(
            button_frame,
            "Stop",
            self.stop_detection,
            "#f44336"
        ).grid(row=0, column=3, padx=6, pady=5)

        self.create_button(
            button_frame,
            "Save Snapshot",
            self.save_snapshot,
            "#ff9800"
        ).grid(row=0, column=4, padx=6, pady=5)

        # Status area
        self.status_label = tk.Label(
            self.root,
            text="Loading AI model...",
            font=("Arial", 12, "bold"),
            bg="#101827",
            fg="#00e5ff"
        )
        self.status_label.pack(pady=5)

        # Detection results
        self.result_label = tk.Label(
            self.root,
            text="Detected Objects: 0",
            font=("Arial", 13, "bold"),
            bg="#1e293b",
            fg="white",
            padx=20,
            pady=12
        )
        self.result_label.pack(
            fill="x",
            padx=25,
            pady=(5, 15)
        )

    def create_button(self, parent, text, command, color):

        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=12,
            relief="flat",
            cursor="hand2"
        )

    # ---------------------------------
    # LOAD YOLO AI MODEL
    # ---------------------------------

    def load_model(self):

        try:

            self.status_label.config(
                text="Loading YOLO AI Model..."
            )

            self.model = YOLO("yolov8n.pt")

            self.status_label.config(
                text="AI Model Loaded Successfully! Ready."
            )

        except Exception as error:

            self.status_label.config(
                text="Model loading failed!"
            )

            messagebox.showerror(
                "Model Error",
                str(error)
            )

    # ---------------------------------
    # CHECK MODEL
    # ---------------------------------

    def check_model(self):

        if self.model is None:

            messagebox.showwarning(
                "Please Wait",
                "The AI model is still loading."
            )

            return False

        return True

    # ---------------------------------
    # DISPLAY FRAME
    # ---------------------------------

    def show_frame(self, frame):

        self.current_frame = frame.copy()

        height, width = frame.shape[:2]

        max_width = 950
        max_height = 500

        scale = min(
            max_width / width,
            max_height / height,
            1.0
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        frame = cv2.resize(
            frame,
            (new_width, new_height)
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        photo = ImageTk.PhotoImage(image=image)

        self.video_label.config(
            image=photo,
            text=""
        )

        self.video_label.image = photo

    # ---------------------------------
    # SHOW DETECTION RESULTS
    # ---------------------------------

    def get_detection_info(self, result):

        objects = []

        for box in result.boxes:

            class_id = int(box.cls[0])

            name = result.names[class_id]

            confidence = float(box.conf[0]) * 100

            objects.append(
                f"{name}: {confidence:.1f}%"
            )

        return objects

    # ---------------------------------
    # DETECT IMAGE
    # ---------------------------------

    def open_image(self):

        if not self.check_model():
            return

        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_detection()

        try:

            image = cv2.imread(path)

            if image is None:
                raise ValueError("Cannot read selected image.")

            self.status_label.config(
                text="Detecting objects in image..."
            )

            results = self.model.predict(
                image,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            self.show_frame(annotated)

            objects = self.get_detection_info(result)

            self.result_label.config(
                text="Detected Objects: "
                     + str(len(objects))
                     + "\n"
                     + (
                         " | ".join(objects)
                         if objects
                         else "No objects detected"
                     )
            )

            self.status_label.config(
                text="Image detection completed!"
            )

        except Exception as error:

            messagebox.showerror(
                "Image Detection Error",
                str(error)
            )

    # ---------------------------------
    # START WEBCAM
    # ---------------------------------

    def start_webcam(self):

        if not self.check_model():
            return

        self.start_stream(0)

    # ---------------------------------
    # OPEN VIDEO FILE
    # ---------------------------------

    def open_video(self):

        if not self.check_model():
            return

        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )

        if path:
            self.start_stream(path)

    # ---------------------------------
    # START VIDEO OR CAMERA STREAM
    # ---------------------------------

    def start_stream(self, source):

        self.stop_detection()

        self.stop_event = threading.Event()

        self.detecting = True

        self.status_label.config(
            text="Starting object detection..."
        )

        self.worker = threading.Thread(
            target=self.detection_worker,
            args=(source, self.stop_event),
            daemon=True
        )

        self.worker.start()

    # ---------------------------------
    # BACKGROUND DETECTION THREAD
    # ---------------------------------

    def detection_worker(self, source, stop_event):

        cap = None

        try:

            cap = cv2.VideoCapture(source)

            if not cap.isOpened():

                self.put_frame(
                    ("error", "Cannot open camera or video.")
                )

                return

            while not stop_event.is_set():

                success, frame = cap.read()

                if not success:
                    break

                results = self.model.predict(
                    frame,
                    conf=0.25,
                    verbose=False
                )

                result = results[0]

                annotated = result.plot()

                objects = self.get_detection_info(result)

                self.put_frame(
                    ("frame", annotated, objects)
                )

        except Exception as error:

            self.put_frame(
                ("error", str(error))
            )

        finally:

            if cap is not None:
                cap.release()

            self.put_frame(("finished",))

    # ---------------------------------
    # THREAD-SAFE FRAME QUEUE
    # ---------------------------------

    def put_frame(self, item):

        try:

            self.frame_queue.put_nowait(item)

        except queue.Full:

            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.frame_queue.put_nowait(item)
            except queue.Full:
                pass

    # ---------------------------------
    # UPDATE GUI
    # ---------------------------------

    def update_display(self):

        try:

            while True:

                item = self.frame_queue.get_nowait()

                event = item[0]

                if event == "frame":

                    frame = item[1]
                    objects = item[2]

                    self.show_frame(frame)

                    self.result_label.config(
                        text="Detected Objects: "
                             + str(len(objects))
                             + "\n"
                             + (
                                 " | ".join(objects)
                                 if objects
                                 else "No objects detected"
                             )
                    )

                    self.status_label.config(
                        text="AI Detection Running..."
                    )

                elif event == "error":

                    self.status_label.config(
                        text="Detection error!"
                    )

                    messagebox.showerror(
                        "Detection Error",
                        item[1]
                    )

                    self.detecting = False

                elif event == "finished":

                    self.detecting = False

                    if not self.stop_event.is_set():

                        self.status_label.config(
                            text="Video completed or stream ended."
                        )

        except queue.Empty:
            pass

        if self.root.winfo_exists():
            self.root.after(30, self.update_display)

    # ---------------------------------
    # STOP DETECTION
    # ---------------------------------

    def stop_detection(self):

        self.detecting = False

        self.stop_event.set()

        self.status_label.config(
            text="Detection stopped."
        )

    # ---------------------------------
    # SAVE DETECTED IMAGE
    # ---------------------------------

    def save_snapshot(self):

        if self.current_frame is None:

            messagebox.showwarning(
                "No Image",
                "There is no detection image to save."
            )

            return

        from datetime import datetime

        filename = datetime.now().strftime(
            "detection_%Y%m%d_%H%M%S.jpg"
        )

        path = filedialog.asksaveasfilename(
            initialdir=self.output_folder,
            initialfile=filename,
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("PNG Image", "*.png")
            ]
        )

        if path:

            success = cv2.imwrite(
                path,
                self.current_frame
            )

            if success:

                messagebox.showinfo(
                    "Saved",
                    "Detection snapshot saved successfully!"
                )

            else:

                messagebox.showerror(
                    "Error",
                    "Could not save image."
                )

    # ---------------------------------
    # CLOSE APPLICATION
    # ---------------------------------

    def close_app(self):

        self.stop_detection()

        self.root.destroy()


# ---------------------------------
# RUN APPLICATION
# ---------------------------------

if __name__ == "__main__":

    root = tk.Tk()

    app = AIObjectDetectionApp(root)

    root.mainloop()
