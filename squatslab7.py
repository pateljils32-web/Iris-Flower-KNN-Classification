"""AI-Based Virtual Assistant – Real-Time Squat Coach

Controls:
  q = quit
  r = reset repetitions and timer

Install dependencies in PyCharm Terminal:
  python -m pip install mediapipe pyttsx3 opencv-python
"""

import math
import queue
import threading
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import pyttsx3
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

CAMERA_INDEX = 0
STANDING_KNEE_ANGLE = 160
SQUAT_KNEE_ANGLE = 100
FEEDBACK_DELAY_SECONDS = 3

MODEL_FILE = Path(__file__).with_name("pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def download_pose_model_if_needed():
    if MODEL_FILE.exists():
        return True
    print("Downloading MediaPipe pose model...")
    try:
        urlretrieve(MODEL_URL, str(MODEL_FILE))
        print("Pose model downloaded successfully.")
        return True
    except Exception as error:
        print(f"Could not download pose model: {error}")
        return False


def calculate_angle(point_a, point_b, point_c):
    ab_x = point_a.x - point_b.x
    ab_y = point_a.y - point_b.y
    cb_x = point_c.x - point_b.x
    cb_y = point_c.y - point_b.y
    dot_product = ab_x * cb_x + ab_y * cb_y
    magnitude_ab = math.hypot(ab_x, ab_y)
    magnitude_cb = math.hypot(cb_x, cb_y)
    if magnitude_ab == 0 or magnitude_cb == 0:
        return 0.0
    cosine = dot_product / (magnitude_ab * magnitude_cb)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


class VoiceCoach:
    def __init__(self):
        self.messages = queue.Queue()
        self.last_message = ""
        self.last_spoken_time = 0.0
        threading.Thread(target=self._speak_messages, daemon=True).start()

    def _speak_messages(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        while True:
            message = self.messages.get()
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception as error:
                print(f"Voice feedback error: {error}")

    def say(self, message):
        now = time.time()
        if (message != self.last_message or
                now - self.last_spoken_time >= FEEDBACK_DELAY_SECONDS):
            self.messages.put(message)
            self.last_message = message
            self.last_spoken_time = now


def landmark_visibility(landmark):
    return landmark.visibility or 0.0


def choose_side(landmarks):
    left_hip = landmarks[LEFT_HIP]
    right_hip = landmarks[RIGHT_HIP]
    if landmark_visibility(left_hip) >= landmark_visibility(right_hip):
        return (
            landmarks[LEFT_SHOULDER], left_hip,
            landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE], "left"
        )
    return (
        landmarks[RIGHT_SHOULDER], right_hip,
        landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE], "right"
    )


def assess_squat(shoulder, hip, knee, ankle):
    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)
    if knee_angle < 60:
        message = "Keep your knees in line with your toes."
    elif hip_angle < 55:
        message = "Keep your chest more upright."
    elif knee_angle <= SQUAT_KNEE_ANGLE:
        message = "Good depth. Push through your heels."
    else:
        message = "Lower slowly into your squat."
    return knee_angle, hip_angle, message


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.7):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def draw_squat_landmarks(frame, shoulder, hip, knee, ankle):
    height, width = frame.shape[:2]
    landmarks = [shoulder, hip, knee, ankle]
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for first, second in zip(points, points[1:]):
        cv2.line(frame, first, second, (0, 255, 0), 3)
    for point in points:
        cv2.circle(frame, point, 6, (0, 0, 255), -1)


def create_pose_detector():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    if not download_pose_model_if_needed():
        return

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Could not open camera. Try changing CAMERA_INDEX to 1.")
        return

    coach = VoiceCoach()
    repetitions = 0
    squat_stage = "up"
    workout_start_time = time.time()

    try:
        with create_pose_detector() as pose_detector:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("Could not read video frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_pipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )
                timestamp_ms = int(time.monotonic() * 1000)
                result = pose_detector.detect_for_video(media_pipe_image, timestamp_ms)

                feedback = "Step sideways so your full body is visible."
                feedback_color = (0, 180, 255)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    shoulder, hip, knee, ankle, side = choose_side(landmarks)
                    knee_angle, hip_angle, feedback = assess_squat(
                        shoulder, hip, knee, ankle
                    )

                    if knee_angle <= SQUAT_KNEE_ANGLE:
                        squat_stage = "down"
                    elif knee_angle >= STANDING_KNEE_ANGLE and squat_stage == "down":
                        squat_stage = "up"
                        repetitions += 1
                        feedback = f"Great squat. Rep {repetitions} completed."
                        feedback_color = (0, 255, 0)
                        coach.say(feedback)
                    else:
                        if "Good" in feedback:
                            feedback_color = (0, 255, 0)
                        coach.say(feedback)

                    draw_squat_landmarks(frame, shoulder, hip, knee, ankle)
                    draw_text(frame, f"{side.title()} Knee: {int(knee_angle)} degrees", (15, 105))
                    draw_text(frame, f"Hip: {int(hip_angle)} degrees", (15, 135))

                elapsed_seconds = int(time.time() - workout_start_time)
                minutes, seconds = divmod(elapsed_seconds, 60)
                draw_text(frame, f"Reps: {repetitions}", (15, 35), (0, 255, 0), 1)
                draw_text(frame, f"Time: {minutes:02}:{seconds:02}", (15, 70), (0, 255, 0), 1)
                draw_text(frame, f"Stage: {squat_stage}", (15, 165))
                draw_text(frame, feedback, (15, frame.shape[0] - 25), feedback_color, 0.6)
                draw_text(frame, "q = quit | r = reset", (15, frame.shape[0] - 55), (200, 200, 200), 0.55)
                cv2.imshow("AI Squat Virtual Assistant", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    repetitions = 0
                    squat_stage = "up"
                    workout_start_time = time.time()
                    coach.say("Counter and timer reset.")
    except Exception as error:
        print(f"Program error: {error}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
