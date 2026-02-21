import argparse
import math
import os
import time
import urllib.request
from collections import deque

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import pose_landmarker


MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
	"pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

HAV_EVENT_THRESHOLD_DEG_S = 15.0
HAV_SLOW_THRESHOLD_DEG_S = 25.0
HAV_EVENT_MIN_DURATION_S = 0.5
HAV_EVENT_END_THRESHOLD_DEG_S = 9.0
WB_TORSO_ANGULAR_THRESHOLD_DEG_S = 15.0
WB_TORSO_TRANSLATION_THRESHOLD = 0.03
WINDOW_SECONDS = 60.0
EMA_ALPHA = 0.25


def ensure_model(model_path: str) -> None:
	if os.path.exists(model_path):
		return
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	urllib.request.urlretrieve(MODEL_URL, model_path)


def draw_pose(frame, landmarks, connections) -> None:
	h, w = frame.shape[:2]
	for lm in landmarks:
		x = int(lm.x * w)
		y = int(lm.y * h)
		if 0 <= x < w and 0 <= y < h:
			cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

	for conn in connections:
		start = landmarks[conn.start]
		end = landmarks[conn.end]
		x1, y1 = int(start.x * w), int(start.y * h)
		x2, y2 = int(end.x * w), int(end.y * h)
		if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
			cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def _avg(a, b):
	return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _sub(a, b):
	return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _wrap_angle(delta):
	while delta > math.pi:
		delta -= 2.0 * math.pi
	while delta < -math.pi:
		delta += 2.0 * math.pi
	return delta


def _angles_from_landmarks(landmarks):
	nose = (landmarks[NOSE].x, landmarks[NOSE].y, landmarks[NOSE].z)
	left_ear = (landmarks[LEFT_EAR].x, landmarks[LEFT_EAR].y, landmarks[LEFT_EAR].z)
	right_ear = (landmarks[RIGHT_EAR].x, landmarks[RIGHT_EAR].y, landmarks[RIGHT_EAR].z)
	mid_ear = _avg(left_ear, right_ear)
	ear_vec = _sub(right_ear, left_ear)
	nose_vec = _sub(nose, mid_ear)
	roll = math.atan2(ear_vec[1], ear_vec[0])
	yaw = math.atan2(nose_vec[0], -nose_vec[2] if nose_vec[2] != 0 else -1e-6)
	pitch = math.atan2(nose_vec[1], -nose_vec[2] if nose_vec[2] != 0 else -1e-6)
	return (yaw, pitch, roll)


def _torso_angles_and_center(landmarks):
	left_shoulder = (
		landmarks[LEFT_SHOULDER].x,
		landmarks[LEFT_SHOULDER].y,
		landmarks[LEFT_SHOULDER].z,
	)
	right_shoulder = (
		landmarks[RIGHT_SHOULDER].x,
		landmarks[RIGHT_SHOULDER].y,
		landmarks[RIGHT_SHOULDER].z,
	)
	left_hip = (landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y, landmarks[LEFT_HIP].z)
	right_hip = (landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y, landmarks[RIGHT_HIP].z)
	shoulder_vec = _sub(right_shoulder, left_shoulder)
	mid_shoulder = _avg(left_shoulder, right_shoulder)
	mid_hip = _avg(left_hip, right_hip)
	torso_vec = _sub(mid_shoulder, mid_hip)
	roll = math.atan2(shoulder_vec[1], shoulder_vec[0])
	yaw = math.atan2(shoulder_vec[0], -shoulder_vec[2] if shoulder_vec[2] != 0 else -1e-6)
	pitch = math.atan2(torso_vec[1], -torso_vec[2] if torso_vec[2] != 0 else -1e-6)
	return (yaw, pitch, roll), mid_hip


def _angular_speed(prev_angles, angles, dt):
	if prev_angles is None or dt <= 0:
		return 0.0
	dyaw = _wrap_angle(angles[0] - prev_angles[0])
	dpitch = _wrap_angle(angles[1] - prev_angles[1])
	droll = _wrap_angle(angles[2] - prev_angles[2])
	magnitude = math.sqrt(dyaw * dyaw + dpitch * dpitch + droll * droll)
	return math.degrees(magnitude) / dt




def _open_capture(video_path: str | None) -> cv2.VideoCapture:
	if video_path:
		return cv2.VideoCapture(video_path)
	return cv2.VideoCapture(0)


def _get_timestamp(now: float, cap: cv2.VideoCapture, use_video_ts: bool) -> float:
	if use_video_ts:
		pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
		if pos_ms > 0:
			return pos_ms / 1000.0
	return now


def main() -> None:
	parser = argparse.ArgumentParser(description="Pose tracker with metrics.")
	parser.add_argument("--video", help="Path to a video file instead of webcam.")
	args = parser.parse_args()

	script_dir = os.path.dirname(os.path.abspath(__file__))
	model_path = os.path.join(script_dir, "models", "pose_landmarker_lite.task")
	ensure_model(model_path)
	log_path = os.path.join(script_dir, "pose_metrics.txt")

	options = vision.PoseLandmarkerOptions(
		base_options=base_options.BaseOptions(model_asset_path=model_path),
		running_mode=vision.RunningMode.VIDEO,
		min_pose_detection_confidence=0.5,
		min_pose_presence_confidence=0.5,
		min_tracking_confidence=0.5,
	)

	cap = _open_capture(args.video)
	if not cap.isOpened():
		raise RuntimeError("Could not open video source")
	cap_fps = cap.get(cv2.CAP_PROP_FPS)
	frame_dt = 1.0 / cap_fps if cap_fps and cap_fps > 0 else 1.0 / 30.0

	connections = pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS
	prev_time = None
	prev_head_angles = None
	prev_torso_angles = None
	prev_mid_hip = None
	current_event_start = None
	current_event_max_trunk_ang = 0.0
	current_event_max_trunk_trans = 0.0
	events = deque()
	total_events = 0
	head_ang_vel_ema = 0.0
	torso_ang_vel_ema = 0.0
	last_log_time = 0.0
	last_console_time = 0.0
	log_interval_s = 1.0

	log_file = open(log_path, "a", encoding="utf-8")
	last_timestamp_s = None

	try:
		with vision.PoseLandmarker.create_from_options(options) as landmarker:
			while True:
				ok, frame = cap.read()
				if not ok:
					break

				now = time.time()
				timestamp_s = _get_timestamp(now, cap, args.video is not None)
				if last_timestamp_s is not None and timestamp_s <= last_timestamp_s:
					timestamp_s = last_timestamp_s + frame_dt
				last_timestamp_s = timestamp_s
				dt = 0.0 if prev_time is None else timestamp_s - prev_time
				prev_time = timestamp_s

				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
				timestamp_ms = int(timestamp_s * 1000)
				result = landmarker.detect_for_video(mp_image, timestamp_ms)

				if result.pose_landmarks:
					for landmarks in result.pose_landmarks:
						draw_pose(frame, landmarks, connections)

					landmarks = result.pose_landmarks[0]
					head_angles = _angles_from_landmarks(landmarks)
					torso_angles, mid_hip = _torso_angles_and_center(landmarks)
					head_ang_vel = _angular_speed(prev_head_angles, head_angles, dt)
					torso_ang_vel = _angular_speed(prev_torso_angles, torso_angles, dt)
					head_ang_vel_ema = (
						EMA_ALPHA * head_ang_vel + (1.0 - EMA_ALPHA) * head_ang_vel_ema
					)
					torso_ang_vel_ema = (
						EMA_ALPHA * torso_ang_vel + (1.0 - EMA_ALPHA) * torso_ang_vel_ema
					)
					if prev_mid_hip is None or dt <= 0:
						torso_trans_speed = 0.0
					else:
						dx = mid_hip[0] - prev_mid_hip[0]
						dy = mid_hip[1] - prev_mid_hip[1]
						torso_trans_speed = math.sqrt(dx * dx + dy * dy) / dt

					prev_head_angles = head_angles
					prev_torso_angles = torso_angles
					prev_mid_hip = mid_hip

					if head_ang_vel_ema > HAV_EVENT_THRESHOLD_DEG_S:
						if current_event_start is None:
							current_event_start = timestamp_s
							current_event_max_trunk_ang = 0.0
							current_event_max_trunk_trans = 0.0
						current_event_max_trunk_ang = max(
							current_event_max_trunk_ang, torso_ang_vel_ema
						)
						current_event_max_trunk_trans = max(
							current_event_max_trunk_trans, torso_trans_speed
						)
					else:
						if current_event_start is not None:
							duration = timestamp_s - current_event_start
							if duration >= HAV_EVENT_MIN_DURATION_S:
								is_wb = (
									current_event_max_trunk_ang
									>= WB_TORSO_ANGULAR_THRESHOLD_DEG_S
									or current_event_max_trunk_trans
									>= WB_TORSO_TRANSLATION_THRESHOLD
								)
								events.append((timestamp_s, is_wb))
								total_events += 1
								timestamp = time.strftime(
									"%Y-%m-%d %H:%M:%S", time.localtime(now)
								)
								log_file.write(
									f"{timestamp}, EVENT=1, TOTAL={total_events}\n"
								)
							current_event_start = None
							current_event_max_trunk_ang = 0.0
							current_event_max_trunk_trans = 0.0

					while events and timestamp_s - events[0][0] > WINDOW_SECONDS:
						events.popleft()

					window_events = list(events)
					hmf = len(window_events) * (60.0 / WINDOW_SECONDS)
					wb_count = sum(1 for _, is_wb in window_events if is_wb)
					no_count = len(window_events) - wb_count
					wb_no_ratio = wb_count / max(1, no_count)

					overlay_lines = [
						f"HAV: {head_ang_vel_ema:.1f} deg/s (slow<{HAV_SLOW_THRESHOLD_DEG_S:.0f})",
						f"HMF: {hmf:.1f} events/min",
						f"WB/NO ratio (60s): {wb_no_ratio:.2f}",
					]
					y = 24
					for line in overlay_lines:
						cv2.putText(
							frame,
							line,
							(10, y),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.6,
							(255, 255, 255),
							2,
						)
						y += 22

					if timestamp_s - last_log_time >= log_interval_s:
						timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
						log_file.write(
							f"{timestamp}, HAV={head_ang_vel_ema:.2f}, HMF={hmf:.2f}, "
							f"WB_NO={wb_no_ratio:.3f}, TOTAL={total_events}\n"
						)
						log_file.flush()
						last_log_time = timestamp_s

					if timestamp_s - last_console_time >= log_interval_s:
						print(
							f"HAV={head_ang_vel_ema:.2f} deg/s | "
							f"HMF={hmf:.2f} events/min | WB/NO={wb_no_ratio:.2f}"
						)
						last_console_time = timestamp_s

				cv2.imshow("Pose Tracker", frame)
				if cv2.waitKey(1) & 0xFF == 27:
					break
	finally:
		log_file.close()

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
