import os
import subprocess
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
def find_siren_audio():
	assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
	for filename in [
		"odyssey_siren 12.18.41 AM.wav"
	]:
		candidate = os.path.join(assets_dir, filename)
		if os.path.exists(candidate):
			return candidate
	return None

siren_file = find_siren_audio()
siren_process = None
window_name = "Drowsiness Detection - Sleep Monitor"

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while True:
	ret, frame = cap.read()
	if not ret:
		continue
	frame = imutils.resize(frame, width=600)
	h, w = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

	has_face = len(subjects) > 0
	current_ear = 0.0

	if has_face:
		ears = []
		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ears.append((leftEAR + rightEAR) / 2.0)
		current_ear = min(ears)
		if current_ear < thresh:
			flag += 1
		else:
			flag = 0
	else:
		flag = 0

	is_drowsy = flag >= frame_check

	# Siren audio control: once triggered, let the whole file ring to completion
	if is_drowsy:
		if siren_process is None or siren_process.poll() is not None:
			if siren_file and os.path.exists(siren_file):
				try:
					siren_process = subprocess.Popen(["afplay", "-v", "8", siren_file])
				except Exception:
					pass

	# Top Header Bar (dark backdrop)
	cv2.rectangle(frame, (0, 0), (w, 80), (25, 25, 25), -1)

	# Status text and EAR metrics without overlapping
	if not has_face:
		cv2.putText(frame, "STATUS: NO FACE DETECTED", (15, 34),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
		cv2.putText(frame, "Please look towards the camera", (15, 65),
			cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
	elif is_drowsy:
		cv2.putText(frame, "STATUS: SLEEPY (ALERT!)", (15, 34),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
		cv2.putText(frame, f"EAR: {current_ear:.2f} (Below {thresh:.2f})  |  Closed Frames: {flag}", (15, 65),
			cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 255), 2)
		# Red alert border
		cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
	elif flag > 0:
		cv2.putText(frame, f"STATUS: CLOSING EYES ({flag}/{frame_check})", (15, 34),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
		cv2.putText(frame, f"EAR: {current_ear:.2f} (Threshold: {thresh:.2f})", (15, 65),
			cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
	else:
		cv2.putText(frame, "STATUS: AWAKE / NORMAL", (15, 34),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		cv2.putText(frame, f"EAR: {current_ear:.2f}  |  Threshold: {thresh:.2f}", (15, 65),
			cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

	# Bottom Footer Bar
	if is_drowsy:
		cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 220), -1)
		cv2.putText(frame, "WAKE UP!", (25, h - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
		cv2.putText(frame, "[q: Quit]", (w - 85, h - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
	else:
		cv2.rectangle(frame, (0, h - 35), (w, h), (25, 25, 25), -1)
		cv2.putText(frame, "Driver Monitoring Active", (15, h - 12),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
		cv2.putText(frame, "Press 'q' to quit", (w - 130, h - 12),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

	cv2.imshow(window_name, frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

if siren_process is not None and siren_process.poll() is None:
	try:
		siren_process.terminate()
	except Exception:
		pass

cv2.destroyAllWindows()
cap.release() 
