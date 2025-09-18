
import base64
import io
import os
import time
import uuid
import mysql.connector
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

from head_pose import process_head_pose
from eye_movement import process_eye_movement


app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

LOG_DIR = os.path.join(BASE_DIR, "log")

@app.route("/log/<path:filename>")
def serve_log(filename):
    return send_from_directory(LOG_DIR, filename)


DB_CONFIG = {
    "host": "localhost",
    "user": "root",          
    "password": "admin@123",  
    "database": "dilsdb"
}


def init_db() -> None:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(64),
            candidate VARCHAR(255),
            event_type VARCHAR(64),
            detail TEXT,
            timestamp_utc DATETIME,
            snapshot_path VARCHAR(512)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id VARCHAR(64) PRIMARY KEY,
            candidate VARCHAR(255),
            started_utc DATETIME,
            ended_utc DATETIME,
            recording_path VARCHAR(512)
        )
        """
    )
    conn.commit()
    conn.close()


init_db()



try:
    generic_model = YOLO("yolov8n.pt")
except Exception:
    generic_model = None



class FocusState:
    def __init__(self) -> None:
        self.calibrated_angles = None
        self.calibration_start = None
        self.last_face_present_ts = time.time()
        self.last_looking_at_screen_ts = time.time()
        self.last_head_straight_ts = time.time()
        self.last_eye_straight_ts = time.time()
        self.multiple_faces_last_ts = 0.0
        self.no_face_flag_emitted = False
        self.not_looking_flag_emitted = False
        self.head_off_flag_emitted = False
        self.eye_off_flag_emitted = False


session_state = {}


FOCUS_AWAY_THRESHOLD_SEC = 5
NO_FACE_THRESHOLD_SEC = 10
HEAD_OFF_THRESHOLD_SEC = 5
EYE_OFF_THRESHOLD_SEC = 5


def decode_base64_image(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame


def save_snapshot(frame: np.ndarray, prefix: str) -> str:
    ts = int(time.time())
    filename = f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(LOG_DIR, filename)
    cv2.imwrite(path, frame)
    return path


def log_event(session_id: str, candidate: str, event_type: str, detail: str, snapshot_path: str | None = None) -> None:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (session_id, candidate, event_type, detail, timestamp_utc, snapshot_path) VALUES (%s,%s,%s,%s,%s,%s)",
        (session_id, candidate, event_type, detail, datetime.utcnow(), snapshot_path),
    )
    conn.commit()
    conn.close()


def ensure_session(session_id: str, candidate: str) -> None:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT id FROM sessions WHERE id=%s", (session_id,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO sessions (id, candidate, started_utc) VALUES (%s,%s,%s)",
            (session_id, candidate, datetime.utcnow()),
        )
        conn.commit()
    conn.close()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/recordings/<path:filename>")
def serve_recording(filename: str):
    return send_from_directory(RECORDINGS_DIR, filename)


@app.route("/api/report/<session_id>")
def api_report(session_id: str):
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT candidate, started_utc, ended_utc, recording_path FROM sessions WHERE id=%s", (session_id,))
    srow = cur.fetchone()
    if not srow:
        conn.close()
        return jsonify({"error": "Session not found"}), 404
    candidate, started_utc, ended_utc, recording_path = srow

    # Convert recording_path to just filename for browser access
    recording_file = os.path.basename(recording_path) if recording_path else None

    cur.execute(
        "SELECT event_type, detail, timestamp_utc, snapshot_path FROM events WHERE session_id=%s ORDER BY timestamp_utc",
        (session_id,),
    )
    events = []
    for r in cur.fetchall():
        snapshot_file = os.path.basename(r[3]) if r[3] else None
        events.append({
            "event_type": r[0],
            "detail": r[1],
            "timestamp": r[2].isoformat() if r[2] else None,
            "snapshot": snapshot_file
        })

    conn.close()

    focus_lost = sum(1 for e in events if e["event_type"] == "focus_lost")
    suspicious = [e for e in events if e["event_type"] in ("no_face", "multiple_faces", "item_detected")]

    return jsonify({
        "session_id": session_id,
        "candidate": candidate,
        "started_utc": started_utc.isoformat() if started_utc else None,
        "ended_utc": ended_utc.isoformat() if ended_utc else None,
        # Provide only filename for front-end to use
        "recording_path": recording_file,
        "num_focus_lost": focus_lost,
        "events": events,
        "suspicious_events": suspicious,
    })


@app.route("/report/<session_id>")
def report_page(session_id: str):
    return render_template("report.html", session_id=session_id)


@app.post("/api/upload_recording")
def upload_recording():
    session_id = request.form.get("session_id", "")
    candidate = request.form.get("candidate", "")
    ensure_session(session_id, candidate)
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    filename = f"{session_id}_{uuid.uuid4().hex[:6]}.webm"
    save_path = os.path.join(RECORDINGS_DIR, filename)
    f.save(save_path)

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET ended_utc=%s, recording_path=%s WHERE id=%s",
        (datetime.utcnow(), save_path, session_id),
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "path": save_path})


@socketio.on("connect")
def on_connect():
    emit("connected", {"message": "connected"})


@socketio.on("frame")
def on_frame(data):
    session_id = data.get("session_id") or str(uuid.uuid4())
    candidate = data.get("candidate") or "Anonymous"
    ensure_session(session_id, candidate)
    

    if session_id not in session_state:
        session_state[session_id] = FocusState()
        session_state[session_id].calibration_start = time.time()

    state = session_state[session_id]

    frame = decode_base64_image(data.get("image", ""))
    if frame is None:
        return

    now = time.time()

    processed_frame, gaze_direction = process_eye_movement(frame.copy())
    processed_frame, head_state = process_head_pose(processed_frame, state.calibrated_angles)

    if state.calibrated_angles is None:
        if state.calibration_start is None:
            state.calibration_start = now
        _, maybe_angles = process_head_pose(frame.copy(), None)
        if isinstance(maybe_angles, tuple):
            state.calibrated_angles = maybe_angles

    looking_at_screen = head_state == "Looking at Screen" and gaze_direction in ("Looking Center", "Looking at Screen")
    if looking_at_screen:
        state.last_looking_at_screen_ts = now
        state.not_looking_flag_emitted = False

    face_present = head_state != "No Face"
    if face_present:
        state.last_face_present_ts = now
        state.no_face_flag_emitted = False

    head_straight = head_state == "Looking at Screen"
    if head_straight:
        state.last_head_straight_ts = now
        state.head_off_flag_emitted = False
    

    eye_straight = gaze_direction in ("Looking Center", "Looking at Screen")
    if eye_straight:
        state.last_eye_straight_ts = now
        state.eye_off_flag_emitted = False

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        num_faces = len(faces)
    except Exception:
        num_faces = 1

    items_detected = []
    if generic_model is not None:
        try:
            results = generic_model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf < 0.6:
                        continue
                    label = r.names.get(cls_id, str(cls_id)).lower()
                    if label in {"cell phone", "book", "laptop", "keyboard", "mouse", "tablet"}:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(processed_frame, f"{label} {conf:.2f}", (x1, max(10, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        items_detected.append(label)
        except Exception:
            pass

    events_emitted = []

    if now - state.last_looking_at_screen_ts > FOCUS_AWAY_THRESHOLD_SEC and not state.not_looking_flag_emitted:
        snap = save_snapshot(processed_frame, "focus_away")
        log_event(session_id, candidate, "focus_lost", f"Not looking for > {FOCUS_AWAY_THRESHOLD_SEC}s", snap)
        state.not_looking_flag_emitted = True
        events_emitted.append({"type": "focus_lost", "snapshot": snap})

    if now - state.last_face_present_ts > NO_FACE_THRESHOLD_SEC and not state.no_face_flag_emitted:
        snap = save_snapshot(processed_frame, "no_face")
        log_event(session_id, candidate, "no_face", f"No face for > {NO_FACE_THRESHOLD_SEC}s", snap)
        state.no_face_flag_emitted = True
        events_emitted.append({"type": "no_face", "snapshot": snap})

    head_warning = False
    if now - state.last_head_straight_ts > HEAD_OFF_THRESHOLD_SEC:
        head_warning = True
        if not state.head_off_flag_emitted:
            snap = save_snapshot(processed_frame, "head_off")
            log_event(session_id, candidate, "head_off", f"Head not straight > {HEAD_OFF_THRESHOLD_SEC}s", snap)
            state.head_off_flag_emitted = True
            events_emitted.append({"type": "head_off", "snapshot": snap})

    eye_warning = False
    if now - state.last_eye_straight_ts > EYE_OFF_THRESHOLD_SEC:
        eye_warning = True
        if not state.eye_off_flag_emitted:
            snap = save_snapshot(processed_frame, "eyes_off")
            log_event(session_id, candidate, "eyes_off", f"Eyes not straight > {EYE_OFF_THRESHOLD_SEC}s", snap)
            state.eye_off_flag_emitted = True
            events_emitted.append({"type": "eyes_off", "snapshot": snap})

    if num_faces >= 2 and now - state.multiple_faces_last_ts > 5:
        snap = save_snapshot(processed_frame, "multiple_faces")
        log_event(session_id, candidate, "multiple_faces", f"{num_faces} faces detected", snap)
        state.multiple_faces_last_ts = now
        events_emitted.append({"type": "multiple_faces", "count": num_faces, "snapshot": snap})

    if items_detected:
        snap = save_snapshot(processed_frame, "item")
        log_event(session_id, candidate, "item_detected", ", ".join(sorted(set(items_detected))), snap)
        events_emitted.append({"type": "item_detected", "items": items_detected, "snapshot": snap})
        cheating_detected = True  # Trigger warning for frontend
    else:
        cheating_detected = False

    display = cv2.resize(processed_frame, (640, int(processed_frame.shape[0] * 640 / processed_frame.shape[1])), interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode('.jpg', display)
    b64 = base64.b64encode(jpeg.tobytes()).decode('ascii')

    emit(
        "status",
        {
            "session_id": session_id,
            "candidate": candidate,
            "head_state": head_state,
            "gaze": gaze_direction,
            "num_faces": num_faces,
            "items": items_detected,
            "head_warning": head_warning,
            "eye_warning": eye_warning,
            "events": events_emitted,
            "cheating_detected": cheating_detected,
            "frame": f"data:image/jpeg;base64,{b64}",
        },
    )


@app.route("/candidates", methods=["GET"])
def get_candidates():
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT candidate FROM sessions")
    candidates = [row[0] for row in cur.fetchall()]
    conn.close()
    return jsonify({"candidates": candidates})

@app.route("/candidatereport",methods=["GET"])
def candidate_report_page():
    return render_template("candidatereport.html")

@app.route("/api/candidate/<name>", methods=["GET"])
def candidate_sessions(name: str):
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, started_utc, ended_utc, recording_path FROM sessions WHERE candidate=%s ORDER BY started_utc DESC",
        (name,),
    )
    sessions = []
    for r in cur.fetchall():
        sessions.append({
            "session_id": r[0],
            "started_utc": r[1],
            "ended_utc": r[2],
            # ✅ only return filename, not full path
            "recording_path": os.path.basename(r[3]) if r[3] else None,
        })
    conn.close()
    return jsonify({"candidate": name, "sessions": sessions})


def run():
    socketio.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
