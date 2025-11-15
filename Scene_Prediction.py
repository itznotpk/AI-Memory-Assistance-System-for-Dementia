# scene_detector.py
import cv2
import numpy as np
import time
import json
import os
from collections import deque
from ultralytics import YOLO

# Optional Firebase (Firestore) support
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, messaging  # add messaging
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

# -------- Config --------
SPEC_EVERY_N = 3  # run spectacles model every N frames
LAST_SEEN_JSON_PATH = 'last_spec_seen.json'
PRESENCE_PUSH_INTERVAL = 15  # seconds between periodic location updates

# Firebase config (set FIREBASE_ENABLED=True and provide a service account JSON to enable)
FIREBASE_ENABLED = True
FIREBASE_CREDENTIALS = 'firebase_service_account.json'
FIREBASE_COLLECTION = 'last_seen'
FIREBASE_DOC = 'spectacle'

# New: patient presence doc + optional push notifications
FIREBASE_PRESENCE_COLLECTION = 'presence'
FIREBASE_PRESENCE_DOC = 'patient_location'
PUSH_NOTIFICATIONS_ENABLED = True  # set True if you want FCM pushes
FCM_TOPIC = 'caregivers'            # or leave token None to use topic
FCM_DEVICE_TOKEN = None             # set a device token string to target a device

# Load your trained models
model = YOLO('my_model.pt', task='detect')            # kitchen anchors
spec_model = YOLO('my_model_spec.pt', task='detect')  # spectacles-only model

# Object label canonical names (exactly as produced by your model)
KITCHEN_OBJECTS = {'Basin', 'Pot', 'Stove', 'Kettle', 'Fridge'}

# Per-object box colors (BGR)
BOX_COLORS = {
    'Stove':  (0, 0, 255),      # red
    'Fridge': (255, 0, 0),      # blue
    'Basin':  (0, 255, 255),    # yellow
    'Pot':    (0, 165, 255),    # orange
    'Kettle': (255, 0, 255),    # magenta
}
SPEC_COLOR = (255, 255, 0)      # cyan for spectacles

# Per-object weights
WEIGHTS = {
    'Stove': 3.0,
    'Fridge': 2.5,
    'Basin': 1.5,
    'Pot': 1.0,
    'Kettle': 1.0,
}

# Confidence thresholds
THRESHOLDS = {
    'Stove': 0.55,
    'Fridge': 0.60,
    'Basin': 0.50,
    'Pot': 0.50,
    'Kettle': 0.50,
}
SPEC_THRESHOLD = 0.60  # spectacles confidence to accept
SPECTACLE_LABELS = {'Spectacle'}

# Toggle drawing spec box (remain off)
DRAW_SPEC_BOX = False

STABLE_WINDOW = 5
STABLE_REQUIRED = 3

recent_decisions = deque(maxlen=STABLE_WINDOW)
last_reason = "awaiting sufficient evidence"

# Track last time/place spectacles were seen
last_spec_seen = {
    'place': None,       # "Kitchen" or "Unknown"
    'time': None,        # epoch seconds
    'conf': None,        # float
    'bbox': None,        # (x1,y1,x2,y2)
    'label': None        # label from model
}

# Firebase state
db = None
def init_firebase():
    global db
    if not (FIREBASE_ENABLED and FIREBASE_AVAILABLE and os.path.exists(FIREBASE_CREDENTIALS)):
        return
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized.")
    except Exception as e:
        print(f"Firebase init failed: {e}")

def load_last_seen():
    global last_spec_seen
    if os.path.exists(LAST_SEEN_JSON_PATH):
        try:
            with open(LAST_SEEN_JSON_PATH, 'r') as f:
                data = json.load(f)
            # Basic validation
            last_spec_seen.update({
                'place': data.get('place'),
                'time': data.get('time'),
                'conf': data.get('conf'),
                'bbox': tuple(data['bbox']) if isinstance(data.get('bbox'), list) else data.get('bbox'),
                'label': data.get('label')
            })
            print("Loaded last_spec_seen from JSON.")
        except Exception as e:
            print(f"Failed to load {LAST_SEEN_JSON_PATH}: {e}")

def save_last_seen():
    # Local JSON
    try:
        data = dict(last_spec_seen)
        if isinstance(data.get('bbox'), tuple):
            data['bbox'] = list(data['bbox'])
        with open(LAST_SEEN_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to save {LAST_SEEN_JSON_PATH}: {e}")
    # Firestore
    try:
        if db is not None:
            payload = dict(last_spec_seen)
            if isinstance(payload.get('bbox'), tuple):
                payload['bbox'] = list(payload['bbox'])
            # Also include ISO time for easy reading
            if payload.get('time'):
                payload['time_iso'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(payload['time']))
            db.collection(FIREBASE_COLLECTION).document(FIREBASE_DOC).set(payload)
    except Exception as e:
        print(f"Failed to write to Firebase: {e}")

def evaluate_frame(object_dict):
    objs = {lbl: conf for lbl, conf in object_dict.items() if lbl in KITCHEN_OBJECTS}

    if 'Stove' in objs and objs['Stove'] >= THRESHOLDS['Stove']:
        return True, f"Stove conf={objs['Stove']:.2f}", WEIGHTS['Stove']
    if ('Fridge' in objs and objs['Fridge'] >= THRESHOLDS['Fridge'] and
        any(o in objs and objs[o] >= THRESHOLDS[o] for o in ['Basin', 'Pot', 'Kettle'])):
        return True, f"Fridge+support conf={objs['Fridge']:.2f}", WEIGHTS['Fridge']
    if ('Basin' in objs and 'Pot' in objs and
        objs['Basin'] >= THRESHOLDS['Basin'] and objs['Pot'] >= THRESHOLDS['Pot']):
        return True, "Basin+Pot combo", WEIGHTS['Basin'] + WEIGHTS['Pot']

    distinct_valid = [o for o, c in objs.items() if c >= THRESHOLDS[o]]
    if len(distinct_valid) >= 3:
        return True, f"3+ objects: {distinct_valid}", sum(WEIGHTS[o] for o in distinct_valid)

    score = 0.0
    for o, conf in objs.items():
        if conf >= THRESHOLDS[o]:
            score += WEIGHTS[o]

    if all(x in objs and objs[x] >= THRESHOLDS[x] for x in ['Basin', 'Pot']):
        score += 1.0
    if 'Stove' in objs and objs['Stove'] >= THRESHOLDS['Stove'] and 'Kettle' in objs and objs['Kettle'] >= THRESHOLDS['Kettle']:
        score += 0.5

    if score >= 3.0:
        return True, f"score {score:.2f} >= 3.0", score

    return False, "insufficient combination", score

def stable_kitchen(decision):
    recent_decisions.append(decision)
    return recent_decisions.count(True) >= STABLE_REQUIRED

def detect_objects(frame):
    result = model(frame, verbose=False)[0]
    best = {}
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        if label not in best or conf > best[label]:
            best[label] = conf
    return result.boxes, best, names

def detect_spectacles(frame):
    result = spec_model(frame, verbose=False)[0]
    return result.boxes, result.names

def push_presence_update(location, reason, is_kitchen, score=None):
    """Write current patient presence to Firestore (without spectacles data)."""
    if db is None:
        return
    try:
        ts = time.time()
        payload = {
            'location': location,            # 'Kitchen' or 'Unknown'
            'is_kitchen': bool(is_kitchen),
            'reason': reason or '',
            'score': float(score) if score is not None else None,
            'time': ts,
            'time_iso': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        }
        db.collection(FIREBASE_PRESENCE_COLLECTION).document(FIREBASE_PRESENCE_DOC).set(payload)
    except Exception as e:
        print(f"Presence update failed: {e}")

# Init persisted state and Firebase
load_last_seen()
init_firebase()

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

frame_idx = 0
prev_kitchen_now = None
last_presence_push = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Kitchen anchors every frame
    boxes, best_conf, names = detect_objects(frame)
    frame_decision, reason, score = evaluate_frame(best_conf)
    kitchen_now = stable_kitchen(frame_decision)

    # Push presence when state changes or periodically
    now = time.time()
    if (prev_kitchen_now is None) or (kitchen_now != prev_kitchen_now) or (now - last_presence_push >= PRESENCE_PUSH_INTERVAL):
        loc = 'Kitchen' if kitchen_now else 'Unknown'
        push_presence_update(loc, reason, kitchen_now, score)
        last_presence_push = now
        prev_kitchen_now = kitchen_now

    # 2) Spectacles detection on schedule
    run_spec_now = (frame_idx % SPEC_EVERY_N == 0)
    best_spec_conf = None
    best_spec_bbox = None
    best_spec_label = None

    if run_spec_now:
        spec_boxes, spec_names = detect_spectacles(frame)
        only_one_class = len(spec_names) == 1

        for sbox in spec_boxes:
            cls_id = int(sbox.cls[0])
            conf = float(sbox.conf[0])
            label = spec_names[cls_id]
            if (not only_one_class) and (label not in SPECTACLE_LABELS):
                continue
            if best_spec_conf is None or conf > best_spec_conf:
                best_spec_conf = conf
                best_spec_bbox = tuple(map(int, sbox.xyxy[0]))
                best_spec_label = label

        # Update last seen (Kitchen or Unknown) and persist
        if best_spec_conf is not None and best_spec_conf >= SPEC_THRESHOLD:
            place = 'Kitchen' if kitchen_now else 'Unknown'
            last_spec_seen.update({
                'place': place,
                'time': time.time(),
                'conf': best_spec_conf,
                'bbox': best_spec_bbox,
                'label': best_spec_label
            })
            save_last_seen()

    # Top-left banner when kitchen confirmed
    if kitchen_now:
        banner_text = "Patients is in Kitchen"
        reason_text = reason
        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        (rw, rh), _ = cv2.getTextSize(reason_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pad = 10
        box_w = max(bw, rw) + pad * 2
        box_h = bh + rh + pad * 3
        cv2.rectangle(frame, (5, 5), (5 + box_w, 5 + box_h), (0, 200, 0), -1)
        cv2.rectangle(frame, (5, 5), (5 + box_w, 5 + box_h), (0, 120, 0), 2)
        cv2.putText(frame, banner_text, (5 + pad, 5 + pad + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, reason_text, (5 + pad, 5 + pad * 2 + bh + rh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw kitchen anchor boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = BOX_COLORS.get(label, (100, 100, 100))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Optional: draw spectacles box (kept off unless enabled)
    if DRAW_SPEC_BOX and best_spec_conf is not None and best_spec_conf >= SPEC_THRESHOLD and best_spec_bbox:
        x1, y1, x2, y2 = best_spec_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), SPEC_COLOR, 2)
        spec_text = f"{best_spec_label or 'Spec'} {best_spec_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(spec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), SPEC_COLOR, -1)
        cv2.putText(frame, spec_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Top-right "last seen of spectacle" panel
    if last_spec_seen['time'] is not None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_spec_seen['time']))
        lines = [
            "Spectacle last seen",
            f"Location: {last_spec_seen['place'] or 'N/A'}",
            f"Time: {ts}"
        ]
        pad = 8
        sizes = []
        w = 0; h_total = 0
        for i, t in enumerate(lines):
            scale = 0.7 if i == 0 else 0.55
            thickness = 2 if i == 0 else 1
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            sizes.append((tw, th, scale, thickness))
            w = max(w, tw)
            h_total += th + (6 if i < len(lines)-1 else 0)
        box_w = w + pad*2
        box_h = h_total + pad*2
        x0 = frame.shape[1] - box_w - 5
        y0 = 5
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (200, 220, 255), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (120, 140, 180), 2)
        y = y0 + pad
        for (t, (tw, th, scale, thickness)) in zip(lines, sizes):
            cv2.putText(frame, t, (x0 + pad, y + th),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
            y += th + 6

    cv2.imshow("Kitchen Anchor + Spectacles Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()