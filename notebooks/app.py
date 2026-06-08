import os
import cv2
import numpy as np
import pandas as pd
import requests

import streamlit as st
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("Face Recognition Attendance")

from datetime import datetime
from PIL import Image
import onnxruntime as ort

KNOWN_FACES_DIR = "known_Faces"
ATTENDANCE_FILE = "attendance.csv"

ARCFACE_MODEL_PATH = "arcface.onnx"
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

DEFAULT_DETECTION_CONF = 0.80
DEFAULT_COSINE_THRESHOLD = 0.40
MIN_FACE_SIZE_PX = 80

USE_PREFILTER = True
PREFILTER_DIMS = 64
PREFILTER_TOPK = 30

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Index", "Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

@st.cache_resource
def download_models() -> bool:
    def _download(url: str, path: str, stream: bool = False):
        r = requests.get(url, allow_redirects=True, stream=stream, timeout=60)
        r.raise_for_status()
        if stream:
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        else:
            with open(path, "wb") as f:
                f.write(r.content)

    try:
        if not os.path.exists(ARCFACE_MODEL_PATH):
            st.info("Downloading ArcFace model...")
            _download(
                "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx",
                ARCFACE_MODEL_PATH,
                stream=True,
            )

        if not os.path.exists(FACE_PROTO):
            _download(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                FACE_PROTO,
                stream=False,
            )

        if not os.path.exists(FACE_MODEL):
            st.info("Downloading face detector model...")
            _download(
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                FACE_MODEL,
                stream=True,
            )

        return True
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

if not download_models():
    st.stop()

@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, os.cpu_count() or 1)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    arc = ort.InferenceSession(
        ARCFACE_MODEL_PATH,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    input_name = arc.get_inputs()[0].name
    return face_net, arc, input_name

face_net, arcface_session, arcface_input_name = load_models()

def detect_faces(image_bgr: np.ndarray, conf_threshold: float):
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    det = face_net.forward()

    faces = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        fw, fh = (x2 - x1), (y2 - y1)
        if fw < MIN_FACE_SIZE_PX or fh < MIN_FACE_SIZE_PX:
            continue

        area = fw * fh
        faces.append(((x1, y1, x2, y2), conf, area))

    return faces

def pick_best_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda x: (x[2], x[1]))

def crop_face_tight(image_bgr: np.ndarray, box):
    x1, y1, x2, y2 = box
    h, w = image_bgr.shape[:2]

    fw, fh = x2 - x1, y2 - y1
    pad = int(min(fw, fh) * 0.08)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    face = image_bgr[y1:y2, x1:x2]
    return face

def preprocess_for_arcface(face_bgr: np.ndarray):
    face = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.expand_dims(face, axis=0)
    return face

def get_embedding(face_bgr: np.ndarray):
    x = preprocess_for_arcface(face_bgr)
    emb = arcface_session.run(None, {arcface_input_name: x})[0].reshape(-1).astype(np.float32)
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb

def parse_filename(file: str):
    parts = file.rsplit("_", 1)
    if len(parts) != 2:
        return None
    index = parts[0]
    name = os.path.splitext(parts[1])[0]
    return index, name

def cosine_distance_vectorized(known_embs: np.ndarray, test_emb: np.ndarray):
    sims = known_embs @ test_emb
    dists = 1.0 - sims
    return dists, sims

@st.cache_data(show_spinner=False)
def load_known_faces_silent():
    files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    embs = []
    meta = []

    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            continue
        img = cv2.imread(os.path.join(KNOWN_FACES_DIR, f))
        if img is None:
            continue

        faces = detect_faces(img, conf_threshold=0.5)
        best = pick_best_face(faces)
        if best is None:
            continue

        face = crop_face_tight(img, best[0])
        if face.size == 0:
            continue

        emb = get_embedding(face)
        if emb is None or emb.shape[0] != 512:
            continue

        embs.append(emb)
        meta.append(parsed)

    if not embs:
        return np.empty((0, 512), dtype=np.float32), [], None

    known = np.vstack(embs).astype(np.float32)

    if USE_PREFILTER:
        pf = known[:, :PREFILTER_DIMS].copy()
        norms = np.linalg.norm(pf, axis=1, keepdims=True)
        pf = pf / np.clip(norms, 1e-12, None)
        return known, meta, pf

    return known, meta, None

def match_face(test_emb: np.ndarray, known_embs: np.ndarray, meta, cosine_thresh: float, prefilter=None):
    if known_embs.shape[0] == 0:
        return None, None

    candidate_idx = None
    if prefilter is not None and known_embs.shape[0] > PREFILTER_TOPK:
        te = test_emb[:PREFILTER_DIMS]
        te = te / max(np.linalg.norm(te), 1e-12)
        d_pf = 1.0 - (prefilter @ te)
        candidate_idx = np.argsort(d_pf)[:PREFILTER_TOPK]
        cand_embs = known_embs[candidate_idx]
    else:
        cand_embs = known_embs

    dists, sims = cosine_distance_vectorized(cand_embs, test_emb)
    j = int(np.argmin(dists))
    best_dist = float(dists[j])
    best_sim = float(sims[j])

    best_global = int(candidate_idx[j]) if candidate_idx is not None else j

    scores = {"cosine_distance": best_dist, "cosine_similarity": best_sim}
    if best_dist < cosine_thresh:
        return meta[best_global], scores
    return None, scores

def mark_attendance(index: str, name: str):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)
    if not ((df["Index"] == index) & (df["Date"] == date)).any():
        df.loc[len(df)] = [index, name, date, time]
        df.to_csv(ATTENDANCE_FILE, index=False)
        return "Attendance marked"
    return "Attendance already marked today"



with st.sidebar:
    st.header("Settings")
    detection_conf = st.slider("Detection confidence", 0.3, 0.95, DEFAULT_DETECTION_CONF, 0.05)
    cosine_thresh = st.slider("Cosine distance threshold", 0.2, 0.7, DEFAULT_COSINE_THRESHOLD, 0.05)

    if st.button("Reload known faces"):
        st.cache_data.clear()
        st.rerun()

known_embeddings, known_meta, known_prefilter = load_known_faces_silent()
if known_embeddings.shape[0] == 0:
    st.warning("No known faces found. Add images to `known_Faces` as `index_name.jpg`.")
    st.stop()

mode = st.radio("Input", ["Upload Image", "Camera"], horizontal=True)

image_bgr = None
if mode == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        pil = Image.open(file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
else:
    cam = st.camera_input("Capture an image")
    if cam:
        pil = Image.open(cam).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

if image_bgr is not None:
    faces = detect_faces(image_bgr, conf_threshold=detection_conf)
    best = pick_best_face(faces)

    if best is None:
        st.error("No face detected.")
    else:
        box, conf, _area = best
        face_bgr = crop_face_tight(image_bgr, box)
        emb = get_embedding(face_bgr)

        match, scores = match_face(
            emb,
            known_embeddings,
            known_meta,
            cosine_thresh=cosine_thresh,
            prefilter=known_prefilter,
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), caption="Detected face", use_column_width=True)
            st.caption(f"Detection confidence: {conf:.2%}")

        with col2:
            st.write(f"Cosine distance: `{scores['cosine_distance']:.4f}`  (match if < `{cosine_thresh}`)")
            st.write(f"Cosine similarity: `{scores['cosine_similarity']:.4f}`")

            if match is None:
                st.error("Not recognized.")
            else:
                index, name = match
                st.success(f"Recognized: {name} ({index})")
                msg = mark_attendance(index, name)
                if "already" in msg.lower():
                    st.warning(msg)
                else:
                    st.success(msg)

st.divider()
st.subheader("Attendance Records")

df = pd.read_csv(ATTENDANCE_FILE)
if len(df) == 0:
    st.info("No attendance records yet.")
else:
    col1, col2 = st.columns(2)
    with col1:
        date_filter = st.date_input("Filter by date", value=datetime.now())
    with col2:
        name_filter = st.multiselect("Filter by name", options=sorted(df["Name"].unique().tolist()))

    out = df.copy()
    if date_filter:
        out = out[out["Date"] == date_filter.strftime("%Y-%m-%d")]
    if name_filter:
        out = out[out["Name"].isin(name_filter)]

    st.dataframe(out, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
