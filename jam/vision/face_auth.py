import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
    LBPH_AVAILABLE = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    LBPH_AVAILABLE = False


def _resolve_haar_cascade_path() -> Optional[str]:
    """Locate OpenCV's frontal face Haar cascade across normal and PyInstaller builds."""
    candidates: List[str] = []
    if CV2_AVAILABLE and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(
            os.path.join(meipass, "cv2", "data", "haarcascade_frontalface_default.xml")
        )
        candidates.append(os.path.join(meipass, "haarcascade_frontalface_default.xml"))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _compute_lbp_histogram(gray_patch: np.ndarray) -> np.ndarray:
    """Compute a 32-bin Local Binary Pattern histogram for a grayscale cell."""
    padded = np.pad(gray_patch, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)

    offsets = [
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
        (1, 1), (1, 0), (1, -1), (0, -1),
    ]
    h, w = center.shape
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        code |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(code, bins=32, range=(0, 256))
    return hist.astype(np.float32)


def _compute_hog_histogram(gray_patch: np.ndarray) -> np.ndarray:
    """Compute an 8-bin gradient orientation histogram weighted by magnitude."""
    gx = cv2.Sobel(gray_patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_patch, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.floor((ang % 180.0) / 22.5).astype(np.int32) % 8
    hist = np.bincount(bins.ravel(), weights=mag.ravel(), minlength=8)
    return hist.astype(np.float32)


class JamFaceAuth:
    """
    100% local, offline face recognition engine for JAM:
    - Uses OpenCV Haar Cascade for fast local face detection (bundled in PyInstaller)
    - Extracts CLAHE-normalized 4x4 spatial LBP + HOG facial descriptors (cosine similarity)
    - Optionally trains OpenCV LBPHFaceRecognizer when opencv-contrib is installed
    - Stores only compact local templates in `<data_dir>/faces` (zero cloud calls)
    """

    FACE_SIZE = (128, 128)

    def __init__(self, faces_dir: str = "./jam_data/faces"):
        self.faces_dir = faces_dir
        self.profiles_path = os.path.join(self.faces_dir, "profiles.json")
        self.samples_path = os.path.join(self.faces_dir, "samples.npz")
        self.lbph_path = os.path.join(self.faces_dir, "lbph_model.yml")
        os.makedirs(self.faces_dir, exist_ok=True)

        self.cascade = None
        if CV2_AVAILABLE:
            cascade_path = _resolve_haar_cascade_path()
            if cascade_path:
                loaded = cv2.CascadeClassifier(cascade_path)
                if not loaded.empty():
                    self.cascade = loaded

        self.profiles: Dict[str, Dict[str, Any]] = self._load_profiles()

    def is_available(self) -> bool:
        return bool(CV2_AVAILABLE and self.cascade is not None)

    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.profiles_path):
            return {}
        try:
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_profiles(self) -> None:
        os.makedirs(self.faces_dir, exist_ok=True)
        with open(self.profiles_path, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, ensure_ascii=False, indent=2)

    def _load_lbph_samples(self):
        if not os.path.exists(self.samples_path):
            return [], []
        try:
            data = np.load(self.samples_path)
            images = list(data["images"])
            labels = list(data["labels"])
            return images, labels
        except Exception:
            return [], []

    def _save_and_train_lbph(self, label_id: int, new_faces: List[np.ndarray]) -> None:
        if not LBPH_AVAILABLE or not new_faces:
            return
        old_images, old_labels = self._load_lbph_samples()
        kept_images = []
        kept_labels = []
        for img, lbl in zip(old_images, old_labels):
            if int(lbl) != int(label_id):
                kept_images.append(img)
                kept_labels.append(int(lbl))

        for face in new_faces:
            kept_images.append(face.astype(np.uint8))
            kept_labels.append(int(label_id))

        if not kept_images:
            return

        np.savez_compressed(
            self.samples_path,
            images=np.asarray(kept_images, dtype=np.uint8),
            labels=np.asarray(kept_labels, dtype=np.int32),
        )

        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(kept_images, np.asarray(kept_labels, dtype=np.int32))
        recognizer.write(self.lbph_path)

    def _retrain_lbph_from_remaining(self) -> None:
        if not LBPH_AVAILABLE or not os.path.exists(self.samples_path):
            if os.path.exists(self.lbph_path):
                os.remove(self.lbph_path)
            return

        active_ids = {int(p["label_id"]) for p in self.profiles.values()}
        old_images, old_labels = self._load_lbph_samples()
        kept_images = [
            img for img, lbl in zip(old_images, old_labels) if int(lbl) in active_ids
        ]
        kept_labels = [int(lbl) for lbl in old_labels if int(lbl) in active_ids]

        if not kept_images:
            for path in (self.samples_path, self.lbph_path):
                if os.path.exists(path):
                    os.remove(path)
            return

        np.savez_compressed(
            self.samples_path,
            images=np.asarray(kept_images, dtype=np.uint8),
            labels=np.asarray(kept_labels, dtype=np.int32),
        )
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(kept_images, np.asarray(kept_labels, dtype=np.int32))
        recognizer.write(self.lbph_path)

    def preprocess_face(self, face_gray: np.ndarray) -> np.ndarray:
        """Resize and apply CLAHE lighting normalization to a cropped face."""
        resized = cv2.resize(face_gray, self.FACE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(resized)
        return cv2.GaussianBlur(normalized, (3, 3), 0)

    def detect_largest_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect the largest frontal face in an image and return normalized 128x128 gray crop."""
        if not self.is_available() or image_bgr is None or image_bgr.size == 0:
            return None

        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        if len(faces) == 0:
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(36, 36),
            )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        crop = gray[y : y + h, x : x + w]
        if crop.size == 0:
            return None
        return self.preprocess_face(crop)

    def extract_face_embedding(self, normalized_face: np.ndarray) -> np.ndarray:
        """
        Extract an L2-normalized spatial 4x4 LBP + HOG descriptor vector (640 dims).
        Resilient to illumination shifts while capturing both texture and geometry.
        """
        h, w = normalized_face.shape
        cell_h, cell_w = h // 4, w // 4
        parts: List[np.ndarray] = []

        for r in range(4):
            for c in range(4):
                cell = normalized_face[
                    r * cell_h : (r + 1) * cell_h,
                    c * cell_w : (c + 1) * cell_w,
                ]
                lbp = _compute_lbp_histogram(cell)
                lbp_norm = np.linalg.norm(lbp)
                if lbp_norm > 0:
                    lbp /= lbp_norm

                hog = _compute_hog_histogram(cell)
                hog_norm = np.linalg.norm(hog)
                if hog_norm > 0:
                    hog /= hog_norm

                parts.append(lbp * 0.65)
                parts.append(hog * 0.35)

        vec = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def _open_camera(self, camera_index: int = 0):
        """Open webcam using DirectShow on Windows for fast reliable capture."""
        if not CV2_AVAILABLE:
            return None
        if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        return cv2.VideoCapture(camera_index)

    def check_camera(self, camera_index: int = 0) -> bool:
        """Check whether a webcam can be opened and read cleanly."""
        if not CV2_AVAILABLE:
            return False
        cap = None
        try:
            cap = self._open_camera(camera_index)
            if not cap or not cap.isOpened():
                return False
            ok, frame = cap.read()
            return bool(ok and frame is not None and frame.size > 0)
        except Exception:
            return False
        finally:
            if cap is not None:
                cap.release()

    def _next_label_id(self, user_name: str) -> int:
        if user_name in self.profiles:
            return int(self.profiles[user_name]["label_id"])
        existing = [int(p.get("label_id", 0)) for p in self.profiles.values()]
        return (max(existing) + 1) if existing else 1

    def enroll_from_faces(self, user_name: str, normalized_faces: List[np.ndarray]) -> Dict[str, Any]:
        """Enroll a user directly from already-cropped/normalized grayscale faces."""
        clean_name = (user_name or "").strip()
        if not clean_name:
            return {"success": False, "error": "Nombre de usuario vacío."}
        if not normalized_faces:
            return {"success": False, "error": "No se detectó ningún rostro válido."}

        processed = [
            f if f.shape == self.FACE_SIZE else self.preprocess_face(f)
            for f in normalized_faces
        ]
        embeddings = [self.extract_face_embedding(f) for f in processed]
        centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm > 0:
            centroid = centroid / c_norm

        now = time.time()
        label_id = self._next_label_id(clean_name)
        created_at = self.profiles.get(clean_name, {}).get("created_at", now)

        self.profiles[clean_name] = {
            "label_id": label_id,
            "samples_count": len(processed),
            "centroid": [round(float(x), 6) for x in centroid.tolist()],
            "embeddings": [
                [round(float(x), 6) for x in emb.tolist()] for emb in embeddings[:12]
            ],
            "created_at": created_at,
            "updated_at": now,
        }
        self._save_profiles()
        self._save_and_train_lbph(label_id, processed)

        return {
            "success": True,
            "user": clean_name,
            "samples": len(processed),
            "lbph_trained": LBPH_AVAILABLE,
        }

    def enroll_from_images(self, user_name: str, images_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """Detect faces across a list of BGR images and enroll the user."""
        if not self.is_available():
            return {"success": False, "error": "OpenCV no está disponible."}
        faces = []
        for img in images_bgr:
            face = self.detect_largest_face(img)
            if face is not None:
                faces.append(face)
        if not faces:
            return {"success": False, "error": "No se detectó ningún rostro en las imágenes."}
        return self.enroll_from_faces(user_name, faces)

    def enroll_from_image_file(self, user_name: str, image_path: str) -> Dict[str, Any]:
        """Enroll a user from a local image file (useful when no webcam is attached)."""
        if not self.is_available():
            return {"success": False, "error": "OpenCV no está disponible."}
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Archivo no encontrado: {image_path}"}
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": f"No se pudo leer la imagen: {image_path}"}

        face = self.detect_largest_face(img)
        if face is None:
            return {"success": False, "error": "No se detectó un rostro frontal en la imagen."}

        # Add slight horizontal flip for better pose resilience from single photo
        flipped = cv2.flip(face, 1)
        return self.enroll_from_faces(user_name, [face, flipped])

    def enroll_from_camera(
        self,
        user_name: str,
        num_samples: int = 8,
        timeout_sec: float = 12.0,
        camera_index: int = 0,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Capture multiple face frames from the webcam and save a local face profile."""
        if not self.is_available():
            return {"success": False, "error": "OpenCV no está disponible."}

        cap = self._open_camera(camera_index)
        if not cap or not cap.isOpened():
            if cap is not None:
                cap.release()
            return {"success": False, "error": "No se pudo abrir la cámara web."}

        captured_faces: List[np.ndarray] = []
        deadline = time.time() + timeout_sec

        try:
            if log_callback:
                log_callback(f"[FACE] Cámara abierta. Capturando {num_samples} muestras para '{user_name}'...")

            while len(captured_faces) < num_samples and time.time() < deadline:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                face = self.detect_largest_face(frame)
                if face is not None:
                    captured_faces.append(face)
                    if log_callback:
                        log_callback(f"[FACE] Muestra {len(captured_faces)}/{num_samples} capturada.")
                    time.sleep(0.18)
                else:
                    time.sleep(0.08)
        finally:
            cap.release()

        if len(captured_faces) < 2:
            return {
                "success": False,
                "error": "No se detectó tu rostro con suficiente claridad. Verifica iluminación y cámara.",
            }

        result = self.enroll_from_faces(user_name, captured_faces)
        if log_callback and result.get("success"):
            log_callback(f"[FACE] Perfil facial guardado localmente para '{user_name}'.")
        return result

    def recognize_from_face(
        self,
        normalized_face: np.ndarray,
        threshold: float = 0.72,
    ) -> Dict[str, Any]:
        """Match a normalized grayscale face against all locally enrolled profiles."""
        if not self.profiles:
            return {
                "recognized": False,
                "user": None,
                "confidence": 0.0,
                "embedding_sim": 0.0,
                "lbph_conf": None,
                "reason": "no_profiles",
            }

        face = (
            normalized_face
            if normalized_face.shape == self.FACE_SIZE
            else self.preprocess_face(normalized_face)
        )
        query_emb = self.extract_face_embedding(face)

        # 1. Cosine similarity against centroids + individual samples
        best_user = None
        best_emb_sim = -1.0
        for name, profile in self.profiles.items():
            centroid = np.asarray(profile["centroid"], dtype=np.float32)
            c_sim = float(np.dot(query_emb, centroid))
            sample_sims = [
                float(np.dot(query_emb, np.asarray(s, dtype=np.float32)))
                for s in profile.get("embeddings", [])
            ]
            max_sample_sim = max(sample_sims) if sample_sims else c_sim
            user_sim = 0.60 * c_sim + 0.40 * max_sample_sim
            if user_sim > best_emb_sim:
                best_emb_sim = user_sim
                best_user = name

        # 2. Optional LBPH corroboration
        lbph_user = None
        lbph_score = None
        lbph_dist = None
        if LBPH_AVAILABLE and os.path.exists(self.lbph_path):
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(self.lbph_path)
                pred_label, pred_dist = recognizer.predict(face.astype(np.uint8))
                lbph_dist = float(pred_dist)
                # Map LBPH distance (lower is better; ~0-55 strong, ~80 weak, >110 no match) to [0, 1]
                lbph_score = max(0.0, min(1.0, 1.0 - (lbph_dist / 120.0)))
                for name, profile in self.profiles.items():
                    if int(profile.get("label_id", -1)) == int(pred_label):
                        lbph_user = name
                        break
            except Exception:
                pass

        # 3. Combine scores cleanly
        if lbph_score is not None and lbph_user == best_user:
            combined = 0.65 * best_emb_sim + 0.35 * lbph_score
        elif lbph_score is not None:
            combined = 0.75 * best_emb_sim + 0.25 * (lbph_score * 0.5)
        else:
            combined = best_emb_sim

        recognized = bool(best_user is not None and combined >= threshold)
        return {
            "recognized": recognized,
            "user": best_user if recognized else None,
            "closest_user": best_user,
            "confidence": round(float(max(0.0, min(1.0, combined))), 4),
            "embedding_sim": round(float(max(0.0, min(1.0, best_emb_sim))), 4),
            "lbph_conf": round(lbph_dist, 2) if lbph_dist is not None else None,
            "reason": "matched" if recognized else "below_threshold",
        }

    def recognize_from_image(
        self,
        image_bgr: np.ndarray,
        threshold: float = 0.72,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "cv2_unavailable"}
        face = self.detect_largest_face(image_bgr)
        if face is None:
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "no_face_detected"}
        return self.recognize_from_face(face, threshold=threshold)

    def recognize_from_image_file(
        self,
        image_path: str,
        threshold: float = 0.72,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "cv2_unavailable"}
        if not os.path.exists(image_path):
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "file_not_found"}
        img = cv2.imread(image_path)
        if img is None:
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "unreadable_image"}
        return self.recognize_from_image(img, threshold=threshold)

    def recognize_from_camera(
        self,
        timeout_sec: float = 5.0,
        camera_index: int = 0,
        threshold: float = 0.72,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Scan webcam briefly and match detected face against enrolled profiles."""
        if not self.is_available():
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "cv2_unavailable"}
        if not self.profiles:
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "no_profiles"}

        cap = self._open_camera(camera_index)
        if not cap or not cap.isOpened():
            if cap is not None:
                cap.release()
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "camera_unavailable"}

        best_result: Optional[Dict[str, Any]] = None
        checks = 0
        deadline = time.time() + timeout_sec

        try:
            if log_callback:
                log_callback("[FACE] Escaneando cámara para reconocimiento facial...")
            while checks < 5 and time.time() < deadline:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                face = self.detect_largest_face(frame)
                if face is not None:
                    checks += 1
                    res = self.recognize_from_face(face, threshold=threshold)
                    if best_result is None or res["confidence"] > best_result["confidence"]:
                        best_result = res
                    if res["recognized"] and res["confidence"] >= max(threshold + 0.06, 0.80):
                        break
                    time.sleep(0.12)
                else:
                    time.sleep(0.08)
        finally:
            cap.release()

        if best_result is None:
            return {"recognized": False, "user": None, "confidence": 0.0, "reason": "no_face_detected"}

        if log_callback:
            if best_result["recognized"]:
                log_callback(
                    f"[FACE] Usuario reconocido: {best_result['user']} "
                    f"(confianza={best_result['confidence']:.0%})."
                )
            else:
                log_callback(
                    f"[FACE] Rostro no reconocido (mejor similitud={best_result['confidence']:.0%})."
                )
        return best_result

    def list_profiles(self) -> List[Dict[str, Any]]:
        items = []
        for name, profile in self.profiles.items():
            items.append({
                "user": name,
                "samples": int(profile.get("samples_count", 0)),
                "updated_at": float(profile.get("updated_at", 0.0)),
            })
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items

    def delete_profile(self, user_name: Optional[str] = None) -> int:
        """Delete a single user profile, or reset all profiles if user_name is None."""
        if user_name is None:
            count = len(self.profiles)
            self.profiles = {}
            self._save_profiles()
            self._retrain_lbph_from_remaining()
            return count

        if user_name in self.profiles:
            del self.profiles[user_name]
            self._save_profiles()
            self._retrain_lbph_from_remaining()
            return 1
        return 0

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "lbph_available": LBPH_AVAILABLE,
            "profiles_count": len(self.profiles),
            "enrolled_users": list(self.profiles.keys()),
            "faces_dir": self.faces_dir,
        }
