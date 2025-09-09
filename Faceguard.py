import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_image_paths(root: Path):
    """Yield (person_name, image_path) for each image under employees dir.
    Expects structure: root/person_name/*.jpg|*.png|*.jpeg
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for person_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        person = person_dir.name
        for img_path in sorted(person_dir.rglob("*")):
            if img_path.suffix.lower() in exts:
                yield person, img_path


def save_unknown_snapshot(frame_bgr: np.ndarray, out_dir: Path):
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = out_dir / f"unknown_{ts}.jpg"
    cv2.imwrite(str(fname), frame_bgr)
    return fname


class FaceEncoder:
    def __init__(self, device: str = None, image_size: int = 160):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
       
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=20,
            post_process=True,
            device=self.device,
            keep_all=True
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    @torch.inference_mode()
    def encode_pil(self, img: Image.Image):
       
        face = self.mtcnn(img)
        if face is None:
            return None
        if face.ndim == 3:
            
            face = face.to(self.device)
        else:
            
            face = face[0].to(self.device)
        
        if face.shape[0] == 1:
            face = face.repeat(3, 1, 1)
        elif face.shape[0] == 4:
            face = face[:3, :, :]
        elif face.shape[0] != 3:
            face = face.expand(3, -1, -1)

        emb = self.resnet(face.unsqueeze(0)) 
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu()  # (512,)

    @torch.inference_mode()
    def detect_and_encode_bgr(self, frame_bgr: np.ndarray):
        """
        Detect faces in a BGR frame. Returns list of dicts:
        - box: [x1,y1,x2,y2]
        - prob: detection score
        - embedding: torch.Tensor (512,) or None if failed
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        boxes, probs = self.mtcnn.detect(pil)
        results = []
        if boxes is None:
            return results

        for box, prob in zip(boxes, probs):
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            box = np.array(box, dtype=np.float32)

            face_aligned = self.mtcnn.extract(
                pil, torch.tensor([box], dtype=torch.float32), save_path=None
            )
            if face_aligned is None or face_aligned.shape[0] == 0:
                emb = None
            else:
                face_tensor = face_aligned[0].to(self.device)

                if face_tensor.shape[0] == 1:
                    face_tensor = face_tensor.repeat(3, 1, 1)
                elif face_tensor.shape[0] == 4:
                    face_tensor = face_tensor[:3, :, :]
                elif face_tensor.shape[0] != 3:
                    face_tensor = face_tensor.expand(3, -1, -1)

                emb = self.resnet(face_tensor.unsqueeze(0))
                emb = torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0).cpu()

            results.append({
                'box': [x1, y1, x2, y2],
                'prob': float(prob if prob is not None else 0.0),
                'embedding': emb
            })
        return results


class EmployeeDB:
    def __init__(self):
        self.labels = []            
        self.embeddings = None     

    def __len__(self):
        return 0 if self.embeddings is None else self.embeddings.shape[0]

    def add(self, label: str, emb: torch.Tensor):
        if self.embeddings is None:
            self.embeddings = emb.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, emb.unsqueeze(0)], dim=0)
        self.labels.append(label)

    def consolidate(self):
        """Average multiple samples per person to a single centroid embedding."""
        if self.embeddings is None:
            return
        by_label = {}
        for idx, label in enumerate(self.labels):
            by_label.setdefault(label, []).append(idx)
        new_embs = []
        new_labels = []
        for label, idxs in by_label.items():
            embs = self.embeddings[idxs]
            centroid = torch.nn.functional.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=1)
            new_embs.append(centroid)
            new_labels.append(label)
        self.embeddings = torch.vstack(new_embs)
        self.labels = new_labels

    def save(self, path: Path):
        ensure_dir(path.parent)
        torch.save({'labels': self.labels, 'embeddings': self.embeddings}, str(path))

    @staticmethod
    def load(path: Path):
        obj = EmployeeDB()
        data = torch.load(str(path), map_location='cpu')
        obj.labels = data['labels']
        obj.embeddings = data['embeddings']
        return obj

    @torch.inference_mode()
    def match(self, query_emb: torch.Tensor):
        """Return (best_label, best_score) using cosine similarity."""
        if self.embeddings is None or len(self.labels) == 0:
            return None, 0.0
        sims = torch.nn.functional.cosine_similarity(self.embeddings, query_emb.unsqueeze(0), dim=1)
        best_idx = int(torch.argmax(sims).item())
        best_score = float(sims[best_idx].item())
        return self.labels[best_idx], best_score


@torch.inference_mode()
def build_db(employees_dir: Path, db_path: Path, device: str = None):
    encoder = FaceEncoder(device=device)
    db = EmployeeDB()

    pairs = list(load_image_paths(employees_dir))
    if not pairs:
        raise SystemExit(f"No images found under {employees_dir}. Add employee photos first.")

    for person, img_path in tqdm(pairs, desc='Encoding employees'):
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}")
            continue
        emb = encoder.encode_pil(img)
        if emb is None:
            print(f"[WARN] No face detected in {img_path}, skipping.")
            continue
        db.add(person, emb)

    db.consolidate()
    db.save(db_path)
    print(f"Saved {len(db.labels)} employees to {db_path}")


def draw_label(frame, text, box, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def run_detection(source,
                  db_path: Path,
                  threshold: float = 0.85,
                  device: str = None,
                  save_unknown: bool = True,
                  out_dir: Path = Path('output/snapshots'),
                  min_prob: float = 0.90,
                  min_face: int = 60,
                  warmup: int = 5,
                  unknown_patience: int = 5,
                  stop_on_unknown: bool = True):
    """
    unknown_patience: number of consecutive high-quality 'unknown' frames required
                      before saving/stopping.
    min_prob: minimum MTCNN detection probability to consider a face valid
    min_face: minimum face width/height in pixels to consider valid
    warmup: number of initial frames to skip decisions (camera exposure settle)
    """
    if not db_path.exists():
        raise SystemExit(f"DB not found at {db_path}. Build it first with --build-db.")
    db = EmployeeDB.load(db_path)
    encoder = FaceEncoder(device=device)

    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open source: {source}")

    print("Press 'q' to quit.")
    frame_idx = 0
    unknown_streak = 0
    trigger_done = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame. Exiting...")
            break

        frame_idx += 1
        results = encoder.detect_and_encode_bgr(frame)

      
        analyzing_only = frame_idx <= warmup

        any_high_quality_unknown = False

        for r in results:
            box = r['box']
            prob = r['prob']
            emb = r['embedding']

            w = box[2] - box[0]
            h = box[3] - box[1]
            big_enough = (w >= min_face and h >= min_face)

            if emb is None:
               
                draw_label(frame, f'NoFace/LowCrop p={prob:.2f}', box, color=(0, 165, 255))
                continue

            if prob < min_prob or not big_enough:
                
                reason = []
                if prob < min_prob: reason.append(f"p<{min_prob:.2f}")
                if not big_enough: reason.append(f"small<{min_face}px")
                draw_label(frame, "LowConf " + ",".join(reason), box, color=(0, 165, 255))
                continue

         
            label, score = db.match(emb)
            if score >= threshold:
                draw_label(frame, f"{label} ({score:.2f})", box, color=(0, 255, 0))
            else:
                draw_label(frame, f"Unknown ({score:.2f})", box, color=(0, 0, 255))
                any_high_quality_unknown = True

       
        if not analyzing_only:
            if any_high_quality_unknown:
                unknown_streak += 1
            else:
                unknown_streak = 0

       
        status = f"Analyzing: {min(unknown_streak, unknown_patience)}/{unknown_patience}"
        cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if unknown_streak >= unknown_patience and not trigger_done:
            trigger_done = True
            if save_unknown:
                save_path = save_unknown_snapshot(frame, out_dir)
                print(f"[!] Unknown confirmed. Saved snapshot: {save_path}")
            if stop_on_unknown:
                print("[!] Stopping due to confirmed unknown.")
                cv2.imshow('FaceGuard - Unknown Face Detector', frame)
                cv2.waitKey(300)  
                break

        cv2.imshow('FaceGuard - Unknown Face Detector', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description='FaceGuard: Unknown face detector using OpenCV + PyTorch (smoothed)')
    p.add_argument('--build-db', action='store_true', help='Build embeddings database from employees folder')
    p.add_argument('--detect', action='store_true', help='Run real-time detection from camera/stream/file')
    p.add_argument('--employees-dir', type=Path, default=Path('data/employees'), help='Employees images root')
    p.add_argument('--db-path', type=Path, default=Path('data/db/embeddings.pt'), help='Path to embeddings DB')
    p.add_argument('--source', type=str, default='0', help='Camera index (e.g., 0) or video/rtsp path')
    p.add_argument('--threshold', type=float, default=0.85, help='Cosine similarity threshold for match')
    p.add_argument('--device', type=str, default=None, help='Force device: cpu or cuda')
    p.add_argument('--min-prob', type=float, default=0.90, help='Min detection probability to consider a face')
    p.add_argument('--min-face', type=int, default=60, help='Min face size (pixels) to consider a face')
    p.add_argument('--warmup', type=int, default=5, help='Frames to wait before making decisions')
    p.add_argument('--unknown-patience', type=int, default=5, help='Consecutive unknown frames required to trigger')
    p.add_argument('--no-save-unknown', action='store_true', help="Don't save unknown snapshots")
    p.add_argument('--no-stop-on-unknown', action='store_true', help="Keep running after unknown is confirmed")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(Path('data/employees'))
    ensure_dir(Path('data/db'))
    ensure_dir(Path('output/snapshots'))
    ensure_dir(Path('output/logs'))

    if args.build_db:
        build_db(args.employees_dir, args.db_path, device=args.device)

    if args.detect:
        src = args.source
        if src.isdigit():
            src = int(src)
        run_detection(
            src,
            args.db_path,
            threshold=args.threshold,
            device=args.device,
            save_unknown=not args.no_save_unknown,
            out_dir=Path('output/snapshots'),
            min_prob=args.min_prob,
            min_face=args.min_face,
            warmup=args.warmup,
            unknown_patience=args.unknown_patience,
            stop_on_unknown=not args.no_stop_on_unknown
        )

    if not args.build_db and not args.detect:
        print("Nothing to do. Use --build-db and/or --detect. Run with -h for help.")


if __name__ == '__main__':
    main()
