# road/dataset/cobot_01/ 폴더 아래에 있는 모든 동영상 파일을 YOLO segmentation dataset 형식으로 변환합니다.
# 동영상의 파일명은 {class_name}_{index}.mp4 형식으로 되어 있습니다.
# 변환 폴더는 road/dataset/cobot_01_yolo_seg/ 입니다.
# 변환 과정에서 동영상의 모든 프레임을 추출하여 이미지로 저장하고,
# 해당 프레임에서 검출된 객체의 마스크를 YOLO segmentation 형식으로 변환합니다.

# segmentation 폴리곤은 model/01_yolo11m-road-sg.pt에서 학습된 모델을 이용하여 도로 영역을 추출합니다.
# 도로 영역 추출시에는 하나의 클래스 road가 추출됩니다.
# 추출한 도로 영역을 입력 파일명에 있는 {class_name}으로 고정하여 새롭게 라벨링하여 YOLO segmentation 형식으로 변환합니다.
# 하나의 영역이 추출되면 모두 YOLO segmentation 형식으로 변환하여 라벨 파일에 저장합니다.
# 2개 이상의 영역이 추출되면, 신뢰도 평균이상의 영역만 YOLO segmentation 형식으로 변환하여 라벨 파일에 저장합니다.

# 변환 과정에서 사용되는 YOLO segmentation dataset 형식은 다음과 같습니다.
# - images/train/ : 학습용 이미지 폴더
# - images/val/ : 검증용 이미지 폴더
# - images/test/ : 테스트용 이미지 폴더
# - labels/train/ : 학습용 라벨 폴더
# - labels/val/ : 검증용 라벨 폴더
# - labels/test/ : 테스트용 라벨 폴더

# road/dataset/cobot_01/colormap_road.txt 파일에 정의된 클래스명에 매핑된 색상을 이용하여,
# 마스크된 이미지를 해당 변환 폴더에 생성합니다.
# 마스크 이미지의 파일명은 {class_name}_{index}_{frame_index}.png 형식으로 저장됩니다.

# Manual run:
# 1. cd ai\road
# 2. python 41_convert_cobot_to_yolo_ds.py

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class ClassSpec:
    yolo_id: int
    name: str


@dataclass(frozen=True)
class TargetClassSpec:
    target_id: int
    name: str


# 스크립트 파일이 위치한 디렉터리 (300_python/ai/road/)
_SCRIPT_DIR = Path(__file__).resolve().parent


def default_cobot_root() -> Path:
    """스크립트 기준 road/dataset/cobot_01 경로를 반환합니다."""
    return _SCRIPT_DIR / "dataset" / "cobot_01"


def default_output_root() -> Path:
    """스크립트 기준 road/dataset/cobot_01_yolo_seg 경로를 반환합니다."""
    return _SCRIPT_DIR / "dataset" / "cobot_01_yolo_seg"


def default_model_path() -> Path:
    """스크립트 기준 모델 경로를 반환합니다."""
    return _SCRIPT_DIR / "model" / "01_yolo11m-road-sg.pt"


def default_colormap_path() -> Path:
    """스크립트 기준 colormap_road.txt 경로를 반환합니다."""
    return _SCRIPT_DIR.parent.parent / "colormap_road.txt"


def class_specs_from_model(model: YOLO) -> List[ClassSpec]:
    names = model.names
    if isinstance(names, dict):
        ordered = sorted(names.items(), key=lambda x: int(x[0]))
        return [ClassSpec(yolo_id=int(k), name=str(v)) for k, v in ordered]
    if isinstance(names, list):
        return [ClassSpec(yolo_id=i, name=str(v)) for i, v in enumerate(names)]
    raise ValueError("모델 클래스 정보를 읽을 수 없습니다.")


def collect_videos(cobot_root: Path) -> List[Path]:
    """cobot_root 폴더에서 모든 .mp4 파일을 수집하여 정렬된 목록으로 반환합니다."""
    videos = sorted(cobot_root.rglob("*.mp4"))
    if not videos:
        raise RuntimeError(f"동영상 파일(.mp4)을 찾을 수 없습니다: {cobot_root}")
    return videos


def extract_video_stem(video_path: Path) -> str:
    """동영상 파일명에서 stem을 반환합니다 (예: road_01 <- road_01.mp4)."""
    return video_path.stem


def extract_class_name_from_stem(stem: str) -> str:
    """{class_name}_{index} 형식의 stem 에서 class_name 을 추출합니다."""
    if "_" not in stem:
        raise ValueError(f"파일명이 {{class_name}}_{{index}} 형식이 아닙니다: {stem}")
    class_name, _ = stem.rsplit("_", 1)
    if not class_name:
        raise ValueError(f"클래스명이 비어 있습니다: {stem}")
    return class_name


def class_name_to_id_map(classes: List[ClassSpec]) -> Dict[str, int]:
    return {c.name: c.yolo_id for c in classes}


def target_class_specs_from_colormap(
    colormap: Dict[str, Tuple[int, int, int]]
) -> List[TargetClassSpec]:
    """colormap 순서를 유지하여 target class id를 생성합니다."""
    return [TargetClassSpec(target_id=i, name=name) for i, name in enumerate(colormap.keys())]


def target_class_name_to_id_map(classes: List[TargetClassSpec]) -> Dict[str, int]:
    return {c.name: c.target_id for c in classes}


def find_model_class_id_by_name(model: YOLO, class_name: str) -> int | None:
    names = model.names
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v) == class_name:
                return int(k)
    elif isinstance(names, list):
        for i, v in enumerate(names):
            if str(v) == class_name:
                return i
    return None


def find_colormap_path(cobot_root: Path, colormap_path: Path | None) -> Path:
    """cobot_root 우선, 그 다음 명시/기본 경로에서 colormap 파일을 찾습니다."""
    candidates: list[Path] = [cobot_root / "colormap_road.txt"]
    if colormap_path is not None:
        candidates.append(colormap_path)
    else:
        candidates.append(default_colormap_path())

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "colormap_road.txt 파일을 찾을 수 없습니다. "
        f"확인 경로: {[str(p) for p in candidates]}"
    )


def load_colormap(colormap_path: Path) -> Dict[str, Tuple[int, int, int]]:
    """class_name R G B 형식의 colormap 파일을 읽어 RGB 딕셔너리로 반환합니다."""
    cmap: dict[str, tuple[int, int, int]] = {}
    for raw in colormap_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        class_name = parts[0]
        try:
            r, g, b = (int(parts[1]), int(parts[2]), int(parts[3]))
        except ValueError:
            continue
        cmap[class_name] = (r, g, b)
    if not cmap:
        raise RuntimeError(f"유효한 colormap 항목이 없습니다: {colormap_path}")
    return cmap


def split_items(
    items: List[Tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, int]]]:
    """(video_path, frame_index) 목록을 train/val/test 로 분할합니다."""
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-9:
        raise ValueError("train_ratio + val_ratio 의 합이 1.0 을 초과합니다.")

    rnd = random.Random(seed)
    shuffled = items[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def ensure_dirs(output_root: Path, splits: Iterable[str]) -> None:
    for split in splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "masks" / split).mkdir(parents=True, exist_ok=True)


def polygon_from_binary_mask(
    binary_mask: np.ndarray,
    min_area: float,
    epsilon_ratio: float,
    width: int,
    height: int,
) -> List[List[float]]:
    """이진 마스크에서 정규화된 YOLO 폴리곤 좌표 목록을 반환합니다."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        epsilon = epsilon_ratio * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0] / width, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / height, 0.0, 1.0)

        flat = pts.flatten().tolist()
        if len(flat) >= 6:
            polygons.append(flat)

    return polygons


def build_label_lines_from_model(
    model: YOLO,
    frame_bgr: np.ndarray,
    min_area: float,
    conf_threshold: float,
    road_model_class_id: int | None,
) -> List[np.ndarray]:
    """YOLO 세그멘테이션 모델 추론 결과에서 유지할 폴리곤 목록을 반환합니다.

    규칙:
    - 1개 영역이면 모두 저장
    - 2개 이상이면 평균 신뢰도 이상(conf >= mean(conf))만 저장
    """
    h, w = frame_bgr.shape[:2]
    results = model.predict(source=frame_bgr, conf=conf_threshold, verbose=False)
    if not results:
        return []

    result = results[0]
    if result.masks is None or result.masks.xy is None:
        return []

    polygons_xy = result.masks.xy
    n = len(polygons_xy)
    if n == 0:
        return []

    if result.boxes is not None and result.boxes.conf is not None and result.boxes.cls is not None:
        confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    else:
        confs = np.ones(n, dtype=np.float32)
        cls_ids = np.zeros(n, dtype=np.int32)

    m = min(n, len(confs), len(cls_ids))
    polygons_xy = polygons_xy[:m]
    confs = confs[:m]
    cls_ids = cls_ids[:m]

    if road_model_class_id is not None:
        road_keep = cls_ids == road_model_class_id
        polygons_xy = [poly for poly, keep in zip(polygons_xy, road_keep) if bool(keep)]
        confs = confs[road_keep]
        m = len(polygons_xy)
        if m == 0:
            return []

    if m >= 2:
        mean_conf = float(np.mean(confs))
        keep_mask = confs >= mean_conf
    else:
        keep_mask = np.ones(m, dtype=bool)

    kept_polygons: list[np.ndarray] = []

    for i, poly_xy in enumerate(polygons_xy):
        if not bool(keep_mask[i]):
            continue

        if poly_xy is None or len(poly_xy) < 3:
            continue

        area = cv2.contourArea(poly_xy.astype(np.float32))
        if area < min_area:
            continue

        pts = poly_xy.astype(np.float32).copy()
        if len(pts) >= 3:
            kept_polygons.append(pts)

    return kept_polygons


def build_label_lines_with_fixed_class(
    polygons_xy: List[np.ndarray],
    yolo_class_id: int,
    width: int,
    height: int,
) -> List[str]:
    lines: list[str] = []
    for poly_xy in polygons_xy:
        pts = poly_xy.astype(np.float32).copy()
        pts[:, 0] = np.clip(pts[:, 0] / width, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / height, 0.0, 1.0)

        flat = pts.flatten().tolist()
        if len(flat) < 6:
            continue

        coord_text = " ".join(f"{x:.6f}" for x in flat)
        lines.append(f"{yolo_class_id} {coord_text}")

    return lines


def render_colored_mask_image(
    image_shape: Tuple[int, int],
    polygons_xy: List[np.ndarray],
    rgb_color: Tuple[int, int, int],
) -> np.ndarray:
    """선택된 폴리곤을 단일 클래스 색상으로 채운 마스크(BGR)를 생성합니다."""
    h, w = image_shape
    mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    if not polygons_xy:
        return mask_bgr

    bgr = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
    for poly_xy in polygons_xy:
        pts_i32 = np.round(poly_xy).astype(np.int32)
        if len(pts_i32) >= 3:
            cv2.fillPoly(mask_bgr, [pts_i32], bgr)

    return mask_bgr


def write_dataset_yaml(output_root: Path, classes: List[ClassSpec]) -> None:
    names = [cls.name for cls in classes]
    yaml_text = (
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    (output_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def clear_output_root(output_root: Path) -> None:
    if output_root.exists():
        print(f"기존 출력 폴더를 삭제합니다: {output_root}")
        shutil.rmtree(output_root)


def convert(
    cobot_root: Path,
    output_root: Path,
    model_path: Path,
    colormap_path: Path | None,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_area: float,
    conf_threshold: float,
    frame_step: int,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    model = YOLO(str(model_path))
    model_classes = class_specs_from_model(model)
    road_model_class_id = find_model_class_id_by_name(model, "road")
    resolved_colormap_path = find_colormap_path(cobot_root, colormap_path)
    class_to_color = load_colormap(resolved_colormap_path)
    target_classes = target_class_specs_from_colormap(class_to_color)
    target_class_to_id = target_class_name_to_id_map(target_classes)
    videos = collect_videos(cobot_root)

    print(f"발견된 동영상: {len(videos)} 개")
    print(f"모델 클래스: {[f'{c.yolo_id}:{c.name}' for c in model_classes]}")
    print(f"타깃 클래스: {[f'{c.target_id}:{c.name}' for c in target_classes]}")
    print(f"colormap 경로: {resolved_colormap_path}")
    if road_model_class_id is None:
        print("[경고] 모델에서 'road' 클래스를 찾지 못했습니다. 모든 예측 폴리곤을 사용합니다.")

    # 전체 (video_path, frame_index) 쌍 수집
    all_items: list[tuple[Path, int]] = []
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[경고] 동영상을 열 수 없습니다: {video_path}")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for fi in range(0, total_frames, frame_step):
            all_items.append((video_path, fi))

    if not all_items:
        raise RuntimeError("추출 가능한 프레임이 없습니다.")

    print(f"총 프레임 수 (frame_step={frame_step}): {len(all_items)}")

    split_map = split_items(all_items, train_ratio, val_ratio, seed)
    clear_output_root(output_root)
    ensure_dirs(output_root, split_map.keys())

    total = len(all_items)
    processed = 0

    # 동영상 파일별로 cap 을 캐싱하여 반복 open/close 최소화
    cap_cache: dict[Path, cv2.VideoCapture] = {}

    try:
        for split, items in split_map.items():
            split_total = len(items)
            print(f"\n[{split}] {split_total} 프레임 처리 중...")
            split_done = 0

            for video_path, frame_idx in items:
                stem = extract_video_stem(video_path)
                input_class_name = extract_class_name_from_stem(stem)
                if input_class_name not in target_class_to_id:
                    raise KeyError(
                        f"입력 클래스명 '{input_class_name}' 이(가) 타깃 클래스(colormap)에 없습니다: "
                        f"{sorted(target_class_to_id.keys())}"
                    )
                input_class_id = target_class_to_id[input_class_name]

                if input_class_name not in class_to_color:
                    raise KeyError(
                        f"입력 클래스명 '{input_class_name}' 이(가) colormap에 없습니다: "
                        f"{resolved_colormap_path}"
                    )

                file_id = f"{stem}_{frame_idx:06d}"

                img_dst = output_root / "images" / split / f"{file_id}.png"
                lbl_dst = output_root / "labels" / split / f"{file_id}.txt"
                mask_dst = output_root / "masks" / split / f"{file_id}.png"

                # VideoCapture 캐싱
                if video_path not in cap_cache:
                    cap_cache[video_path] = cv2.VideoCapture(str(video_path))
                cap = cap_cache[video_path]

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    print(f"\n[경고] 프레임 읽기 실패: {video_path} frame={frame_idx}")
                    continue

                # 마스크 프레임을 이미지로 저장
                cv2.imwrite(str(img_dst), frame_bgr)

                # YOLO 세그멘테이션 모델 추론 결과에서 폴리곤 추출
                polygons_xy = build_label_lines_from_model(
                    model=model,
                    frame_bgr=frame_bgr,
                    min_area=min_area,
                    conf_threshold=conf_threshold,
                    road_model_class_id=road_model_class_id,
                )

                # 추출된 모든 영역의 클래스는 입력 파일 클래스명으로 고정
                h, w = frame_bgr.shape[:2]
                label_lines = build_label_lines_with_fixed_class(
                    polygons_xy=polygons_xy,
                    yolo_class_id=input_class_id,
                    width=w,
                    height=h,
                )
                lbl_dst.write_text("\n".join(label_lines), encoding="utf-8")

                # colormap 기반 컬러 마스크 이미지 생성
                mask_bgr = render_colored_mask_image(
                    image_shape=(h, w),
                    polygons_xy=polygons_xy,
                    rgb_color=class_to_color[input_class_name],
                )
                cv2.imwrite(str(mask_dst), mask_bgr)

                processed += 1
                split_done += 1
                overall_pct = processed / total * 100.0
                split_pct = split_done / split_total * 100.0 if split_total else 100.0
                print(
                    f"  [{processed}/{total}] {overall_pct:6.2f}% | "
                    f"{split}: {split_done}/{split_total} ({split_pct:6.2f}%)",
                    end="\r",
                    flush=True,
                )
    finally:
        for cap in cap_cache.values():
            cap.release()

    print()
    write_dataset_yaml(
        output_root,
        [ClassSpec(yolo_id=c.target_id, name=c.name) for c in target_classes],
    )

    print("\n변환 완료.")
    print(f"입력 경로  : {cobot_root}")
    print(f"출력 경로  : {output_root}")
    print(f"모델 경로  : {model_path}")
    print(f"컬러맵 경로: {resolved_colormap_path}")
    print(f"클래스 수  : {len(target_classes)}")
    print(f"총 프레임  : {processed}")
    print(
        f"분할 결과  -> "
        f"train: {len(split_map['train'])}, "
        f"val: {len(split_map['val'])}, "
        f"test: {len(split_map['test'])}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cobot_01 마스크 동영상을 YOLO segmentation dataset 형식으로 변환합니다."
    )
    cobot_default = default_cobot_root()
    parser.add_argument(
        "--cobot-root",
        type=Path,
        default=cobot_default,
        help=f"cobot_01 데이터 폴더 경로 (기본값: {cobot_default})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help=f"출력 폴더 경로 (기본값: {default_output_root()})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help=f"세그멘테이션 모델 경로 (기본값: {default_model_path()})",
    )
    parser.add_argument(
        "--colormap",
        type=Path,
        default=None,
        help=(
            "colormap_road.txt 경로 (미지정 시 cobot_root/colormap_road.txt "
            f"또는 {default_colormap_path()} 자동 탐색)"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="학습 데이터 비율 (기본값: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="검증 데이터 비율 (기본값: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본값: 42)")
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="폴리곤 최소 픽셀 면적 (기본값: 20.0)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.001,
        help="모델 추론 최소 신뢰도 (기본값: 0.001)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="프레임 샘플링 간격 (기본값: 1, 모든 프레임 사용)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        cobot_root=args.cobot_root,
        output_root=args.output_root,
        model_path=args.model,
        colormap_path=args.colormap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_area=args.min_area,
        conf_threshold=args.conf_threshold,
        frame_step=args.frame_step,
    )
