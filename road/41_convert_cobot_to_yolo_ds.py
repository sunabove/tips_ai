# road/dataset/cobot_01/ 폴더 아래에 있는 모든 동영상 파일을 YOLO segmentation dataset 형식으로 변환합니다.
# 동영상의 파일명은 {class_name}_{index}.mp4 형식으로 되어 있습니다.
# 변환 폴더는 road/dataset/cobot_01_yolo_seg/ 입니다.
# 변환 과정에서 동영상의 모든 프레임을 추출하여 이미지로 저장하고,
# 해당 프레임에서 검출된 객체의 마스크를 YOLO segmentation 형식으로 변환합니다.
# 변환 과정에서 사용되는 YOLO segmentation dataset 형식은 다음과 같습니다.
# road/dataset/cobot_01/colormap_road.txt 파일에 정의된 클래스명에 매핑된 색상을 이용하여,
# 마스크된 이미지를 변환 폴더에 생성합니다.
# 마스크 이미지의 파일명은 {class_name}_{index}_{frame_index}.png 형식으로 저장됩니다.
# - images/train/ : 학습용 이미지 폴더
# - images/val/ : 검증용 이미지 폴더
# - images/test/ : 테스트용 이미지 폴더
# - labels/train/ : 학습용 라벨 폴더
# - labels/val/ : 검증용 라벨 폴더
# - labels/test/ : 테스트용 라벨 폴더

# Manual run:
# 1. cd ai\road
# 2. python 41_convert_cobot_to_yolo_ds.py

from __future__ import annotations

import argparse
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ClassSpec:
    yolo_id: int
    name: str
    rgb: Tuple[int, int, int]


def default_cobot_root() -> Path:
    primary = Path("road/dataset/cobot_01")
    fallback = Path("dataset/cobot_01")
    return primary if primary.exists() else fallback


def default_colormap_path(cobot_root: Path) -> Path:
    """colormap_road.txt를 cobot_01 폴더 또는 상위 경로에서 탐색."""
    candidates = [
        cobot_root / "colormap_road.txt",
        cobot_root.parent.parent / "colormap_road.txt",  # 300_python/
    ]
    for p in candidates:
        if p.exists():
            return p
    return cobot_root / "colormap_road.txt"


def parse_colormap(colormap_path: Path) -> List[ClassSpec]:
    """colormap_road.txt를 파싱하여 ClassSpec 목록을 반환합니다.

    파일 형식 (공백 구분):
        {name} {R} {G} {B}
    예시:
        void 0 0 0
        road 255 0 0
    'void' 항목은 배경으로 간주하여 제외합니다.
    """
    specs: list[ClassSpec] = []
    yolo_id = 0
    with colormap_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0].lower()
            if name == "void":
                continue
            try:
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            specs.append(ClassSpec(yolo_id=yolo_id, name=name, rgb=(r, g, b)))
            yolo_id += 1
    if not specs:
        raise ValueError(f"colormap 파일에서 유효한 클래스를 찾을 수 없습니다: {colormap_path}")
    return specs


def collect_videos(cobot_root: Path) -> List[Path]:
    """cobot_root 폴더에서 모든 .mp4 파일을 수집하여 정렬된 목록으로 반환합니다."""
    videos = sorted(cobot_root.rglob("*.mp4"))
    if not videos:
        raise RuntimeError(f"동영상 파일(.mp4)을 찾을 수 없습니다: {cobot_root}")
    return videos


def extract_video_stem(video_path: Path) -> str:
    """동영상 파일명에서 stem을 반환합니다 (예: road_01 <- road_01.mp4)."""
    return video_path.stem


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


def build_label_lines(
    frame_rgb: np.ndarray,
    classes: List[ClassSpec],
    min_area: float,
    epsilon_ratio: float,
) -> List[str]:
    """마스크 프레임(RGB) 에서 YOLO segmentation 라벨 라인 목록을 생성합니다."""
    h, w = frame_rgb.shape[:2]
    lines: list[str] = []

    for cls in classes:
        color = np.array(cls.rgb, dtype=np.uint8)
        binary = cv2.inRange(frame_rgb, color, color)
        if not np.any(binary):
            continue

        polygons = polygon_from_binary_mask(binary, min_area, epsilon_ratio, w, h)
        for poly in polygons:
            coord_text = " ".join(f"{x:.6f}" for x in poly)
            lines.append(f"{cls.yolo_id} {coord_text}")

    return lines


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
    colormap_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_area: float,
    epsilon_ratio: float,
    frame_step: int,
) -> None:
    classes = parse_colormap(colormap_path)
    videos = collect_videos(cobot_root)

    print(f"발견된 동영상: {len(videos)} 개")
    print(f"클래스: {[f'{c.yolo_id}:{c.name}' for c in classes]}")

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
                file_id = f"{stem}_{frame_idx:06d}"

                img_dst = output_root / "images" / split / f"{file_id}.png"
                lbl_dst = output_root / "labels" / split / f"{file_id}.txt"

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

                # RGB 변환 후 YOLO 라벨 생성
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                label_lines = build_label_lines(frame_rgb, classes, min_area, epsilon_ratio)
                lbl_dst.write_text("\n".join(label_lines), encoding="utf-8")

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
    write_dataset_yaml(output_root, classes)

    print("\n변환 완료.")
    print(f"입력 경로  : {cobot_root}")
    print(f"출력 경로  : {output_root}")
    print(f"colormap   : {colormap_path}")
    print(f"클래스 수  : {len(classes)}")
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
        default=Path("road/dataset/cobot_01_yolo_seg"),
        help="출력 폴더 경로 (기본값: road/dataset/cobot_01_yolo_seg)",
    )
    parser.add_argument(
        "--colormap",
        type=Path,
        default=None,
        help="colormap_road.txt 경로 (미지정 시 자동 탐색)",
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
        "--epsilon-ratio",
        type=float,
        default=0.01,
        help="윤곽선 근사 epsilon 비율 (기본값: 0.01)",
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
    colormap = args.colormap if args.colormap else default_colormap_path(args.cobot_root)
    convert(
        cobot_root=args.cobot_root,
        output_root=args.output_root,
        colormap_path=colormap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
        frame_step=args.frame_step,
    )
