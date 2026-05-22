"""
Open and visualize a SensoryNBLab session .mat file.

Usage from this folder:
    python visualize_session.py
    python visualize_session.py session.mat
    python visualize_session.py -_session.mat --save-only

The script prints the session metadata, saves PNG previews, and opens one
OpenCV window per report unless --save-only is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io as sio


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def unwrap(value: Any) -> Any:
    """Convert MATLAB/scipy scalar arrays into plain Python values when possible."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.shape == ():
            return unwrap(value.item())
        if value.size == 1:
            return unwrap(value.reshape(-1)[0])
    return value


def as_text(value: Any) -> str:
    value = unwrap(value)
    if value is None:
        return "-"
    if isinstance(value, np.ndarray):
        return ", ".join(str(unwrap(item)) for item in value.reshape(-1))
    return str(value)


def get_field(struct: Any, name: str, default: Any = None) -> Any:
    return getattr(struct, name, default)


def find_default_mat_file() -> Path:
    session_mat = SCRIPT_DIR / "session.mat"
    if session_mat.exists():
        return session_mat

    candidates = sorted(SCRIPT_DIR.glob("*_session.mat"))
    if candidates:
        return candidates[0]

    candidates = sorted(SCRIPT_DIR.glob("*.mat"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No .mat file found in {SCRIPT_DIR}")


def load_session(mat_path: Path) -> Any:
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        raise KeyError(f"{mat_path} does not contain a top-level 'data' struct")
    return mat["data"]


def iter_reports(data: Any) -> list[tuple[str, Any]]:
    report = get_field(data, "report")
    if report is None:
        return []

    names = list(getattr(report, "_fieldnames", []))
    names.sort(key=lambda name: int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else name)
    return [(name, getattr(report, name)) for name in names]


def find_hand_image(hand: str) -> Path | None:
    hand_dir = PROJECT_ROOT / "PIC" / hand.capitalize()
    for filename in ("Hand.png", "Hand.jpg", "Hand.jpeg", "Hand.bmp"):
        path = hand_dir / filename
        if path.exists():
            return path
    return None


def normalize_map(selection_map: np.ndarray) -> np.ndarray:
    selection_map = np.asarray(selection_map)
    if selection_map.ndim > 2:
        selection_map = np.squeeze(selection_map)
    if selection_map.ndim != 2:
        raise ValueError(f"Expected a 2D Map matrix, got shape {selection_map.shape}")
    return selection_map


def make_map_preview(selection_map: np.ndarray) -> np.ndarray:
    selected = selection_map > 0
    preview = np.full((*selection_map.shape, 3), 255, dtype=np.uint8)
    preview[selected] = (0, 0, 255)
    return preview


def make_overlay(hand_image: np.ndarray, selection_map: np.ndarray) -> tuple[np.ndarray, str | None]:
    map_height, map_width = selection_map.shape
    image_height, image_width = hand_image.shape[:2]
    warning = None

    display_map = selection_map
    if (map_width, map_height) != (image_width, image_height):
        warning = (
            f"Map is {map_width}x{map_height}, hand image is "
            f"{image_width}x{image_height}; resized map for display only."
        )
        display_map = cv2.resize(
            selection_map,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = hand_image.copy()
    selected = display_map > 0
    red = np.zeros_like(overlay)
    red[:, :] = (0, 0, 255)
    overlay[selected] = cv2.addWeighted(overlay[selected], 0.45, red[selected], 0.55, 0)
    return overlay, warning


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    title_height = 58
    canvas = np.full((image.shape[0] + title_height, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_height:, :] = image
    cv2.putText(
        canvas,
        title,
        (16, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def combine_preview(selection_map: np.ndarray, overlay: np.ndarray, title: str) -> np.ndarray:
    map_preview = make_map_preview(selection_map)
    if map_preview.shape[0] != overlay.shape[0]:
        map_preview = resize_to_height(map_preview, overlay.shape[0])

    divider = np.full((overlay.shape[0], 10, 3), 220, dtype=np.uint8)
    combined = np.hstack([map_preview, divider, overlay])
    return add_title(combined, title)


def print_session_summary(mat_path: Path, data: Any, reports: list[tuple[str, Any]]) -> None:
    print(f"\nLoaded: {mat_path}")
    print("\nSession metadata")
    print("----------------")
    for field in (
        "Date",
        "PatientID",
        "Hand",
        "ModulationType",
        "Nerve",
        "InterphaseDistance_us",
        "Current",
        "Frequency",
        "PulseWidth",
        "MotorThreshold",
        "SensoryThreshold",
    ):
        print(f"{field}: {as_text(get_field(data, field))}")

    print(f"\nReports: {len(reports)}")
    for name, report in reports:
        selection_map = normalize_map(get_field(report, "Map"))
        print(
            f"  {name}: Map shape={selection_map.shape}, "
            f"selected_pixels={int(np.count_nonzero(selection_map))}, "
            f"modulated={as_text(get_field(report, 'ModulatedParameter'))}, "
            f"sensation={as_text(get_field(report, 'Sensation'))}"
        )


def visualize(mat_path: Path, save_only: bool = False) -> None:
    data = load_session(mat_path)
    reports = iter_reports(data)
    print_session_summary(mat_path, data, reports)

    if not reports:
        print("\nNo reports found to visualize.")
        return

    output_dir = SCRIPT_DIR / f"{mat_path.stem}_visualization"
    output_dir.mkdir(exist_ok=True)

    hand = as_text(get_field(data, "Hand"))
    hand_image_path = find_hand_image(hand)
    hand_image = cv2.imread(str(hand_image_path), cv2.IMREAD_COLOR) if hand_image_path else None

    if hand_image_path:
        print(f"\nUsing hand image: {hand_image_path}")
    else:
        print(f"\nNo hand image found for hand={hand}; showing maps only.")

    for name, report in reports:
        selection_map = normalize_map(get_field(report, "Map"))
        title = (
            f"{name} | {hand} | {as_text(get_field(report, 'Sensation'))} | "
            f"Map {selection_map.shape[0]}x{selection_map.shape[1]}"
        )

        if hand_image is not None:
            overlay, warning = make_overlay(hand_image, selection_map)
            if warning:
                print(f"Warning for {name}: {warning}")
            preview = combine_preview(selection_map, overlay, title)
        else:
            preview = add_title(make_map_preview(selection_map), title)

        output_path = output_dir / f"{name}.png"
        cv2.imwrite(str(output_path), preview)
        print(f"Saved preview: {output_path}")

        if not save_only:
            window_title = f"{mat_path.name} - {name}"
            cv2.imshow(window_title, preview)

    if not save_only:
        print("\nPress any key in an image window to close all windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a SensoryNBLab session .mat file.")
    parser.add_argument(
        "mat_file",
        nargs="?",
        help="Path to the .mat file. Defaults to session.mat, then *_session.mat.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save PNG previews without opening image windows.",
    )
    args = parser.parse_args()

    if args.mat_file:
        mat_path = Path(args.mat_file).expanduser()
        if not mat_path.is_absolute():
            cwd_path = Path.cwd() / mat_path
            mat_path = cwd_path if cwd_path.exists() else SCRIPT_DIR / mat_path
    else:
        mat_path = find_default_mat_file()
    mat_path = mat_path.resolve()

    visualize(mat_path, save_only=args.save_only)


if __name__ == "__main__":
    main()
