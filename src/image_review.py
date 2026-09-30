"""Read-only channel inspection, perceptual hashes and review thumbnails."""

import io
import hashlib
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fft import dctn


def perceptual_hash(image):
    """64 bits: L/32x32 Lanczos, DCT-II ortho, 8x8 block > its median.

    DC is included, explicitly. This is an audit representation, not CNN
    preprocessing. Hamming distance <=4 is the conservative candidate rule.
    """
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    block = dctn(np.asarray(gray, dtype=float), type=2, norm="ortho")[:8, :8]
    bits = (block > np.median(block)).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class BKTree:
    """Exact radius search in Hamming metric; no full-resolution pair scan."""

    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        found = []
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                found.extend((index, distance) for index in node[1])
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)
        return sorted(found)


def channel_differences(image):
    """RGB numerical differences are not proof of clinically useful color."""
    if image.mode != "RGB":
        return {
            "rgb_channels_identical": None, "max_channel_difference": None,
            "mean_channel_difference": None,
            "actual_channel_differences": False,
            "nonzero_spread_fraction": None, "strong_color_fraction": None,
            "colored_marker_assessment": "NOT_RGB",
        }
    rgb = np.asarray(image, dtype=np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    pair_mean = (
        np.abs(rgb[..., 0] - rgb[..., 1]).mean()
        + np.abs(rgb[..., 0] - rgb[..., 2]).mean()
        + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    ) / 3
    strong_fraction = float((spread >= 20).mean())
    difference = bool(spread.max() > 0)
    return {
        "rgb_channels_identical": not difference,
        "max_channel_difference": int(spread.max()),
        "mean_channel_difference": float(pair_mean),
        "actual_channel_differences": difference,
        "nonzero_spread_fraction": float((spread > 0).mean()),
        "strong_color_fraction": strong_fraction,
        "colored_marker_assessment": (
            "NO_COLOR_DIFFERENCES" if not difference else
            "POSSIBLE_COLOR_REQUIRES_VISUAL_REVIEW"
        ),
    }


def inspect_canonical(frame, root, output):
    """Read each canonical source once; verify its hash before decoding."""
    rows = []
    for index, row in enumerate(frame.itertuples(), 1):
        path = (root / row.filepath).resolve()
        if root not in path.parents:
            raise ValueError("Canonical path outside raw root")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row.sha256_hash:
            raise ValueError(f"Changed raw image: {row.filepath}")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.mode != row.image_mode:
                raise ValueError("Stale image-mode metadata")
            rows.append({
                "filepath": row.filepath, "filename": row.filename,
                "sha256_hash": row.sha256_hash,
                "target_label": row.target_label,
                "image_mode": image.mode,
                "channel_count": len(image.getbands()),
                "phash_hex": f"{perceptual_hash(image):016x}",
                **channel_differences(image),
            })
        if index % 500 == 0:
            print(f"Channel/hash inspection: {index}/{len(frame)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "channel_mode_audit.csv", index=False)
    return result


def near_duplicate_pairs(frame, channels, radius=4):
    """Enumerate every radius match exactly once in canonical hash order."""
    tree = BKTree()
    records = list(frame.itertuples())
    values = [int(v, 16) for v in channels.phash_hex]
    rows = []
    for index, value in enumerate(values):
        for previous, distance in tree.query(value, radius):
            a, b = records[previous], records[index]
            rows.append({
                "image_1": a.filepath, "image_2": b.filepath,
                "target_1": a.target_label, "target_2": b.target_label,
                "phash_distance": distance,
                "sha256_1": a.sha256_hash, "sha256_2": b.sha256_hash,
                "original_splits_1": a.all_original_splits,
                "original_splits_2": b.all_original_splits,
                "all_source_paths_1": a.all_filepaths,
                "all_source_paths_2": b.all_filepaths,
                "cross_class": a.target_label != b.target_label,
                "cross_original_split": a.original_split != b.original_split,
                "review_status": "CANDIDATE_NOT_CONFIRMED",
            })
        tree.add(value, index)
    columns = [
        "image_1", "image_2", "target_1", "target_2", "phash_distance",
        "sha256_1", "sha256_2", "original_splits_1", "original_splits_2",
        "all_source_paths_1", "all_source_paths_2", "cross_class",
        "cross_original_split", "review_status",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["cross_class", "cross_original_split", "phash_distance",
             "sha256_1", "sha256_2"],
            ascending=[False, False, True, True, True],
        ).reset_index(drop=True)
    return result


def make_sheet(items, root, destination, title, columns=4):
    """Display-only thumbnails; preserve ratio, no contrast enhancement."""
    cell_w, cell_h = 330, 320
    rows = max(1, (len(items) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_w, 62 + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    draw.text((12, 10), title, fill="black", font=font)
    draw.text((12, 32), "Audit thumbnails; no clinical judgment or exclusion.",
              fill="black", font=font)
    if not items:
        draw.text((12, 90), "No images meet this review category.",
                  fill="black", font=font)
    for index, (path, label, reason) in enumerate(items):
        x, y = (index % columns) * cell_w, 62 + (index // columns) * cell_h
        with Image.open(root / path) as image:
            thumb = ImageOps.contain(
                image.convert("RGB"), (cell_w - 14, 218),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
        lines = textwrap.wrap(Path(path).name, width=38)[:2]
        lines += [label] + textwrap.wrap(reason, width=39)[:3]
        draw.multiline_text((x + 7, y + 222), "\n".join(lines),
                            fill="black", font=font, spacing=1)
    canvas.save(destination, format="PNG")


def review_sheets(frame, channels, pairs, root, figures, top):
    """Deterministic review sets with a thumbnail-to-source index."""
    directory = figures / "manual_review"
    directory.mkdir(parents=True, exist_ok=True)
    index_rows = []

    def emit(name, items, title):
        make_sheet(items, root, directory / name, title)
        for number, (path, label, reason) in enumerate(items, 1):
            index_rows.append({
                "sheet": name, "thumbnail_number": number, "filepath": path,
                "target_label": label, "reason": reason,
            })

    lookup = frame.set_index("sha256_hash")
    top_items = [(lookup.loc[r.sha256_hash].filepath,
                  lookup.loc[r.sha256_hash].target_label, r.reasons)
                 for r in top.itertuples()]
    for start in range(0, len(top_items), 15):
        emit(f"A_top30_{start // 15 + 1:02}.png", top_items[start:start + 15],
             "A: top previously flagged unique contents")
    for letter, metric in zip("BCDEF", [
        "width", "height", "aspect_ratio", "mean_intensity", "std_intensity",
    ]):
        items = []
        for label, part in frame.groupby("target_label", sort=True):
            ordered = part.sort_values([metric, "sha256_hash"])
            for tail, selected in [("low", ordered.head(2)),
                                   ("high", ordered.tail(2))]:
                items.extend((r.filepath, label,
                              f"{tail} {metric}={getattr(r, metric):.2f}")
                             for r in selected.itertuples())
        emit(f"{letter}_{metric}.png", items,
             f"{letter}: two low and two high {metric} per class")
    color = channels[channels.actual_channel_differences]
    color_items = [(r.filepath, r.target_label,
                    f"RGB max difference={r.max_channel_difference}")
                   for r in color.itertuples()]
    for start in range(0, max(1, len(color_items)), 16):
        emit(f"G_RGB_differences_{start // 16 + 1:02}.png",
             color_items[start:start + 16], "G: RGB channel differences")
    selected = pd.concat([
        pairs[pairs.cross_class].head(8),
        pairs[pairs.cross_original_split].head(8),
        pairs.sort_values(["phash_distance", "sha256_1", "sha256_2"]).head(24),
    ]).drop_duplicates(["sha256_1", "sha256_2"]).head(40)
    for start in range(0, len(selected), 8):
        items = []
        for pair in selected.iloc[start:start + 8].itertuples():
            reason = (
                f"pair {pair.Index + 1}, pHash distance={pair.phash_distance}"
            )
            items.extend([(pair.image_1, pair.target_1, reason),
                          (pair.image_2, pair.target_2, reason)])
        name = f"H_near_duplicates_{start // 8 + 1:02}.png"
        emit(name, items, "H: adjacent thumbnails form a candidate pair")
        make_sheet(items, root,
                   figures / f"near_duplicate_review_{start // 8 + 1:02}.png",
                   "Candidate pairs; pHash similarity does not prove identity")
    pd.DataFrame(index_rows).to_csv(
        directory / "review_index.csv", index=False
    )
