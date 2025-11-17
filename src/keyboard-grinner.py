import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Literal, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle as MplCircle

# セグメントの種別:
#   "horizontal" : 回転なし
#   "bottom1"    : 左側の下円で回転
#   "top"        : 上円で回転
#   "bottom2"    : 右側の下円で回転
SegmentKind = Literal["horizontal", "bottom1", "top", "bottom2"]


@dataclass
class RowSpec:
    """
    1行分の定義。

    key_widths:
        左から順の矩形幅（0.25単位の整数倍を想定。標準キーなら 1.0 など）
        行内の幅の総和は、すべての行で等しい前提。

    segment_lengths:
        [水平左, 下円1, 上円, 下円2, 水平右] の矩形個数
        例えば 11キー行なら [2,3,3,2,1] など。

    ※ユーザーが初期値として与えることを想定。
    """
    key_widths: List[float]
    segment_lengths: List[int]

    def __post_init__(self) -> None:
        if len(self.segment_lengths) != 5:
            raise ValueError(
                "segment_lengths must have 5 elements [h_left, b1, top, b2, h_right]"
            )
        if sum(self.segment_lengths) != len(self.key_widths):
            raise ValueError(
                "sum(segment_lengths) must equal len(key_widths)"
            )


@dataclass
class LayoutConfig:
    row_specs: List[RowSpec]
    key_height: float = 1.0
    row_gap: float = 0.2
    max_angle: Optional[float] = None


@dataclass
class RectInstance:
    """
    1つの矩形インスタンス（行・列・幅など＋回転情報）。
    """
    row: int
    index_in_row: int
    width: float
    height: float
    base_x: float  # 回転前の左下x
    base_y: float  # 回転前の左下y（行の下辺）
    segment: SegmentKind

    circle_name: Optional[str] = None  # "top", "bottom_left", "bottom_right", or None
    angle_deg: float = 0.0            # 回転角（度）: ピボット回り

    # 回転後の4頂点（描画用）。shape = (4,2)
    polygon: Optional[np.ndarray] = None


@dataclass
class Circle:
    """
    回転レイアウトの基準となる円。
    各行ごとに異なるy座標と半径を持つ。
    """
    name: str           # "top", "bottom_left", "bottom_right"
    row: int            # 行番号
    cx: float           # x座標（全行で共通）
    cy: float           # y座標（行ごとに異なる）
    radius: float = 0.0  # 半径（行ごとに異なる。描画用に矩形から自動推定）


def build_rects(
    row_specs: List[RowSpec],
    key_height: float = 1.0,
    row_gap: float = 0.2,
) -> List[RectInstance]:
    """
    ユーザー入力（行ごとの RowSpec）から、回転前の矩形レイアウトを生成する。

    y座標系:
        行0（最上段）の下辺を y=0 とし、
        下方向に行くほど y がマイナスになるように配置している。
        （最後に plot 時に invert_yaxis してキーボードっぽく見せる）
    """
    rects: List[RectInstance] = []

    for row_index, spec in enumerate(row_specs):
        # 行の下辺 y
        y = -(key_height + row_gap) * row_index
        x = 0.0

        # segment 割り当ての境界
        s_h_left, s_b1, s_top, s_b2, s_h_right = spec.segment_lengths
        boundaries = [
            s_h_left,
            s_h_left + s_b1,
            s_h_left + s_b1 + s_top,
            s_h_left + s_b1 + s_top + s_b2,
            s_h_left + s_b1 + s_top + s_b2 + s_h_right,
        ]

        for i, w in enumerate(spec.key_widths):
            if i < boundaries[0]:
                seg: SegmentKind = "horizontal"
            elif i < boundaries[1]:
                seg = "bottom1"
            elif i < boundaries[2]:
                seg = "top"
            elif i < boundaries[3]:
                seg = "bottom2"
            else:
                seg = "horizontal"

            rects.append(
                RectInstance(
                    row=row_index,
                    index_in_row=i,
                    width=w,
                    height=key_height,
                    base_x=x,
                    base_y=y,
                    segment=seg,
                )
            )
            x += w

    return rects


def infer_total_width(rects: List[RectInstance]) -> float:
    """
    全体の横幅（最も右の矩形の右端）を返す。
    """
    return max(r.base_x + r.width for r in rects)


def place_circles(
    rects: List[RectInstance],
    row_specs: List[RowSpec],
    key_height: float = 1.0,
    row_gap: float = 0.2,
) -> Dict[int, Dict[str, Circle]]:
    """
    各行に3つの円（中心座標）を配置する。

    方針:
        * x座標は全行で共通:
          - 上円: 全体幅の中央 (W/2)
          - 下円: 全体幅の 1/4, 3/4
        * y座標は各行の矩形位置に基づいて計算:
          - 上円: その行より少し上
          - 下円: その行より少し下

    半径は後で矩形のピボット距離から更新する。
    """
    total_width = infer_total_width(rects)

    # 3つの中心x座標（全行で共通）
    top_cx = total_width * 0.5
    bottom_left_cx = total_width * 0.25
    bottom_right_cx = total_width * 0.75

    circles: Dict[int, Dict[str, Circle]] = {}

    for row_index in range(len(row_specs)):
        # 各行のy座標
        y_row = -(key_height + row_gap) * row_index

        # 各行の円のy座標を計算
        top_cy = y_row + key_height + (key_height + row_gap) * 0.5
        bottom_cy = y_row - (key_height + row_gap) * 0.5

        circles[row_index] = {
            "top": Circle(name="top", row=row_index, cx=top_cx, cy=top_cy),
            "bottom_left": Circle(name="bottom_left", row=row_index, cx=bottom_left_cx, cy=bottom_cy),
            "bottom_right": Circle(name="bottom_right", row=row_index, cx=bottom_right_cx, cy=bottom_cy),
        }

    return circles


def assign_circles_to_rects(rects: List[RectInstance], circles: Dict[int, Dict[str, Circle]]) -> None:
    """
    segment 種別に応じて、各矩形に円を割り当てる。

    * "top"     セグメント → 上円 "top"
    * "bottom1" セグメント → 左下円 "bottom_left"
    * "bottom2" セグメント → 右下円 "bottom_right"
    * "horizontal"           → 円なし（回転なし）

    各矩形は自分の行の円を参照する。
    """
    for r in rects:
        if r.segment == "top":
            r.circle_name = "top"
        elif r.segment == "bottom1":
            r.circle_name = "bottom_left"
        elif r.segment == "bottom2":
            r.circle_name = "bottom_right"
        else:
            r.circle_name = None


def choose_pivot(rect: RectInstance, circle: Circle) -> np.ndarray:
    """
    円中心に最も近い角をピボットにする。
    """
    x0, y0 = rect.base_x, rect.base_y
    x1, y1 = rect.base_x + rect.width, rect.base_y + rect.height
    corners = np.array(
        [
            [x0, y0],  # 左下
            [x1, y0],  # 右下
            [x1, y1],  # 右上
            [x0, y1],  # 左上
        ]
    )
    d2 = np.sum((corners - np.array([circle.cx, circle.cy])) ** 2, axis=1)
    pivot = corners[int(np.argmin(d2))]
    return pivot


def compute_rotation_angles(
    rects: List[RectInstance],
    circles: Dict[int, Dict[str, Circle]],
    max_abs_angle_deg: float = 90.0,
) -> None:
    """
    各矩形について回転角度を決定する。

    ロジック:
        * ピボット P（矩形の角）を選ぶ。
        * 回転前の矩形中心 C0 を求める。
        * ベクトル u = C0 - P, v = 円中心 O - P を計算。
        * u を v の方向に向けたいので、angle = arg(v) - arg(u) とする。
        * 角度を -180〜180 に正規化し、±max_abs_angle_deg にクリップ。

    これにより、「ピボットから見て矩形中心が円中心方向を向く」ように傾く。
    各矩形は自分の行の円を参照する。
    """
    for r in rects:
        if r.circle_name is None:
            r.angle_deg = 0.0
            continue

        circle = circles[r.row][r.circle_name]
        pivot = choose_pivot(r, circle)

        # 回転前の中心
        cx0 = r.base_x + r.width * 0.5
        cy0 = r.base_y + r.height * 0.5
        u = np.array([cx0 - pivot[0], cy0 - pivot[1]])
        v = np.array([circle.cx - pivot[0], circle.cy - pivot[1]])

        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
            r.angle_deg = 0.0
            continue

        angle_u = math.degrees(math.atan2(u[1], u[0]))
        angle_v = math.degrees(math.atan2(v[1], v[0]))
        angle = angle_v - angle_u

        # -180〜180 に正規化
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0

        # 制限（絶対値 90° を超えないように）
        if angle > max_abs_angle_deg:
            angle = max_abs_angle_deg
        if angle < -max_abs_angle_deg:
            angle = -max_abs_angle_deg

        r.angle_deg = angle


def rotate_rectangles(rects: List[RectInstance], circles: Dict[int, Dict[str, Circle]]) -> None:
    """
    計算済みの angle_deg を使って、各矩形の回転後ポリゴンを求める。
    各矩形は自分の行の円を参照する。
    """
    for r in rects:
        # 基本の4頂点（回転前）
        x0, y0 = r.base_x, r.base_y
        x1, y1 = r.base_x + r.width, r.base_y + r.height
        corners = np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ]
        )

        if r.circle_name is None or abs(r.angle_deg) < 1e-6:
            # 回転なし（水平）
            r.polygon = corners
            continue

        circle = circles[r.row][r.circle_name]
        pivot = choose_pivot(r, circle)

        theta = math.radians(r.angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        shifted = corners - pivot
        rotated = shifted @ R.T + pivot
        r.polygon = rotated


def update_circle_radii(rects: List[RectInstance], circles: Dict[int, Dict[str, Circle]]) -> None:
    """
    描画用に、各行の各円の半径を「その行の担当矩形のピボット〜円中心距離の平均」で設定する。
    本質的なロジックではなく、単に見た目のため。
    """
    # 行ごと、円名ごとの距離を蓄積
    accum: Dict[int, Dict[str, List[float]]] = {}
    for row in circles.keys():
        accum[row] = {name: [] for name in circles[row].keys()}

    for r in rects:
        if r.circle_name is None:
            continue
        circle = circles[r.row][r.circle_name]
        pivot = choose_pivot(r, circle)
        d = math.dist((pivot[0], pivot[1]), (circle.cx, circle.cy))
        accum[r.row][r.circle_name].append(d)

    # 各行の各円の半径を平均距離で更新
    for row in circles.keys():
        for name, dists in accum[row].items():
            if dists:
                circles[row][name].radius = sum(dists) / len(dists)
            else:
                circles[row][name].radius = 0.0


def evaluate_row_gaps(rects: List[RectInstance], row_count: int) -> List[float]:
    """
    各行ごとの y 範囲から、上下の行間ギャップ（最小縦距離）を計算する。
    回転後ポリゴンが必要。
    """
    # 行ごとの min_y, max_y
    min_y = [float("inf")] * row_count
    max_y = [float("-inf")] * row_count

    for r in rects:
        if r.polygon is None:
            raise RuntimeError("rectangles must be rotated before evaluating gaps")
        ys = r.polygon[:, 1]
        min_y[r.row] = min(min_y[r.row], float(np.min(ys)))
        max_y[r.row] = max(max_y[r.row], float(np.max(ys)))

    gaps: List[float] = []
    for row in range(row_count - 1):
        upper = row      # 行番号が小さい方が上
        lower = row + 1
        gap = min_y[upper] - max_y[lower]
        gaps.append(gap)
    return gaps


def plot_layout(rects: List[RectInstance], circles: Dict[int, Dict[str, Circle]]) -> None:
    """
    matplotlib で矩形と円を描画する。
    各行の円を描画する。
    """
    fig, ax = plt.subplots()
    ax.set_facecolor("white")

    if rects:
        row_count = max(r.row for r in rects) + 1
    else:
        row_count = 0
    cmap = plt.get_cmap("tab10", max(row_count, 1))

    # 矩形
    for r in rects:
        poly = r.polygon
        if poly is None:
            continue
        color = cmap(r.row % cmap.N)
        patch = Polygon(
            poly,
            closed=True,
            fill=True,
            facecolor=(color[0], color[1], color[2], 0.35),
            edgecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(patch)

    # 円（各行の円を描画）
    for row_circles in circles.values():
        for c in row_circles.values():
            if c.radius <= 0:
                continue
            circle_patch = MplCircle((c.cx, c.cy), c.radius, fill=False, linestyle="--")
            ax.add_patch(circle_patch)

    if rects:
        xs = np.concatenate([r.polygon[:, 0] for r in rects if r.polygon is not None])
        ys = np.concatenate([r.polygon[:, 1] for r in rects if r.polygon is not None])
        padding = 0.5
        ax.set_xlim(float(xs.min() - padding), float(xs.max() + padding))
        ax.set_ylim(float(ys.min() - padding), float(ys.max() + padding))
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # 行0が画面上側になるよう反転
    plt.show()


def run_layout(
    row_specs: List[RowSpec],
    *,
    key_height: float,
    row_gap: float,
    max_abs_angle_deg: float,
) -> Tuple[List[RectInstance], Dict[str, Circle], List[float]]:
    """パラメータ一式を受け取り、矩形・円・ギャップを計算して返す。"""
    rects = build_rects(row_specs, key_height=key_height, row_gap=row_gap)
    circles = place_circles(rects, row_specs, key_height=key_height, row_gap=row_gap)
    assign_circles_to_rects(rects, circles)
    compute_rotation_angles(rects, circles, max_abs_angle_deg=max_abs_angle_deg)
    rotate_rectangles(rects, circles)
    update_circle_radii(rects, circles)
    gaps = evaluate_row_gaps(rects, row_count=len(row_specs))
    return rects, circles, gaps


def example_usage(max_abs_angle_deg: float = 90.0) -> None:
    """
    最低限の例。

    * 行数: 4
    * 各行 11キー（すべて標準幅=1.0）
    * セグメント: [水平2, 下円3, 上円3, 下円2, 水平1] など任意
      （本番では、ここを「水平は1 or 2」「下円は1〜3」などの制約を守る形で
       ユーザー入力から決める想定）
    * max_abs_angle_deg: 1矩形あたりの回転角の上限（度数法）
    """
    row_count = 4
    key_height = 1.0
    row_gap = 0.3

    row_specs: List[RowSpec] = []
    for _ in range(row_count):
        key_widths = [1.0] * 11
        segment_lengths = [2, 3, 3, 2, 1]  # 2+3+3+2+1 = 11
        row_specs.append(RowSpec(key_widths=key_widths, segment_lengths=segment_lengths))

    rects, circles, gaps = run_layout(
        row_specs,
        key_height=key_height,
        row_gap=row_gap,
        max_abs_angle_deg=max_abs_angle_deg,
    )
    print("row gaps:", gaps)
    plot_layout(rects, circles)


def parse_layout_matrix(layout_matrix: List[List[Any]]) -> List[List[float]]:
    """
    KLE 由来のレイアウト配列を矩形幅のリストに変換する。

    フォーマット例:
        [
          ["Q", "W", ...],
          [{"w": 1.25}, "A", ...]
        ]

    dict 要素は直後のキーに対する width などの修飾を表すとみなし、
    ここでは `w` のみサポートする。`x`/`y` シフトなどは未対応。
    """
    if not isinstance(layout_matrix, list) or not layout_matrix:
        raise ValueError("layout_matrix must be a non-empty list of rows")

    rows: List[List[float]] = []
    totals: List[float] = []

    for row_index, row in enumerate(layout_matrix):
        if not isinstance(row, list) or not row:
            raise ValueError(f"layout_matrix[{row_index}] must be a non-empty list")

        widths: List[float] = []
        pending_width = 1.0

        for token in row:
            if isinstance(token, dict):
                if "x" in token and float(token["x"]) != 0.0:
                    raise ValueError(
                        "x offsets in layout_matrix are not supported yet (row %d)" % row_index
                    )
                if "w" in token:
                    pending_width = float(token["w"])
                continue

            if isinstance(token, str):
                widths.append(float(pending_width))
                pending_width = 1.0
            else:
                raise ValueError(
                    f"layout_matrix[{row_index}] contains unsupported token type: {type(token)!r}"
                )

        if not widths:
            raise ValueError(f"layout_matrix[{row_index}] produced no keys")

        rows.append(widths)
        totals.append(sum(widths))

    first_total = totals[0]
    for idx, total in enumerate(totals[1:], start=1):
        if abs(total - first_total) > 1e-6:
            raise ValueError(
                "all rows in layout_matrix must have the same total width; "
                f"row 0 has {first_total}, row {idx} has {total}"
            )

    return rows


def _validate_segment_lengths(values: List[int], expected: int, *, label: str) -> List[int]:
    if len(values) != 5:
        raise ValueError(f"{label} segment_lengths must have 5 integers")
    if sum(values) != expected:
        raise ValueError(
            f"{label} segment_lengths must sum to {expected} (got {sum(values)})"
        )
    return [int(v) for v in values]


def row_specs_from_layout_matrix(
    layout_matrix: List[List[Any]],
    *,
    default_segments: Optional[List[int]] = None,
    per_row_segments: Optional[List[List[int]]] = None,
) -> List[RowSpec]:
    widths_per_row = parse_layout_matrix(layout_matrix)

    if per_row_segments is not None:
        if len(per_row_segments) != len(widths_per_row):
            raise ValueError(
                "segment_lengths_per_row must match number of rows in layout_matrix"
            )

    specs: List[RowSpec] = []
    for row_index, widths in enumerate(widths_per_row):
        if per_row_segments is not None:
            segments_raw = per_row_segments[row_index]
        elif default_segments is not None:
            segments_raw = default_segments
        else:
            segments_raw = [len(widths), 0, 0, 0, 0]

        segments = _validate_segment_lengths(
            [int(v) for v in segments_raw],
            len(widths),
            label=f"layout row {row_index}",
        )

        specs.append(RowSpec(key_widths=list(widths), segment_lengths=list(segments)))

    return specs


def load_layout_config(path: Path) -> LayoutConfig:
    data = json.loads(path.read_text())

    row_specs: List[RowSpec]

    if "layout_matrix" in data:
        default_segments = data.get("segment_lengths")
        per_row_segments = data.get("segment_lengths_per_row")
        row_specs = row_specs_from_layout_matrix(
            data["layout_matrix"],
            default_segments=default_segments,
            per_row_segments=per_row_segments,
        )
    elif "row_specs" in data:
        row_specs = []
        for idx, entry in enumerate(data["row_specs"]):
            if "key_widths" not in entry or "segment_lengths" not in entry:
                raise ValueError(
                    f"row_specs[{idx}] must contain key_widths and segment_lengths"
                )
            repeat = int(entry.get("repeat", 1))
            if repeat <= 0:
                raise ValueError(f"row_specs[{idx}].repeat must be >= 1")
            key_widths = [float(w) for w in entry["key_widths"]]
            segments = _validate_segment_lengths(
                [int(s) for s in entry["segment_lengths"]],
                len(key_widths),
                label=f"row_specs[{idx}]",
            )
            for _ in range(repeat):
                row_specs.append(RowSpec(key_widths=list(key_widths), segment_lengths=segments))
        if not row_specs:
            raise ValueError("config must define at least one row_spec entry")
    else:
        raise ValueError("config must define either layout_matrix or row_specs")

    key_height = float(data.get("key_height", 1.0))
    row_gap = float(data.get("row_gap", 0.3))
    max_angle = data.get("max_angle")
    if max_angle is not None:
        max_angle = float(max_angle)

    return LayoutConfig(
        row_specs=row_specs,
        key_height=key_height,
        row_gap=row_gap,
        max_angle=max_angle,
    )


def default_layout_config() -> LayoutConfig:
    layout_matrix = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "Back Space"],
        [{"w": 1.25}, "A", "S", "D", "F", "G", "H", "J", "K", "L", {"w": 1.75}, "\"\n'"],
        [{"w": 1.75}, "Z", "X", "C", "V", "B", "N", "M", "<\n,", ">\n.", {"w": 1.25}, "?\n/"],
        [{"w": 1.5}, "Super", {"w": 1.5}, "Super", {"w": 2.25}, "Meta", {"a": 0, "w": 2.75}, "", {"a": 4, "w": 1.5}, "Meta", {"w": 1.5}, "Super"],
    ]
    segment_lengths_per_row = [
        [2, 3, 3, 2, 1],
        [1, 3, 3, 2, 1],
        [1, 3, 3, 2, 1],
        [2, 0, 2, 0, 2],
    ]
    row_specs = row_specs_from_layout_matrix(
        layout_matrix, per_row_segments=segment_lengths_per_row
    )
    return LayoutConfig(
        row_specs=row_specs,
        key_height=1.0,
        row_gap=0.3,
        max_angle=20.0,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Keyboard layout previewer with rotational segments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="path to JSON config that defines row_specs, key_height, row_gap, etc.",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        help="maximum absolute rotation angle per rectangle in degrees",
    )
    parser.add_argument(
        "--key-height",
        type=float,
        help="override key height (default 1.0 or config value)",
    )
    parser.add_argument(
        "--row-gap",
        type=float,
        help="override vertical gap between rows (default 0.3 or config value)",
    )
    args = parser.parse_args()

    if args.config:
        layout = load_layout_config(args.config)
    else:
        layout = default_layout_config()

    key_height = args.key_height if args.key_height is not None else layout.key_height
    row_gap = args.row_gap if args.row_gap is not None else layout.row_gap

    max_angle = layout.max_angle if layout.max_angle is not None else 90.0
    if args.max_angle is not None:
        max_angle = args.max_angle

    rects, circles, gaps = run_layout(
        layout.row_specs,
        key_height=key_height,
        row_gap=row_gap,
        max_abs_angle_deg=max_angle,
    )
    print("row gaps:", gaps)
    plot_layout(rects, circles)


if __name__ == "__main__":
    main()
