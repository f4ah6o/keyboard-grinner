from dataclasses import dataclass
from typing import List, Optional, Literal, Dict
import math

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
    回転レイアウトの基準となる円（上1・下2）。
    """
    name: str           # "top", "bottom_left", "bottom_right"
    cx: float
    cy: float
    radius: float = 0.0  # 描画用。矩形から自動で推定する。


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
) -> Dict[str, Circle]:
    """
    3円の初期配置を決める。

    方針:
        * 上円: 全体幅の中央に x を置き、最上段の少し上に y を置く。
        * 下円: 全体幅の 1/4, 3/4 あたりに x を置き、最下段の少し下に y を置く。

    半径はここでは決めず、後で矩形のピボット距離から更新する。
    """
    total_width = infer_total_width(rects)

    top_row_index = 0
    bottom_row_index = len(row_specs) - 1
    y_top_row = -(key_height + row_gap) * top_row_index
    y_bottom_row = -(key_height + row_gap) * bottom_row_index

    # 上円の中心（矩形群の中心 x に置く）
    top_cx = total_width * 0.5
    top_cy = y_top_row + key_height + (key_height + row_gap)

    # 下円2つの中心（左右対称）
    bottom_cy = y_bottom_row - (key_height + row_gap)
    bottom_left_cx = total_width * 0.25
    bottom_right_cx = total_width * 0.75

    circles: Dict[str, Circle] = {
        "top": Circle(name="top", cx=top_cx, cy=top_cy),
        "bottom_left": Circle(name="bottom_left", cx=bottom_left_cx, cy=bottom_cy),
        "bottom_right": Circle(name="bottom_right", cx=bottom_right_cx, cy=bottom_cy),
    }
    return circles


def assign_circles_to_rects(rects: List[RectInstance], circles: Dict[str, Circle]) -> None:
    """
    segment 種別に応じて、各矩形に円を割り当てる。

    * "top"     セグメント → 上円 "top"
    * "bottom1" セグメント → 左下円 "bottom_left"
    * "bottom2" セグメント → 右下円 "bottom_right"
    * "horizontal"           → 円なし（回転なし）
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
    circles: Dict[str, Circle],
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
    """
    for r in rects:
        if r.circle_name is None:
            r.angle_deg = 0.0
            continue

        circle = circles[r.circle_name]
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


def rotate_rectangles(rects: List[RectInstance], circles: Dict[str, Circle]) -> None:
    """
    計算済みの angle_deg を使って、各矩形の回転後ポリゴンを求める。
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

        circle = circles[r.circle_name]
        pivot = choose_pivot(r, circle)

        theta = math.radians(r.angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        shifted = corners - pivot
        rotated = shifted @ R.T + pivot
        r.polygon = rotated


def update_circle_radii(rects: List[RectInstance], circles: Dict[str, Circle]) -> None:
    """
    描画用に、各円の半径を「担当矩形のピボット〜円中心距離の平均」で設定する。
    本質的なロジックではなく、単に見た目のため。
    """
    accum: Dict[str, List[float]] = {name: [] for name in circles.keys()}

    for r in rects:
        if r.circle_name is None:
            continue
        circle = circles[r.circle_name]
        pivot = choose_pivot(r, circle)
        d = math.dist((pivot[0], pivot[1]), (circle.cx, circle.cy))
        accum[r.circle_name].append(d)

    for name, dists in accum.items():
        if dists:
            circles[name].radius = sum(dists) / len(dists)
        else:
            circles[name].radius = 0.0


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


def plot_layout(rects: List[RectInstance], circles: Dict[str, Circle]) -> None:
    """
    matplotlib で矩形と円を描画する。
    """
    fig, ax = plt.subplots()

    # 矩形
    for r in rects:
        poly = r.polygon
        if poly is None:
            continue
        patch = Polygon(poly, closed=True, fill=False, linewidth=1.0)
        ax.add_patch(patch)

    # 円（描画用）
    for c in circles.values():
        if c.radius <= 0:
            continue
        circle_patch = MplCircle((c.cx, c.cy), c.radius, fill=False, linestyle="--")
        ax.add_patch(circle_patch)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # 行0が画面上側になるよう反転
    plt.show()


def example_usage() -> None:
    """
    最低限の例。

    * 行数: 4
    * 各行 11キー（すべて標準幅=1.0）
    * セグメント: [水平2, 下円3, 上円3, 下円2, 水平1] など任意
      （本番では、ここを「水平は1 or 2」「下円は1〜3」などの制約を守る形で
       ユーザー入力から決める想定）
    """
    row_count = 4
    key_height = 1.0
    row_gap = 0.3

    row_specs: List[RowSpec] = []
    for _ in range(row_count):
        key_widths = [1.0] * 11
        segment_lengths = [2, 3, 3, 2, 1]  # 2+3+3+2+1 = 11
        row_specs.append(RowSpec(key_widths=key_widths, segment_lengths=segment_lengths))

    # 1. 基準グリッド配置
    rects = build_rects(row_specs, key_height=key_height, row_gap=row_gap)

    # 2. 円の自動配置
    circles = place_circles(rects, row_specs, key_height=key_height, row_gap=row_gap)

    # 3. 矩形への円割り当て
    assign_circles_to_rects(rects, circles)

    # 4. 回転角計算（±90°以内）
    compute_rotation_angles(rects, circles, max_abs_angle_deg=90.0)

    # 5. 回転適用
    rotate_rectangles(rects, circles)

    # 6. 円の半径を描画用に補正
    update_circle_radii(rects, circles)

    # 7. 行間ギャップの評価（最適化の評価関数に使える）
    gaps = evaluate_row_gaps(rects, row_count=row_count)
    print("row gaps:", gaps)

    # 8. 可視化
    plot_layout(rects, circles)


if __name__ == "__main__":
    example_usage()
