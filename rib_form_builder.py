import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from bowl_from_soundboard import set_axes_equal_3d, build_bowl_for_lute, MidCurve

EPS = 1e-9


def _safe_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _align_with_normal(vec, nrm):
    """Flip vec so it points roughly in the same half-space as nrm."""
    vec = np.asarray(vec, dtype=float)
    nrm = np.asarray(nrm, dtype=float)
    return vec if np.dot(vec, nrm) >= 0.0 else -vec


def _intersect_line_plane_pd(p, d, p0, nrm):
    """Line defined by point p and direction d; plane by (p0, nrm)."""
    d = np.asarray(d, dtype=float)
    nrm = np.asarray(nrm, dtype=float)
    denom = float(np.dot(nrm, d))
    if abs(denom) < EPS:
        return None
    t = float(np.dot(nrm, (p0 - p)) / denom)
    return p + t * d


def _intersect_line_plane_p1p2(p1, p2, p0, nrm):
    """Line defined by two points p1->p2; plane by (p0, nrm)."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return _intersect_line_plane_pd(p1, p2 - p1, p0, nrm)


def _rib_surface_extended(outline1, outline2, plane_offset=10.0, allowance_left=0.0, allowance_right=0.0, end_extension=10.0):
    """
    Build rib surface extended to two parallel planes (left/right).
    Returns quads normalized into a local frame (X=along, Y=across, Z=normal),
    translated so the widest connector midpoint is at (0,0,0).
    """
    rib1 = np.array(outline1, float)
    rib2 = np.array(outline2, float)
    n = min(len(rib1), len(rib2))
    rib1, rib2 = rib1[:n], rib2[:n]

    # Across/plane normal based on means
    across = rib2.mean(axis=0) - rib1.mean(axis=0)
    across = _safe_unit(across)
    nrm = across.copy()

    # Parallel planes with independent allowances
    p0_left  = rib1.mean(axis=0) - (plane_offset + allowance_left)  * nrm
    p0_right = rib2.mean(axis=0) + (plane_offset + allowance_right) * nrm

    strips = []

    # ---------- TOP END ----------
    tip_top = 0.5 * (rib1[0] + rib2[0])
    # nearest valid across (parallel to connector near the end)
    across0 = (rib2[1] - rib1[1]) if n > 1 else [1.0, 0.0, 0.0]
    across0 = _align_with_normal(_safe_unit(across0), nrm)
    # endpoint cap (true strip at the endpoint center)
    left_top_cap  = _intersect_line_plane_pd(tip_top, across0, p0_left,  nrm)
    right_top_cap = _intersect_line_plane_pd(tip_top, across0, p0_right, nrm)
    if left_top_cap is None:  left_top_cap  = tip_top - nrm * (plane_offset + allowance_left)
    if right_top_cap is None: right_top_cap = tip_top + nrm * (plane_offset + allowance_right)
    strips.append([left_top_cap, tip_top, tip_top, right_top_cap])

    # overshoot strip beyond endpoint
    tip_dir0 = rib1[1] - rib1[0] if n > 1 else np.array([0.0, 0.0, 1.0])
    tip_dir0 = _safe_unit(tip_dir0)
    tip_top_ext = tip_top - tip_dir0 * end_extension
    left_top_ext  = _intersect_line_plane_pd(tip_top_ext, across0, p0_left,  nrm)
    right_top_ext = _intersect_line_plane_pd(tip_top_ext, across0, p0_right, nrm)
    if left_top_ext is None:  left_top_ext  = tip_top_ext - nrm * (plane_offset + allowance_left)
    if right_top_ext is None: right_top_ext = tip_top_ext + nrm * (plane_offset + allowance_right)
    strips.insert(0, [left_top_ext, tip_top_ext, tip_top_ext, right_top_ext])

    # ---------- INTERMEDIATE STATIONS ----------
    for j in range(1, n - 1):
        a = rib1[j]
        b = rib2[j]
        left_pt  = _intersect_line_plane_p1p2(a, b, p0_left,  nrm)
        right_pt = _intersect_line_plane_p1p2(a, b, p0_right, nrm)
        if left_pt is not None and right_pt is not None:
            strips.append([left_pt, a, b, right_pt])

    # ---------- BOTTOM END ----------
    tip_bot = 0.5 * (rib1[-1] + rib2[-1])
    across1 = (rib2[-2] - rib1[-2]) if n > 1 else np.array([1.0, 0.0, 0.0])
    across1 = _align_with_normal(_safe_unit(across1), nrm)
    # endpoint cap (true strip at the endpoint center)
    left_bot_cap  = _intersect_line_plane_pd(tip_bot, across1, p0_left,  nrm)
    right_bot_cap = _intersect_line_plane_pd(tip_bot, across1, p0_right, nrm)
    if left_bot_cap is None:  left_bot_cap  = tip_bot - nrm * (plane_offset + allowance_left)
    if right_bot_cap is None: right_bot_cap = tip_bot + nrm * (plane_offset + allowance_right)
    strips.append([left_bot_cap, tip_bot, tip_bot, right_bot_cap])

    # overshoot strip beyond endpoint
    tip_dir1 = rib1[-1] - rib1[-2] if n > 1 else np.array([0.0, 0.0, 1.0])
    tip_dir1 = _safe_unit(tip_dir1)
    tip_bot_ext = tip_bot + tip_dir1 * end_extension
    left_bot_ext  = _intersect_line_plane_pd(tip_bot_ext, across1, p0_left,  nrm)
    right_bot_ext = _intersect_line_plane_pd(tip_bot_ext, across1, p0_right, nrm)
    if left_bot_ext is None:  left_bot_ext  = tip_bot_ext - nrm * (plane_offset + allowance_left)
    if right_bot_ext is None: right_bot_ext = tip_bot_ext + nrm * (plane_offset + allowance_right)
    strips.append([left_bot_ext, tip_bot_ext, tip_bot_ext, right_bot_ext])

    # Surface quads (connect every adjacent pair of strips)
    quads = []
    for j in range(len(strips) - 1):
        s1, s2 = strips[j], strips[j + 1]
        for k in range(4):
            p1, p2 = np.array(s1[k]), np.array(s1[(k + 1) % 4])
            q2, q1 = np.array(s2[(k + 1) % 4]), np.array(s2[k])
            quads.append(np.array([p1, p2, q2, q1]))

    # Add caps (first and last strips)
    quads.append(np.array(strips[0]))
    quads.append(np.array(strips[-1]))

    quads_norm = _normalize_quads(rib1, rib2, quads)
    return quads_norm

def _normalize_quads(outline1: np.ndarray, outline2: np.ndarray, quads: list[np.ndarray]):
    """
    Normalize a rib surface given its outlines + quads:
      - Find widest connector → +Y
      - Tangent at that index → +X
      - Normal → +Z
      - Translate so widest connector midpoint → (0,0,0)
    Returns transformed quads.
    """
    outline1 = np.asarray(outline1, float)
    outline2 = np.asarray(outline2, float)
    if outline1.shape != outline2.shape or outline1.ndim != 2 or outline1.shape[1] != 3:
        raise ValueError("_normalize_quads expects two (N,3) outlines with same shape")

    # widest connector
    connectors = outline2 - outline1
    idx = int(np.argmax(np.linalg.norm(connectors, axis=1)))
    across = connectors[idx] / (np.linalg.norm(connectors[idx]) + EPS)

    # tangent direction (along)
    if idx < len(outline1) - 1:
        along = outline1[idx + 1] - outline1[idx]
    else:
        along = outline1[idx] - outline1[idx - 1]
    along /= (np.linalg.norm(along) + EPS)

    # normal
    z_local = np.cross(along, across)
    nz = np.linalg.norm(z_local)
    if nz < EPS:
        return quads
    z_local /= nz
    y_local = across
    x_local = np.cross(y_local, z_local)
    x_local /= (np.linalg.norm(x_local) + EPS)

    # rotation (global→local)
    R = np.vstack([x_local, y_local, z_local]).T
    R_inv = R.T

    def to_local(A):
        return (R_inv @ A.T).T

    # transform all quads
    q_local = [to_local(q) for q in quads]

    # translate so connector midpoint at idx → (0,0,0)
    mid = 0.5 * (to_local(outline1)[idx] + to_local(outline2)[idx])
    q_shifted = [q - mid for q in q_local]

    return q_shifted

def _plot_ribs_extended(rib_quads, spacing=200.0, title="Extended Rib Surfaces"):
    """
    Plot extended rib surfaces (normalized quads).
    Applies Y-spacing between ribs.

    Parameters
    ----------
    rib_quads : list of (idx, [quads])
        Normalized quads per rib index.
    spacing : float
        Spacing applied along Y for each rib index.
    title : str
        Plot title.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    plotted_pts = []
    for idx, quads in rib_quads:
        shift = np.array([0, idx * spacing, 0])
        for q in quads:
            q_shifted = q + shift
            ax.add_collection3d(
                Poly3DCollection([q_shifted],
                                 alpha=0.6,
                                 facecolor="lightblue",
                                 edgecolor="k",
                                 linewidths=0.3)
            )
            plotted_pts.append(q_shifted)

    if plotted_pts:
        pts = np.vstack(plotted_pts)
        set_axes_equal_3d(ax, pts[:, 0], pts[:, 1], pts[:, 2])

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    plt.close(fig)


def plot_lute_ribs(
    lute,
    top_curve,
    rib=1,
    draw_all=False,
    n_ribs=13,
    n_sections=50,
    plane_offset=10.0,
    allowance_left=0.0,
    allowance_right=0.0,
    end_extension=10.0,
    spacing=150.0,
    title="Extended Rib Surfaces (normalized)"
):
    """
    Build normalized rib surfaces for a lute and plot them.
    """
    # --- build bowl geometry
    _, rib_outlines = build_bowl_for_lute(
        lute,
        n_ribs=n_ribs,
        n_sections=n_sections,
        top_curve=top_curve,
    )

    if draw_all:
        rib_indices = range(len(rib_outlines) - 1)
    else:
        rib_indices = [rib-1]

    rib_quads = []
    for idx in rib_indices:
        if idx < 0 or idx >= len(rib_outlines):
            raise ValueError("Rib index is outside range:", idx)
        rib1, rib2 = rib_outlines[idx], rib_outlines[idx+1]
        quads = _rib_surface_extended(
            rib1, rib2,
            plane_offset=plane_offset,
            allowance_left=allowance_left,
            allowance_right=allowance_right,
            end_extension=end_extension,
        )
        rib_quads.append((idx, quads))

    # delegate to the plotter
    _plot_ribs_extended(rib_quads, spacing=spacing, title=title)

if __name__ == "__main__":
    import lutes

    plot_lute_ribs(lutes.ManolLavta(), top_curve=MidCurve, n_ribs=13, rib=7)
