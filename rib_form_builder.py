import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from bowl_from_soundboard import set_axes_equal_3d, build_bowl_for_lute, LuteCurve

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


def rib_surface_extended(
    rib1, rib2,
    plane_offset=10.0,
    allowance_left=0.0,
    allowance_right=0.0,
    end_extension=10.0
):
    """
    Build rib surface extended to two parallel planes (left/right).
    Enhancements:
      - Ribbon continues as a ribbon past the tips (end_extension).
      - Independent left/right allowances.
    Returns polygons for the extended surface, plus the plane definitions.
    """
    rib1, rib2 = np.array(rib1, dtype=float), np.array(rib2, dtype=float)
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
    across0 = (rib2[1] - rib1[1]) if n > 1 else np.array([1.0, 0.0, 0.0])
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
    quads.append(np.array([np.array(p) for p in strips[0]]))
    quads.append(np.array([np.array(p) for p in strips[-1]]))

    return quads, (p0_left, p0_right, nrm)


def plot_extended_surface(
    rib1, rib2,
    plane_offset=10.0,
    allowance_left=0.0,
    allowance_right=0.0,
    end_extension=10.0,
    title="Extended Rib Surface"
):
    """Plot extended rib surface + original outlines + cutting planes."""
    quads, (p0_left, p0_right, nrm) = rib_surface_extended(
        rib1, rib2,
        plane_offset=plane_offset,
        allowance_left=allowance_left,
        allowance_right=allowance_right,
        end_extension=end_extension
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(quads, alpha=0.6,
                            facecolor="lightblue", edgecolor="k", linewidths=0.3)
    ax.add_collection3d(poly)

    rib1 = np.asarray(rib1)
    rib2 = np.asarray(rib2)
    ax.plot(rib1[:, 0], rib1[:, 1], rib1[:, 2], color="red", lw=2, label="rib1")
    ax.plot(rib2[:, 0], rib2[:, 1], rib2[:, 2], color="green", lw=2, label="rib2")

    def plane_patch(p0, nrm, size=50):
        v = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v, nrm)) > 0.9:
            v = np.array([0.0, 1.0, 0.0])
        u = np.cross(nrm, v); u /= np.linalg.norm(u) + EPS
        v = np.cross(nrm, u); v /= np.linalg.norm(v) + EPS
        corners = [p0 + sx*u*size + sy*v*size
                   for sx, sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        return np.array(corners)

    ax.add_collection3d(Poly3DCollection([plane_patch(p0_left, nrm)],
                                         alpha=0.2, facecolor="red"))
    ax.add_collection3d(Poly3DCollection([plane_patch(p0_right, nrm)],
                                         alpha=0.2, facecolor="green"))

    pts = np.vstack(quads + [rib1, rib2])
    set_axes_equal_3d(ax, pts[:, 0], pts[:, 1], pts[:, 2])

    ax.set_title(title)
    ax.legend()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import lutes

    lute = lutes.ManolLavta()
    lute.draw_all()

    sections, ribs = build_bowl_for_lute(
        lute,
        n_ribs=13,
        n_sections=300,
        top_curve=LuteCurve
    )

    plot_extended_surface(
        ribs[12], ribs[13],
        plane_offset=15.0,
        allowance_left=2.0,
        allowance_right=4.0,
        end_extension=12.0,
        title="Rib 3 Extended with Endpoint Caps + Overshoot"
    )
