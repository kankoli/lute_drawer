import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from bowl_from_soundboard import set_axes_equal_3d, build_bowl_for_lute, LuteCurve

def rib_surface_extended(rib1, rib2, plane_offset=10.0):
    """
    Build rib surface extended to two parallel planes (left/right).
    Includes tip handling so the surface closes at rib ends.
    Returns polygons for the extended surface, plus the plane definitions.
    """
    rib1, rib2 = np.array(rib1), np.array(rib2)
    n = min(len(rib1), len(rib2))
    rib1, rib2 = rib1[:n], rib2[:n]

    across = rib2.mean(axis=0) - rib1.mean(axis=0)
    across = across / np.linalg.norm(across)

    nrm = across
    p0_left = rib1.mean(axis=0) - plane_offset * nrm
    p0_right = rib2.mean(axis=0) + plane_offset * nrm

    def intersect_line_plane(p1, p2, p0, nrm):
        u = p2 - p1
        denom = np.dot(nrm, u)
        if abs(denom) < 1e-9:
            return None
        t = np.dot(nrm, p0 - p1) / denom
        return p1 + t * u

    strips = []

    # --- top tip ---
    tip_top = (rib1[0] + rib2[0]) / 2
    left_top = intersect_line_plane(rib1[0], rib2[0], p0_left, nrm)
    right_top = intersect_line_plane(rib1[0], rib2[0], p0_right, nrm)
    if left_top is None: left_top = rib1[0]
    if right_top is None: right_top = rib2[0]
    strips.append([left_top, tip_top, tip_top, right_top])

    # --- intermediate samples ---
    for j in range(1, n - 1):
        a = rib1[j]
        b = rib2[j]
        left_pt = intersect_line_plane(a, b, p0_left, nrm)
        right_pt = intersect_line_plane(a, b, p0_right, nrm)
        if left_pt is not None and right_pt is not None:
            strips.append([left_pt, a, b, right_pt])

    # --- bottom tip ---
    tip_bot = (rib1[-1] + rib2[-1]) / 2
    left_bot = intersect_line_plane(rib1[-1], rib2[-1], p0_left, nrm)
    right_bot = intersect_line_plane(rib1[-1], rib2[-1], p0_right, nrm)
    if left_bot is None: left_bot = rib1[-1]
    if right_bot is None: right_bot = rib2[-1]
    strips.append([left_bot, tip_bot, tip_bot, right_bot])

    # Surface quads
    quads = []
    for j in range(len(strips) - 1):
        s1, s2 = strips[j], strips[j + 1]
        for k in range(4):
            p1, p2 = np.array(s1[k]), np.array(s1[(k + 1) % 4])
            q2, q1 = np.array(s2[(k + 1) % 4]), np.array(s2[k])
            quads.append(np.array([p1, p2, q2, q1]))

    # Caps (optional, but helps visualization)
    quads.append(np.array([np.array(p) for p in strips[0]]))
    quads.append(np.array([np.array(p) for p in strips[-1]]))

    return quads, (p0_left, p0_right, nrm)


def plot_extended_surface(rib1, rib2, plane_offset=10.0, title="Extended Rib Surface"):
    """Plot extended rib surface + original outlines + cutting planes."""
    quads, (p0_left, p0_right, nrm) = rib_surface_extended(rib1, rib2, plane_offset)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extended surface
    poly = Poly3DCollection(quads, alpha=0.6,
                            facecolor="lightblue", edgecolor="k", linewidths=0.3)
    ax.add_collection3d(poly)

    # Original rib outlines
    rib1 = np.asarray(rib1)
    rib2 = np.asarray(rib2)
    ax.plot(rib1[:, 0], rib1[:, 1], rib1[:, 2], color="red", lw=2, label="rib1")
    ax.plot(rib2[:, 0], rib2[:, 1], rib2[:, 2], color="green", lw=2, label="rib2")

    # Show planes (as translucent quads)
    def plane_patch(p0, nrm, size=50):
        v = np.array([1, 0, 0])
        if abs(np.dot(v, nrm)) > 0.9:
            v = np.array([0, 1, 0])
        u = np.cross(nrm, v); u /= np.linalg.norm(u)
        v = np.cross(nrm, u); v /= np.linalg.norm(v)
        corners = [p0 + sx*u*size + sy*v*size
                   for sx, sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        return np.array(corners)

    ax.add_collection3d(Poly3DCollection([plane_patch(p0_left, nrm)],
                                         alpha=0.2, facecolor="red"))
    ax.add_collection3d(Poly3DCollection([plane_patch(p0_right, nrm)],
                                         alpha=0.2, facecolor="green"))

    # Equal aspect
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

    plot_extended_surface(ribs[2], ribs[3], plane_offset=15.0, title="Rib 3 Extended")