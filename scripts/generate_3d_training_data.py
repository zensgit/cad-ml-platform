"""
Synthetic 3D Point Cloud Training Data Generator.

Generates geometrically meaningful point clouds for each of 8 CAD part types
used by the PointNet classifier. All generation is pure numpy -- no mesh
library required.

Part types (matching PointNetClassifier num_classes=8):
  0: flange    (法兰盘)
  1: shaft     (轴)
  2: shell     (壳体)
  3: bracket   (支架)
  4: gear      (齿轮)
  5: connector (连接件)
  6: seal      (密封件)
  7: other     (其他)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = [
    "flange",
    "shaft",
    "shell",
    "bracket",
    "gear",
    "connector",
    "seal",
    "other",
]


# ---------------------------------------------------------------------------
# Primitive sampling helpers
# ---------------------------------------------------------------------------


def _sample_cylinder_surface(
    rng: np.random.RandomState,
    n: int,
    radius: float,
    height: float,
    z_offset: float = 0.0,
) -> np.ndarray:
    """Sample points on the lateral surface of a cylinder centred at the origin."""
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-height / 2, height / 2, n) + z_offset
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def _sample_disk(
    rng: np.random.RandomState,
    n: int,
    inner_radius: float,
    outer_radius: float,
    z: float = 0.0,
) -> np.ndarray:
    """Sample points on a flat annular disk at height *z*."""
    # sqrt for uniform area sampling
    r = np.sqrt(
        rng.uniform(inner_radius**2, outer_radius**2, n)
    )
    theta = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    zs = np.full(n, z)
    return np.column_stack([x, y, zs])


def _sample_box_surface(
    rng: np.random.RandomState,
    n: int,
    lx: float,
    ly: float,
    lz: float,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Sample points uniformly on the 6 faces of a box centred at the origin."""
    # Face areas
    areas = np.array([ly * lz, ly * lz, lx * lz, lx * lz, lx * ly, lx * ly])
    probs = areas / areas.sum()
    face_ids = rng.choice(6, size=n, p=probs)

    pts = np.zeros((n, 3), dtype=np.float64)
    for fid in range(6):
        mask = face_ids == fid
        cnt = mask.sum()
        if cnt == 0:
            continue
        if fid == 0:  # +x face
            pts[mask, 0] = lx / 2
            pts[mask, 1] = rng.uniform(-ly / 2, ly / 2, cnt)
            pts[mask, 2] = rng.uniform(-lz / 2, lz / 2, cnt)
        elif fid == 1:  # -x face
            pts[mask, 0] = -lx / 2
            pts[mask, 1] = rng.uniform(-ly / 2, ly / 2, cnt)
            pts[mask, 2] = rng.uniform(-lz / 2, lz / 2, cnt)
        elif fid == 2:  # +y face
            pts[mask, 0] = rng.uniform(-lx / 2, lx / 2, cnt)
            pts[mask, 1] = ly / 2
            pts[mask, 2] = rng.uniform(-lz / 2, lz / 2, cnt)
        elif fid == 3:  # -y face
            pts[mask, 0] = rng.uniform(-lx / 2, lx / 2, cnt)
            pts[mask, 1] = -ly / 2
            pts[mask, 2] = rng.uniform(-lz / 2, lz / 2, cnt)
        elif fid == 4:  # +z face
            pts[mask, 0] = rng.uniform(-lx / 2, lx / 2, cnt)
            pts[mask, 1] = rng.uniform(-ly / 2, ly / 2, cnt)
            pts[mask, 2] = lz / 2
        else:  # -z face
            pts[mask, 0] = rng.uniform(-lx / 2, lx / 2, cnt)
            pts[mask, 1] = rng.uniform(-ly / 2, ly / 2, cnt)
            pts[mask, 2] = -lz / 2

    if offset is not None:
        pts += offset
    return pts


def _sample_torus(
    rng: np.random.RandomState,
    n: int,
    major_r: float,
    minor_r: float,
) -> np.ndarray:
    """Sample points on a torus surface using parametric equations."""
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    x = (major_r + minor_r * np.cos(v)) * np.cos(u)
    y = (major_r + minor_r * np.cos(v)) * np.sin(u)
    z = minor_r * np.sin(v)
    return np.column_stack([x, y, z])


def _sample_hexagonal_prism(
    rng: np.random.RandomState,
    n: int,
    circumradius: float,
    height: float,
) -> np.ndarray:
    """Sample points on the surface of a regular hexagonal prism."""
    # 6 rectangular side faces + 2 hexagonal caps
    side_area = 6 * (circumradius * height)  # approximate
    cap_area = 2 * (3 * np.sqrt(3) / 2) * circumradius**2
    total = side_area + cap_area
    n_side = max(1, int(n * side_area / total))
    n_cap = n - n_side

    pts_list: list[np.ndarray] = []

    # Side faces
    if n_side > 0:
        corners_angle = np.array([i * np.pi / 3 for i in range(6)])
        cx = circumradius * np.cos(corners_angle)
        cy = circumradius * np.sin(corners_angle)
        face_idx = rng.randint(0, 6, n_side)
        t = rng.uniform(0, 1, n_side)
        z = rng.uniform(-height / 2, height / 2, n_side)
        i0 = face_idx
        i1 = (face_idx + 1) % 6
        x = cx[i0] * (1 - t) + cx[i1] * t
        y = cy[i0] * (1 - t) + cy[i1] * t
        pts_list.append(np.column_stack([x, y, z]))

    # Caps (top and bottom)
    if n_cap > 0:
        # Sample inside hexagon using rejection sampling
        cap_pts: list[np.ndarray] = []
        while len(cap_pts) < n_cap:
            batch = max(n_cap * 2, 256)
            rx = rng.uniform(-circumradius, circumradius, batch)
            ry = rng.uniform(-circumradius, circumradius, batch)
            # Check if inside hexagon (intersection of 3 pairs of half-planes)
            inside = np.ones(batch, dtype=bool)
            for k in range(3):
                angle = k * np.pi / 3
                d = np.abs(rx * np.cos(angle) + ry * np.sin(angle))
                inside &= d <= circumradius * np.sqrt(3) / 2
            valid_x = rx[inside]
            valid_y = ry[inside]
            for idx in range(len(valid_x)):
                if len(cap_pts) >= n_cap:
                    break
                cap_pts.append([valid_x[idx], valid_y[idx]])
        cap_arr = np.array(cap_pts[:n_cap])
        zs = rng.choice([-height / 2, height / 2], n_cap)
        pts_list.append(np.column_stack([cap_arr[:, 0], cap_arr[:, 1], zs]))

    return np.vstack(pts_list)


def _sample_sphere(
    rng: np.random.RandomState,
    n: int,
    radius: float,
    centre: np.ndarray | None = None,
) -> np.ndarray:
    """Sample points on the surface of a sphere."""
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    x = radius * sin_theta * np.cos(phi)
    y = radius * sin_theta * np.sin(phi)
    z = radius * cos_theta
    pts = np.column_stack([x, y, z])
    if centre is not None:
        pts += centre
    return pts


def _subsample(
    rng: np.random.RandomState, pts: np.ndarray, n: int
) -> np.ndarray:
    """Sub-sample or pad *pts* to exactly *n* points."""
    m = len(pts)
    if m == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if m == n:
        return pts
    if m > n:
        idx = rng.choice(m, n, replace=False)
        return pts[idx]
    # Pad
    pad_idx = rng.choice(m, n - m, replace=True)
    return np.concatenate([pts, pts[pad_idx]], axis=0)


def _normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    """Centre at origin and scale to unit sphere."""
    pts = pts.copy()
    centroid = pts.mean(axis=0)
    pts -= centroid
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist
    return pts


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class Synthetic3DGenerator:
    """Generate synthetic 3D point clouds for each of the 8 part types."""

    def __init__(self, num_points: int = 2048, seed: int = 42):
        self.num_points = num_points
        self.rng = np.random.RandomState(seed)

    # -- individual generators ------------------------------------------------

    def generate_flange(self) -> np.ndarray:
        """Disk with bolt holes (法兰盘).

        Main body is a flat cylinder.  A centre hole is subtracted and 4-8
        bolt holes are placed around the perimeter.
        """
        outer_r = self.rng.uniform(4.0, 6.0)
        inner_r = self.rng.uniform(1.0, 1.8)
        thickness = self.rng.uniform(0.4, 0.8)
        n_bolts = self.rng.randint(4, 9)  # 4-8
        bolt_r = self.rng.uniform(0.25, 0.4)
        bolt_circle_r = self.rng.uniform(outer_r * 0.55, outer_r * 0.75)

        n_body = int(self.num_points * 0.55)
        n_caps = int(self.num_points * 0.20)
        n_holes = self.num_points - n_body - n_caps

        # Lateral surface of outer cylinder
        body = _sample_cylinder_surface(self.rng, n_body, outer_r, thickness)

        # Top and bottom disks (annular)
        top = _sample_disk(self.rng, n_caps // 2, inner_r, outer_r, thickness / 2)
        bot = _sample_disk(self.rng, n_caps - n_caps // 2, inner_r, outer_r, -thickness / 2)

        # Centre hole lateral surface
        n_centre = n_holes // 2
        centre_hole = _sample_cylinder_surface(self.rng, n_centre, inner_r, thickness)

        # Bolt holes
        n_bolt_total = n_holes - n_centre
        n_per_bolt = max(1, n_bolt_total // n_bolts)
        bolt_pts: list[np.ndarray] = []
        for i in range(n_bolts):
            angle = 2 * np.pi * i / n_bolts
            cx = bolt_circle_r * np.cos(angle)
            cy = bolt_circle_r * np.sin(angle)
            bp = _sample_cylinder_surface(self.rng, n_per_bolt, bolt_r, thickness)
            bp[:, 0] += cx
            bp[:, 1] += cy
            bolt_pts.append(bp)

        all_pts = np.vstack([body, top, bot, centre_hole] + bolt_pts)
        all_pts = _subsample(self.rng, all_pts, self.num_points)
        return _normalize_unit_sphere(all_pts)

    def generate_shaft(self) -> np.ndarray:
        """Elongated cylinder with diameter steps (轴).

        2-3 sections with different diameters, plus an optional keyway slot.
        """
        n_sections = self.rng.randint(2, 4)
        total_length = self.rng.uniform(8.0, 14.0)
        section_lengths = self.rng.dirichlet(np.ones(n_sections)) * total_length
        diameters = sorted(self.rng.uniform(0.8, 2.5, n_sections), reverse=True)
        # Shuffle so steps are not always monotonic
        self.rng.shuffle(diameters)

        pts_list: list[np.ndarray] = []
        z_cursor = -total_length / 2
        pts_per_section = self.num_points // n_sections

        for i in range(n_sections):
            r = diameters[i] / 2
            sec_len = section_lengths[i]
            n_lat = int(pts_per_section * 0.7)
            n_cap = pts_per_section - n_lat

            lat = _sample_cylinder_surface(self.rng, n_lat, r, sec_len)
            lat[:, 2] += z_cursor + sec_len / 2

            # End caps (solid disks)
            cap1 = _sample_disk(self.rng, n_cap // 2, 0.0, r, z_cursor)
            cap2 = _sample_disk(self.rng, n_cap - n_cap // 2, 0.0, r, z_cursor + sec_len)
            pts_list.extend([lat, cap1, cap2])
            z_cursor += sec_len

        # Optional keyway (50 % chance)
        if self.rng.rand() > 0.5:
            kw_width = self.rng.uniform(0.15, 0.3)
            kw_depth = self.rng.uniform(0.1, 0.2)
            kw_len = total_length * self.rng.uniform(0.3, 0.6)
            max_r = max(diameters) / 2
            n_kw = max(1, self.num_points // 10)
            kw = _sample_box_surface(self.rng, n_kw, kw_width, kw_depth, kw_len)
            kw[:, 1] += max_r  # place on top surface
            pts_list.append(kw)

        all_pts = np.vstack(pts_list)
        all_pts = _subsample(self.rng, all_pts, self.num_points)
        return _normalize_unit_sphere(all_pts)

    def generate_shell(self) -> np.ndarray:
        """Hollow box or cylinder (壳体).

        Outer surface + inner surface with wall thickness, plus mounting holes.
        """
        use_box = self.rng.rand() > 0.5

        if use_box:
            lx = self.rng.uniform(4.0, 7.0)
            ly = self.rng.uniform(4.0, 7.0)
            lz = self.rng.uniform(3.0, 6.0)
            wall = self.rng.uniform(0.3, 0.6)

            n_outer = int(self.num_points * 0.55)
            n_inner = int(self.num_points * 0.35)
            n_holes = self.num_points - n_outer - n_inner

            outer = _sample_box_surface(self.rng, n_outer, lx, ly, lz)
            inner = _sample_box_surface(
                self.rng, n_inner,
                lx - 2 * wall, ly - 2 * wall, lz - 2 * wall,
            )

            # Mounting holes on the +z face
            hole_pts: list[np.ndarray] = []
            n_mount = self.rng.randint(2, 5)
            n_per = max(1, n_holes // n_mount)
            for _ in range(n_mount):
                hx = self.rng.uniform(-lx / 3, lx / 3)
                hy = self.rng.uniform(-ly / 3, ly / 3)
                hr = self.rng.uniform(0.2, 0.4)
                hp = _sample_cylinder_surface(self.rng, n_per, hr, wall)
                hp[:, 0] += hx
                hp[:, 1] += hy
                hp[:, 2] += lz / 2 - wall / 2
                hole_pts.append(hp)

            all_pts = np.vstack([outer, inner] + hole_pts)
        else:
            outer_r = self.rng.uniform(3.0, 5.0)
            height = self.rng.uniform(4.0, 8.0)
            wall = self.rng.uniform(0.3, 0.6)
            inner_r = outer_r - wall

            n_outer = int(self.num_points * 0.40)
            n_inner = int(self.num_points * 0.30)
            n_caps = int(self.num_points * 0.20)
            n_holes = self.num_points - n_outer - n_inner - n_caps

            outer = _sample_cylinder_surface(self.rng, n_outer, outer_r, height)
            inner = _sample_cylinder_surface(self.rng, n_inner, inner_r, height)
            top = _sample_disk(self.rng, n_caps // 2, inner_r, outer_r, height / 2)
            bot = _sample_disk(self.rng, n_caps - n_caps // 2, inner_r, outer_r, -height / 2)

            hole_pts = []
            n_mount = self.rng.randint(2, 5)
            n_per = max(1, n_holes // n_mount)
            for i in range(n_mount):
                angle = 2 * np.pi * i / n_mount
                hx = outer_r * np.cos(angle)
                hy = outer_r * np.sin(angle)
                hr = self.rng.uniform(0.15, 0.3)
                # Holes through the wall aligned radially -- approximate as
                # small cylinder segments
                hp = _sample_cylinder_surface(self.rng, n_per, hr, wall)
                hp[:, 0] += hx
                hp[:, 1] += hy
                hole_pts.append(hp)

            all_pts = np.vstack([outer, inner, top, bot] + hole_pts)

        all_pts = _subsample(self.rng, all_pts, self.num_points)
        return _normalize_unit_sphere(all_pts)

    def generate_bracket(self) -> np.ndarray:
        """L-shaped or U-shaped profile (支架).

        A base plate + vertical wall, optional gusset and mounting holes.
        """
        base_lx = self.rng.uniform(3.0, 6.0)
        base_ly = self.rng.uniform(3.0, 6.0)
        base_lz = self.rng.uniform(0.3, 0.6)  # thickness

        wall_height = self.rng.uniform(3.0, 6.0)
        wall_thickness = self.rng.uniform(0.3, 0.6)

        u_shape = self.rng.rand() > 0.5

        n_base = int(self.num_points * 0.30)
        n_wall = int(self.num_points * 0.35)
        n_gusset = int(self.num_points * 0.15)
        n_holes = self.num_points - n_base - n_wall - n_gusset

        # Base plate
        base = _sample_box_surface(self.rng, n_base, base_lx, base_ly, base_lz)

        # Vertical wall(s)
        if u_shape:
            n_each = n_wall // 2
            wall1 = _sample_box_surface(
                self.rng, n_each, wall_thickness, base_ly, wall_height,
            )
            wall1[:, 0] += base_lx / 2 - wall_thickness / 2
            wall1[:, 2] += (wall_height + base_lz) / 2

            wall2 = _sample_box_surface(
                self.rng, n_wall - n_each, wall_thickness, base_ly, wall_height,
            )
            wall2[:, 0] -= base_lx / 2 - wall_thickness / 2
            wall2[:, 2] += (wall_height + base_lz) / 2

            walls = np.vstack([wall1, wall2])
        else:
            # L-shape: single wall at one edge
            walls = _sample_box_surface(
                self.rng, n_wall, wall_thickness, base_ly, wall_height,
            )
            walls[:, 0] += base_lx / 2 - wall_thickness / 2
            walls[:, 2] += (wall_height + base_lz) / 2

        # Gusset (triangular reinforcement approximated as thin wedge)
        gusset_size = min(wall_height, base_lx) * self.rng.uniform(0.3, 0.6)
        gx = self.rng.uniform(0, gusset_size, n_gusset)
        gz = gusset_size - gx  # linear taper
        gy = self.rng.uniform(-0.05, 0.05, n_gusset)  # thin sheet
        gusset = np.column_stack([gx, gy, gz])
        gusset[:, 0] += base_lx / 2 - gusset_size
        gusset[:, 2] += base_lz / 2

        # Mounting holes on the base
        hole_pts: list[np.ndarray] = []
        n_mount = self.rng.randint(2, 5)
        n_per = max(1, n_holes // n_mount)
        for _ in range(n_mount):
            hx = self.rng.uniform(-base_lx / 3, base_lx / 3)
            hy = self.rng.uniform(-base_ly / 3, base_ly / 3)
            hr = self.rng.uniform(0.15, 0.3)
            hp = _sample_cylinder_surface(self.rng, n_per, hr, base_lz)
            hp[:, 0] += hx
            hp[:, 1] += hy
            hole_pts.append(hp)

        all_pts = np.vstack([base, walls, gusset] + hole_pts)
        all_pts = _subsample(self.rng, all_pts, self.num_points)
        return _normalize_unit_sphere(all_pts)

    def generate_gear(self) -> np.ndarray:
        """Cylinder with teeth profile (齿轮).

        Central hub with radial teeth around the perimeter and a centre bore.
        """
        n_teeth = self.rng.randint(12, 25)
        module = self.rng.uniform(0.2, 0.5)
        pitch_r = n_teeth * module / 2
        addendum = module
        dedendum = module * 1.25
        outer_r = pitch_r + addendum
        root_r = pitch_r - dedendum
        bore_r = self.rng.uniform(0.5, 1.2)
        face_w = self.rng.uniform(1.0, 2.5)

        n_hub = int(self.num_points * 0.25)
        n_teeth_pts = int(self.num_points * 0.40)
        n_caps = int(self.num_points * 0.20)
        n_bore = self.num_points - n_hub - n_teeth_pts - n_caps

        # Hub (cylinder from root_r)
        hub = _sample_cylinder_surface(self.rng, n_hub, root_r, face_w)

        # Teeth -- approximate each tooth as a small box protruding radially
        tooth_width_angle = np.pi / n_teeth  # half the angular pitch
        pts_per_tooth = max(1, n_teeth_pts // n_teeth)
        teeth_list: list[np.ndarray] = []
        for i in range(n_teeth):
            angle = 2 * np.pi * i / n_teeth
            # Tooth centre at pitch_r along angle direction
            t_len = outer_r - root_r
            t_width = 2 * pitch_r * np.sin(tooth_width_angle / 2)
            # Generate in local frame then rotate
            tp = _sample_box_surface(self.rng, pts_per_tooth, t_len, t_width, face_w)
            # Shift radially
            tp[:, 0] += root_r + t_len / 2
            # Rotate around Z
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rx = tp[:, 0] * cos_a - tp[:, 1] * sin_a
            ry = tp[:, 0] * sin_a + tp[:, 1] * cos_a
            tp[:, 0] = rx
            tp[:, 1] = ry
            teeth_list.append(tp)

        teeth_all = np.vstack(teeth_list) if teeth_list else np.zeros((0, 3))

        # Caps (annular disks)
        top = _sample_disk(self.rng, n_caps // 2, bore_r, outer_r, face_w / 2)
        bot = _sample_disk(self.rng, n_caps - n_caps // 2, bore_r, outer_r, -face_w / 2)

        # Bore
        bore = _sample_cylinder_surface(self.rng, n_bore, bore_r, face_w)

        all_pts = np.vstack([hub, teeth_all, top, bot, bore])
        all_pts = _subsample(self.rng, all_pts, self.num_points)
        return _normalize_unit_sphere(all_pts)

    def generate_connector(self) -> np.ndarray:
        """Bolt/nut-like shapes (连接件).

        A head (hex prism or cylinder) plus a cylindrical shank with
        a helical thread perturbation.
        """
        hex_head = self.rng.rand() > 0.5
        head_r = self.rng.uniform(1.0, 2.0)
        head_h = self.rng.uniform(0.6, 1.2)
        shank_r = self.rng.uniform(0.4, 0.8)
        shank_len = self.rng.uniform(3.0, 7.0)
        thread_pitch = self.rng.uniform(0.3, 0.6)
        thread_depth = self.rng.uniform(0.03, 0.08)

        n_head = int(self.num_points * 0.35)
        n_shank = self.num_points - n_head

        # Head
        if hex_head:
            head = _sample_hexagonal_prism(self.rng, n_head, head_r, head_h)
        else:
            n_lat = int(n_head * 0.6)
            n_cap = n_head - n_lat
            head = np.vstack([
                _sample_cylinder_surface(self.rng, n_lat, head_r, head_h),
                _sample_disk(self.rng, n_cap // 2, 0.0, head_r, head_h / 2),
                _sample_disk(self.rng, n_cap - n_cap // 2, 0.0, head_r, -head_h / 2),
            ])

        # Position head above the shank
        head[:, 2] += shank_len / 2 + head_h / 2

        # Shank with thread perturbation
        theta = self.rng.uniform(0, 2 * np.pi, n_shank)
        z = self.rng.uniform(-shank_len / 2, shank_len / 2, n_shank)
        # Helical perturbation: radial offset varies sinusoidally along z
        r_perturb = shank_r + thread_depth * np.sin(
            2 * np.pi * z / thread_pitch + theta
        )
        x = r_perturb * np.cos(theta)
        y = r_perturb * np.sin(theta)
        shank = np.column_stack([x, y, z])

        all_pts = np.vstack([head, shank])
        all_pts = _subsample(self.rng, all_pts, self.num_points)
        return _normalize_unit_sphere(all_pts)

    def generate_seal(self) -> np.ndarray:
        """O-ring or gasket (密封件).

        A torus (O-ring) or a flat annular ring (gasket).
        """
        is_oring = self.rng.rand() > 0.4  # 60 % O-ring

        if is_oring:
            major_r = self.rng.uniform(2.0, 4.0)
            minor_r = self.rng.uniform(0.15, 0.5)
            pts = _sample_torus(self.rng, self.num_points, major_r, minor_r)
        else:
            # Flat gasket: annular disk with small thickness
            outer_r = self.rng.uniform(3.0, 5.0)
            inner_r = self.rng.uniform(1.0, outer_r * 0.5)
            thickness = self.rng.uniform(0.1, 0.3)

            n_top = int(self.num_points * 0.30)
            n_bot = int(self.num_points * 0.30)
            n_outer = int(self.num_points * 0.20)
            n_inner = self.num_points - n_top - n_bot - n_outer

            top = _sample_disk(self.rng, n_top, inner_r, outer_r, thickness / 2)
            bot = _sample_disk(self.rng, n_bot, inner_r, outer_r, -thickness / 2)
            outer_cyl = _sample_cylinder_surface(self.rng, n_outer, outer_r, thickness)
            inner_cyl = _sample_cylinder_surface(self.rng, n_inner, inner_r, thickness)

            pts = np.vstack([top, bot, outer_cyl, inner_cyl])

        pts = _subsample(self.rng, pts, self.num_points)
        return _normalize_unit_sphere(pts)

    def generate_other(self) -> np.ndarray:
        """Random complex shapes from combined primitives (其他)."""
        n_prims = self.rng.randint(2, 5)
        pts_list: list[np.ndarray] = []
        n_per = self.num_points // n_prims

        for _ in range(n_prims):
            ptype = self.rng.randint(0, 3)
            if ptype == 0:
                # Sphere
                r = self.rng.uniform(0.5, 2.0)
                c = self.rng.uniform(-2, 2, 3)
                pts_list.append(_sample_sphere(self.rng, n_per, r, c))
            elif ptype == 1:
                # Box
                dims = self.rng.uniform(0.5, 3.0, 3)
                off = self.rng.uniform(-2, 2, 3)
                pts_list.append(_sample_box_surface(self.rng, n_per, *dims, offset=off))
            else:
                # Cylinder
                r = self.rng.uniform(0.3, 1.5)
                h = self.rng.uniform(1.0, 4.0)
                c = _sample_cylinder_surface(self.rng, n_per, r, h)
                c += self.rng.uniform(-2, 2, 3)
                pts_list.append(c)

        all_pts = np.vstack(pts_list)
        all_pts = _subsample(self.rng, all_pts, self.num_points)
        return _normalize_unit_sphere(all_pts)

    # -- dataset generation ---------------------------------------------------

    _GENERATORS = [
        "generate_flange",
        "generate_shaft",
        "generate_shell",
        "generate_bracket",
        "generate_gear",
        "generate_connector",
        "generate_seal",
        "generate_other",
    ]

    def generate_dataset(
        self, samples_per_class: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate a balanced dataset of synthetic point clouds.

        Returns:
            (points_array, labels_array, label_names) where
            - points_array has shape (N, num_points, 3)
            - labels_array has shape (N,) with integer class labels
            - label_names is a list of the 8 class name strings
        """
        all_points: list[np.ndarray] = []
        all_labels: list[int] = []

        for class_idx, gen_name in enumerate(self._GENERATORS):
            gen_fn = getattr(self, gen_name)
            for _ in range(samples_per_class):
                pc = gen_fn()
                all_points.append(pc)
                all_labels.append(class_idx)

        points_array = np.stack(all_points, axis=0)  # (N, num_points, 3)
        labels_array = np.array(all_labels, dtype=np.int64)

        return points_array, labels_array, list(CLASS_NAMES)

    def save_dataset(
        self, output_dir: str, samples_per_class: int = 100
    ) -> Dict[str, object]:
        """Generate, split, and save dataset to disk.

        Creates:
            output_dir/train.npz  (70 %)
            output_dir/val.npz    (15 %)
            output_dir/test.npz   (15 %)
            output_dir/manifest.json

        Returns:
            Dictionary of metadata / statistics.
        """
        points, labels, label_names = self.generate_dataset(samples_per_class)
        n_total = len(labels)

        # Shuffle
        perm = self.rng.permutation(n_total)
        points = points[perm]
        labels = labels[perm]

        # Split indices
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        # n_test = remainder

        train_pts, train_lbl = points[:n_train], labels[:n_train]
        val_pts, val_lbl = points[n_train : n_train + n_val], labels[n_train : n_train + n_val]
        test_pts, test_lbl = points[n_train + n_val :], labels[n_train + n_val :]

        os.makedirs(output_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(output_dir, "train.npz"),
            points=train_pts,
            labels=train_lbl,
        )
        np.savez_compressed(
            os.path.join(output_dir, "val.npz"),
            points=val_pts,
            labels=val_lbl,
        )
        np.savez_compressed(
            os.path.join(output_dir, "test.npz"),
            points=test_pts,
            labels=test_lbl,
        )

        # Class distribution per split
        def _dist(lbl: np.ndarray) -> Dict[str, int]:
            unique, counts = np.unique(lbl, return_counts=True)
            return {label_names[int(u)]: int(c) for u, c in zip(unique, counts)}

        manifest: Dict[str, object] = {
            "total_samples": int(n_total),
            "samples_per_class": int(samples_per_class),
            "num_points": int(self.num_points),
            "num_classes": len(label_names),
            "class_names": label_names,
            "split_sizes": {
                "train": int(len(train_lbl)),
                "val": int(len(val_lbl)),
                "test": int(len(test_lbl)),
            },
            "class_distribution": {
                "train": _dist(train_lbl),
                "val": _dist(val_lbl),
                "test": _dist(test_lbl),
            },
        }

        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Saved %d samples (%d train / %d val / %d test) to %s",
            n_total,
            len(train_lbl),
            len(val_lbl),
            len(test_lbl),
            output_dir,
        )
        return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate synthetic 3D training data for PointNet"
    )
    parser.add_argument(
        "--output",
        default="data/pointnet_synthetic",
        help="Output directory (default: data/pointnet_synthetic)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples per class (default: 100)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=2048,
        help="Points per cloud (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    gen = Synthetic3DGenerator(num_points=args.points, seed=args.seed)
    stats = gen.save_dataset(args.output, samples_per_class=args.samples)
    print(f"Generated {stats['total_samples']} samples in {args.output}")
