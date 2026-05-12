#!/usr/bin/env python3
"""
Interactive Muscle Path Editor for SMPL-H meshes.

Click waypoints on the mesh surface to define muscle bands using geodesic
distance. Saves definitions as JSON for use with generate_muscle_atlas_v4.py.

Controls:
  Left click      Add waypoint (snaps to nearest vertex)
  Z               Undo last waypoint
  +/=             Increase band radius
  -               Decrease band radius
  W               Toggle wireframe overlay
  S               Save muscle definition to JSON
  N               New muscle (clear waypoints)
  Q / Esc         Quit
  Left drag       Orbit camera
  Right drag      Pan camera
  Scroll          Zoom
"""

import sys
import os
import json
import math
import numpy as np
import moderngl
import pyglet
from pyglet.window import key, mouse
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

# Force unbuffered stdout so prints appear in terminal
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ---------------------------------------------------------------------------
# SMPL Model Loading
# ---------------------------------------------------------------------------
def load_smpl_mesh(model_path, gender='male'):
    """Load T-pose SMPL-H mesh."""
    import smplx
    import torch

    model = smplx.create(
        model_path=model_path, model_type='smplh',
        gender=gender.upper(), num_betas=10, ext='pkl'
    )
    model.eval()
    with torch.no_grad():
        output = model()
        verts = output.vertices[0].cpu().numpy().astype(np.float32)
        jpos = output.joints[0, :22].cpu().numpy().astype(np.float32)

    faces = np.array(model.faces, dtype=np.int32)
    return verts, faces, jpos


def compute_normals(verts, faces):
    """Compute per-vertex normals (area-weighted)."""
    normals = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return (normals / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Geodesic distance on mesh
# ---------------------------------------------------------------------------
def build_edge_graph(verts, faces):
    """Build sparse edge distance matrix for geodesic computation."""
    n = len(verts)
    graph = lil_matrix((n, n), dtype=np.float64)
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            d = np.linalg.norm(verts[a] - verts[b])
            if graph[a, b] == 0 or d < graph[a, b]:
                graph[a, b] = d
                graph[b, a] = d
    return graph.tocsr()


def geodesic_band(graph, waypoint_vids, radius, n_verts):
    """Compute geodesic distance from a path of vertices, return weights.

    Computes geodesic distance from each waypoint vertex, then takes
    the minimum distance to any point on the path. Vertices within
    radius get weight 1.0, with Gaussian falloff beyond.
    """
    # Compute geodesic from ALL waypoints at once
    dists = shortest_path(graph, directed=False, indices=waypoint_vids)
    # dists shape: (n_waypoints, n_verts)

    # For path segments between consecutive waypoints, approximate the
    # distance to the path as min distance to any waypoint
    min_dist = dists.min(axis=0)

    # Weights: 1.0 inside radius, Gaussian falloff outside
    weights = np.zeros(n_verts, dtype=np.float32)
    inside = min_dist <= radius
    weights[inside] = 1.0
    outside = ~inside & (min_dist < radius * 4)  # cap at 4x radius
    falloff = radius * 0.4  # falloff sigma
    weights[outside] = np.exp(-0.5 * ((min_dist[outside] - radius) / falloff) ** 2)

    # Normalize to peak = 1
    wmax = weights.max()
    if wmax > 1e-6:
        weights /= wmax

    return weights


# ---------------------------------------------------------------------------
# Orbit Camera
# ---------------------------------------------------------------------------
class OrbitCamera:
    def __init__(self, target=(0, 0, 0), distance=2.0, azimuth=0, elevation=20):
        self.target = np.array(target, dtype=np.float32)
        self.distance = distance
        self.azimuth = azimuth      # degrees
        self.elevation = elevation  # degrees

    def get_view_matrix(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        eye = self.target + self.distance * np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ], dtype=np.float32)
        return self._look_at(eye, self.target, np.array([0, 1, 0], dtype=np.float32))

    def get_eye(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        return self.target + self.distance * np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ], dtype=np.float32)

    @staticmethod
    def _look_at(eye, target, up):
        f = target - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.identity(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return m

    @staticmethod
    def perspective(fov, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov) / 2)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1
        return m

    def orbit(self, dx, dy):
        self.azimuth += dx * 0.3
        self.elevation = max(-89, min(89, self.elevation + dy * 0.3))

    def pan(self, dx, dy):
        az = math.radians(self.azimuth)
        right = np.array([math.cos(az), 0, -math.sin(az)], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        scale = self.distance * 0.002
        self.target -= right * dx * scale
        self.target += up * dy * scale

    def zoom(self, delta):
        self.distance *= 0.9 ** delta
        self.distance = max(0.1, min(20.0, self.distance))


# ---------------------------------------------------------------------------
# Main Editor Window
# ---------------------------------------------------------------------------
class MusclePathEditor(pyglet.window.Window):

    MESH_VERT_SHADER = '''
        #version 330
        uniform mat4 MVP;
        uniform mat4 M;
        in vec3 in_position;
        in vec3 in_normal;
        in float in_weight;
        in float in_ref_weight;
        out vec3 v_normal;
        out vec3 v_pos;
        out float v_weight;
        out float v_ref_weight;
        void main() {
            gl_Position = MVP * vec4(in_position, 1.0);
            v_normal = mat3(M) * in_normal;
            v_pos = (M * vec4(in_position, 1.0)).xyz;
            v_weight = in_weight;
            v_ref_weight = in_ref_weight;
        }
    '''

    MESH_FRAG_SHADER = '''
        #version 330
        uniform vec3 light_dir;
        in vec3 v_normal;
        in vec3 v_pos;
        in float v_weight;
        in float v_ref_weight;
        out vec4 f_color;
        void main() {
            vec3 n = normalize(v_normal);
            vec3 l = normalize(light_dir);
            float diff = max(dot(n, l), 0.0) * 0.6 + 0.3;

            // Base color: purple mesh
            vec3 base = vec3(0.45, 0.2, 0.5);

            // Reference atlas: cyan
            vec3 ref_color = vec3(0.1, 0.6, 0.9);
            // New waypoints: yellow-green
            vec3 new_color = vec3(0.2, 1.0, 0.3);

            // Blend: ref first, then new on top
            vec3 color = mix(base, ref_color, v_ref_weight * 0.5);
            color = mix(color, new_color, v_weight);
            color *= diff;
            f_color = vec4(color, 0.95);
        }
    '''

    WIRE_VERT_SHADER = '''
        #version 330
        uniform mat4 MVP;
        in vec3 in_position;
        void main() {
            gl_Position = MVP * vec4(in_position, 1.0);
            gl_Position.z -= 0.0001;  // depth bias to prevent z-fighting
        }
    '''

    WIRE_FRAG_SHADER = '''
        #version 330
        out vec4 f_color;
        void main() {
            f_color = vec4(0.3, 0.25, 0.35, 0.4);
        }
    '''

    PICK_VERT_SHADER = '''
        #version 330
        uniform mat4 MVP;
        in vec3 in_position;
        flat out int v_face_id;
        void main() {
            gl_Position = MVP * vec4(in_position, 1.0);
            v_face_id = gl_VertexID / 3;
        }
    '''

    PICK_FRAG_SHADER = '''
        #version 330
        flat in int v_face_id;
        out vec4 f_color;
        void main() {
            int id = v_face_id + 1;  // 0 = background
            float r = float((id >> 16) & 0xFF) / 255.0;
            float g = float((id >> 8)  & 0xFF) / 255.0;
            float b = float( id        & 0xFF) / 255.0;
            f_color = vec4(r, g, b, 1.0);
        }
    '''

    POINT_VERT_SHADER = '''
        #version 330
        uniform mat4 MVP;
        in vec3 in_position;
        void main() {
            gl_Position = MVP * vec4(in_position, 1.0);
            gl_PointSize = 12.0;
        }
    '''

    POINT_FRAG_SHADER = '''
        #version 330
        uniform vec4 color;
        out vec4 f_color;
        void main() {
            vec2 c = 2.0 * gl_PointCoord - 1.0;
            if (dot(c, c) > 1.0) discard;
            f_color = color;
        }
    '''

    def __init__(self, model_path, gender='male', width=1280, height=800,
                 muscle_name='NewMuscle', muscle_joint=1, muscle_side=None):
        super().__init__(width=width, height=height, caption='Muscle Path Editor',
                         resizable=True)

        # OpenGL context via pyglet
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # Load mesh
        print("Loading SMPL mesh...")
        self.verts, self.faces, self.jpos = load_smpl_mesh(model_path, gender)
        self.normals = compute_normals(self.verts, self.faces)
        self.n_verts = len(self.verts)
        self.n_faces = len(self.faces)
        print(f"Mesh: {self.n_verts} verts, {self.n_faces} faces")

        # Load existing atlas for reference
        self._load_atlas_ref(model_path, muscle_name)

        # Build edge graph for geodesic distance
        print("Building edge graph...")
        self.edge_graph = build_edge_graph(self.verts, self.faces)
        print("Edge graph ready")

        # Build adjacency list for Laplacian smoothing
        self._neighbors = [[] for _ in range(self.n_verts)]
        for f in self.faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                if b not in self._neighbors[a]:
                    self._neighbors[a].append(b)
                if a not in self._neighbors[b]:
                    self._neighbors[b].append(a)

        # Camera centered on mesh
        center = self.verts.mean(axis=0)
        self.camera = OrbitCamera(target=center, distance=2.5, azimuth=180, elevation=10)

        # Muscle editing state
        self.waypoints = []         # list of vertex indices
        self.band_radius = 0.025    # geodesic radius in meters
        self.muscle_weights = np.zeros(self.n_verts, dtype=np.float32)
        self.muscle_name = muscle_name
        self.muscle_joint = muscle_joint
        self.muscle_x_side = muscle_side
        self.muscle_flex_axis = [0, 0, 0]
        self.saved_muscles = []     # list of saved definitions
        self.show_wireframe = True  # wireframe overlay on by default
        self.show_dots = True       # waypoint dots on by default
        self.click_mode = 'spread'  # 'spread' = geodesic band, 'point' = local per-vertex
        self.paint_weight = 1.0     # current weight for new waypoints
        self._waypoint_cache = {}   # {muscle_name: (waypoints, radius)}
        self._dirty = False         # track if user edited current muscle

        # Mouse state
        self._mouse_buttons = set()
        self._mouse_x = 0
        self._mouse_y = 0

        # Build GPU resources
        self._build_shaders()
        self._build_mesh_buffers()
        self._build_pick_fbo()

        # Schedule draw
        pyglet.clock.schedule_interval(self._tick, 1/60)

        # HUD label
        self._hud_label = pyglet.text.Label(
            '', font_name='Menlo', font_size=12,
            x=10, y=height - 20,
            anchor_x='left', anchor_y='top',
            multiline=True, width=600,
            color=(220, 220, 220, 255),
        )
        self._update_hud()

        # Load existing waypoints/weights for this muscle if available
        self._load_existing_muscle()

        print(f"\nMuscle: {self.muscle_name}, joint={self.muscle_joint}, "
              f"side={self.muscle_x_side}")
        print(f"Controls: click=waypoint, Z=undo, +/-=radius, W=wireframe, "
              f"S=save, N=new, Q=quit")

    def _load_existing_muscle(self):
        """Load waypoints and/or baked weights for current muscle from disk."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'muscle_paths.json')
        baked_path = os.path.join(script_dir, 'baked_weights',
                                  f"{self.muscle_name}.npy")

        # Try loading from JSON
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    all_defs = json.load(f)
                for defn in all_defs:
                    if defn['name'] == self.muscle_name:
                        vids = defn.get('waypoint_vertices', [])
                        wts = defn.get('waypoint_weights', [1.0] * len(vids))
                        self.waypoints = list(zip(vids, wts))
                        self.band_radius = defn.get('radius', self.band_radius)
                        print(f"  Loaded {len(self.waypoints)} waypoints from JSON")
                        break
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  WARNING: muscle_paths.json corrupted: {e}")

        # Try loading baked weights
        if os.path.exists(baked_path):
            baked = np.load(baked_path).astype(np.float32)
            if len(baked) == self.n_verts:
                self.muscle_weights = baked
                self.vbo_weight.write(self.muscle_weights.tobytes())
                print(f"  Loaded baked weights from {baked_path}")
                self._update_hud()
                return

        # If we have waypoints but no baked weights, recompute
        if self.waypoints:
            self._update_band()

    def _load_atlas_ref(self, model_path, muscle_name):
        """Load existing muscle atlas for reference display."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        atlas_path = os.path.join(script_dir, 'muscle_atlas_v4.npy')
        meta_path = os.path.join(script_dir, 'muscle_atlas_v4_meta.npy')
        names_json_path = os.path.join(script_dir, 'muscle_names_v4.json')

        self.ref_atlas = None
        self.ref_muscle_names = []
        self.ref_muscle_idx = 0
        self.ref_weights = np.zeros(self.n_verts, dtype=np.float32)
        self._pinned_refs = set()   # set of pinned muscle indices
        self._coverage_mode = False # show all muscles at once

        if os.path.exists(atlas_path):
            self.ref_atlas = np.load(atlas_path).astype(np.float32)
            print(f"Loaded atlas: {atlas_path}  shape={self.ref_atlas.shape}")

            # Load muscle names from metadata npy or json fallback
            if os.path.exists(meta_path):
                meta = np.load(meta_path, allow_pickle=True).item()
                all_names = meta.get('muscle_names', [])
                # Deduplicate: keep first occurrence of each name
                seen = set()
                keep_idx = []
                for i, n in enumerate(all_names):
                    if n not in seen:
                        seen.add(n)
                        keep_idx.append(i)
                if len(keep_idx) < len(all_names):
                    print(f"  Removed {len(all_names) - len(keep_idx)} duplicate atlas entries")
                    self.ref_atlas = self.ref_atlas[:, keep_idx]
                self.ref_muscle_names = [all_names[i] for i in keep_idx]
            elif os.path.exists(names_json_path):
                with open(names_json_path) as f:
                    self.ref_muscle_names = json.load(f)
            else:
                self.ref_muscle_names = [f"muscle_{i}" for i in range(self.ref_atlas.shape[1])]

            # Select matching muscle if name matches
            if muscle_name in self.ref_muscle_names:
                self.ref_muscle_idx = self.ref_muscle_names.index(muscle_name)
            else:
                for i, n in enumerate(self.ref_muscle_names):
                    if muscle_name.lower() in n.lower():
                        self.ref_muscle_idx = i
                        break

            self._set_ref_weights()

            # Also load joint info for navigation (deduplicated with same keep_idx)
            if os.path.exists(meta_path):
                meta2 = np.load(meta_path, allow_pickle=True).item()
                all_joints = list(meta2.get('muscle_joints', []))
                all_flex = list(meta2.get('flex_axes', []))
                all_sides = list(meta2.get('muscle_sides', ['' for _ in all_joints]))
                self.ref_muscle_joints = [all_joints[i] for i in keep_idx if i < len(all_joints)]
                self.ref_muscle_flex_axes = [all_flex[i] for i in keep_idx if i < len(all_flex)]
                self.ref_muscle_sides = [all_sides[i] for i in keep_idx if i < len(all_sides)]
            else:
                self.ref_muscle_joints = []
                self.ref_muscle_flex_axes = []
                self.ref_muscle_sides = []

            # Append muscles from muscle_paths.json not yet in atlas,
            # and override atlas metadata with JSON values (JSON is authoritative)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mp_path = os.path.join(script_dir, 'muscle_paths.json')
            if os.path.exists(mp_path):
                try:
                    with open(mp_path) as f:
                        mp_defs = json.load(f)
                    for md in mp_defs:
                        if md['name'] in self.ref_muscle_names:
                            # Update existing atlas entry with JSON metadata
                            idx = self.ref_muscle_names.index(md['name'])
                            if idx < len(self.ref_muscle_joints):
                                self.ref_muscle_joints[idx] = md.get('joint', self.ref_muscle_joints[idx])
                            if idx < len(self.ref_muscle_flex_axes):
                                self.ref_muscle_flex_axes[idx] = md.get('flex_axis', self.ref_muscle_flex_axes[idx])
                            if idx < len(self.ref_muscle_sides):
                                self.ref_muscle_sides[idx] = md.get('x_side', self.ref_muscle_sides[idx])
                        else:
                            # Add new muscle from JSON
                            self.ref_muscle_names.append(md['name'])
                            self.ref_muscle_joints.append(md.get('joint', 0))
                            self.ref_muscle_flex_axes.append(md.get('flex_axis', [0,0,0]))
                            self.ref_muscle_sides.append(md.get('x_side', ''))
                            # Extend atlas with zeros for this new column
                            if self.ref_atlas is not None:
                                new_col = np.zeros((self.n_verts, 1), dtype=np.float32)
                                self.ref_atlas = np.hstack([self.ref_atlas, new_col])
                            print(f"  Added from JSON: {md['name']} (joint {md.get('joint',0)})")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  WARNING: muscle_paths.json is corrupted: {e}")
                    print(f"  Editor will continue without JSON muscles.")
        else:
            print(f"No atlas found at {atlas_path} — reference overlay disabled")

    def _get_ref_muscle_weights(self, idx):
        """Get best available weights for atlas muscle idx: baked > atlas."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if idx < len(self.ref_muscle_names):
            name = self.ref_muscle_names[idx]
            baked_path = os.path.join(script_dir, 'baked_weights', f"{name}.npy")
            if os.path.exists(baked_path):
                baked = np.load(baked_path).astype(np.float32)
                if len(baked) == self.n_verts:
                    return baked
        # Fall back to atlas
        if self.ref_atlas is not None and idx < self.ref_atlas.shape[1]:
            return self.ref_atlas[:, idx].copy()
        return np.zeros(self.n_verts, dtype=np.float32)

    def _set_ref_weights(self):
        """Set reference weights from atlas based on current display mode."""
        if self.ref_atlas is None:
            self.ref_weights = np.zeros(self.n_verts, dtype=np.float32)
            return

        if self._coverage_mode:
            # Show max across ALL muscles, using baked weights where available
            n = self.ref_atlas.shape[1]
            combined = np.zeros(self.n_verts, dtype=np.float32)
            for i in range(n):
                w = self._get_ref_muscle_weights(i)
                combined = np.maximum(combined, w)
            self.ref_weights = combined
            n_covered = np.sum(self.ref_weights > 0.01)
            n_uncovered = self.n_verts - n_covered
            print(f"  Coverage: {n_covered} covered, {n_uncovered} gaps")
        elif self._pinned_refs:
            # Show max across pinned muscles + current
            indices = list(self._pinned_refs)
            if self.ref_muscle_idx not in self._pinned_refs:
                indices.append(self.ref_muscle_idx)
            combined = np.zeros(self.n_verts, dtype=np.float32)
            for i in indices:
                combined = np.maximum(combined, self._get_ref_muscle_weights(i))
            self.ref_weights = combined
            names = [self.ref_muscle_names[i] for i in sorted(self._pinned_refs)
                     if i < len(self.ref_muscle_names)]
            print(f"  Pinned refs: {', '.join(names)}")
        else:
            # Single muscle mode
            self.ref_weights = self._get_ref_muscle_weights(self.ref_muscle_idx)
            if self.ref_muscle_idx < len(self.ref_muscle_names):
                name = self.ref_muscle_names[self.ref_muscle_idx]
                n_active = np.sum(self.ref_weights > 0.01)
                print(f"  Ref muscle: {name} ({n_active} verts)")

    def _cycle_ref(self, direction):
        """Cycle through reference atlas muscles."""
        if self.ref_atlas is None or len(self.ref_muscle_names) == 0:
            print("  No atlas loaded")
            return
        if self._coverage_mode:
            print("  (coverage mode — press C to exit)")
            return
        n = self.ref_atlas.shape[1]
        self.ref_muscle_idx = (self.ref_muscle_idx + direction) % n
        self._set_ref_weights()
        self.vbo_ref_weight.write(self.ref_weights.tobytes())
        self._update_hud()

    def _toggle_pin_ref(self):
        """Pin/unpin the current reference muscle."""
        if self.ref_atlas is None:
            return
        idx = self.ref_muscle_idx
        name = self.ref_muscle_names[idx] if idx < len(self.ref_muscle_names) else f"#{idx}"
        if idx in self._pinned_refs:
            self._pinned_refs.discard(idx)
            print(f"  Unpinned: {name}")
        else:
            self._pinned_refs.add(idx)
            print(f"  Pinned: {name} ({len(self._pinned_refs)} pinned)")
        self._set_ref_weights()
        self.vbo_ref_weight.write(self.ref_weights.tobytes())
        self._update_hud()

    def _toggle_coverage(self):
        """Toggle all-muscles coverage display."""
        if self.ref_atlas is None:
            print("  No atlas loaded")
            return
        self._coverage_mode = not self._coverage_mode
        print(f"  Coverage mode: {'ON' if self._coverage_mode else 'OFF'}")
        self._set_ref_weights()
        self.vbo_ref_weight.write(self.ref_weights.tobytes())
        self._update_hud()

    def _build_shaders(self):
        # Main mesh shader
        self.mesh_prog = self.ctx.program(
            vertex_shader=self.MESH_VERT_SHADER,
            fragment_shader=self.MESH_FRAG_SHADER,
        )
        # Pick shader
        self.pick_prog = self.ctx.program(
            vertex_shader=self.PICK_VERT_SHADER,
            fragment_shader=self.PICK_FRAG_SHADER,
        )
        # Point shader for waypoint markers
        self.point_prog = self.ctx.program(
            vertex_shader=self.POINT_VERT_SHADER,
            fragment_shader=self.POINT_FRAG_SHADER,
        )
        # Wireframe shader
        self.wire_prog = self.ctx.program(
            vertex_shader=self.WIRE_VERT_SHADER,
            fragment_shader=self.WIRE_FRAG_SHADER,
        )

    def _build_mesh_buffers(self):
        # Indexed mesh for main rendering
        self.vbo_pos = self.ctx.buffer(self.verts.tobytes())
        self.vbo_norm = self.ctx.buffer(self.normals.tobytes())
        self.vbo_weight = self.ctx.buffer(self.muscle_weights.tobytes())
        self.vbo_ref_weight = self.ctx.buffer(self.ref_weights.tobytes())
        self.ibo = self.ctx.buffer(self.faces.tobytes())

        self.mesh_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(self.vbo_pos, '3f', 'in_position'),
             (self.vbo_norm, '3f', 'in_normal'),
             (self.vbo_weight, 'f', 'in_weight'),
             (self.vbo_ref_weight, 'f', 'in_ref_weight')],
            index_buffer=self.ibo,
        )

        # Expanded (non-indexed) mesh for pick rendering
        # Each face becomes 3 separate vertices so gl_VertexID / 3 = face index
        expanded_verts = self.verts[self.faces.flatten()]
        self.pick_vbo = self.ctx.buffer(expanded_verts.astype(np.float32).tobytes())
        self.pick_vao = self.ctx.vertex_array(
            self.pick_prog,
            [(self.pick_vbo, '3f', 'in_position')],
        )

        # Point VBO for waypoints (updated dynamically)
        self.point_vbo = self.ctx.buffer(reserve=1024 * 12)  # up to 1024 points
        self.point_vao = self.ctx.vertex_array(
            self.point_prog,
            [(self.point_vbo, '3f', 'in_position')],
        )

        # Wireframe edge buffer
        edges = set()
        for f in self.faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                edges.add((min(a, b), max(a, b)))
        edge_indices = np.array(list(edges), dtype=np.int32).flatten()
        self.n_edge_indices = len(edge_indices)
        self.wire_ibo = self.ctx.buffer(edge_indices.tobytes())
        self.wire_vao = self.ctx.vertex_array(
            self.wire_prog,
            [(self.vbo_pos, '3f', 'in_position')],
            index_buffer=self.wire_ibo,
        )

    def _build_pick_fbo(self):
        w, h = self.width, self.height
        self.pick_texture = self.ctx.texture((w, h), 4, dtype='f1')
        self.pick_depth = self.ctx.depth_texture((w, h))
        self.pick_fbo = self.ctx.framebuffer(
            color_attachments=[self.pick_texture],
            depth_attachment=self.pick_depth,
        )

    def _update_pick_fbo_size(self):
        w, h = self.width, self.height
        if self.pick_texture.size != (w, h):
            self.pick_texture.release()
            self.pick_depth.release()
            self.pick_fbo.release()
            self._build_pick_fbo()

    def _get_mvp(self):
        aspect = self.width / max(self.height, 1)
        proj = OrbitCamera.perspective(45, aspect, 0.01, 50.0)
        view = self.camera.get_view_matrix()
        model = np.identity(4, dtype=np.float32)
        mvp = (proj @ view @ model)
        return mvp, model

    def _tick(self, dt):
        pass  # drawing happens in on_draw

    def on_draw(self):
        self.clear()
        mvp, model = self._get_mvp()

        # Main mesh
        self.ctx.clear(0.12, 0.10, 0.14, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.mesh_prog['MVP'].write(mvp.T.astype(np.float32).tobytes())
        self.mesh_prog['M'].write(model.T.astype(np.float32).tobytes())
        self.mesh_prog['light_dir'].value = (0.5, 1.0, 0.8)
        self.mesh_vao.render(moderngl.TRIANGLES)

        # Wireframe overlay
        if self.show_wireframe:
            self.wire_prog['MVP'].write(mvp.T.astype(np.float32).tobytes())
            self.wire_vao.render(moderngl.LINES)

        # Waypoint markers
        if self.waypoints and self.show_dots:
            vids = [wp[0] for wp in self.waypoints]
            pts = self.verts[vids].astype(np.float32)
            self.point_vbo.write(pts.tobytes())
            self.point_prog['MVP'].write(mvp.T.astype(np.float32).tobytes())
            self.point_prog['color'].value = (1.0, 0.3, 0.1, 1.0)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.point_vao.render(moderngl.POINTS, vertices=len(self.waypoints))
            self.ctx.enable(moderngl.DEPTH_TEST)

        # HUD text overlay
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._hud_label.draw()
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_pick(self):
        """Render face IDs to pick FBO."""
        self._update_pick_fbo_size()
        mvp, _ = self._get_mvp()

        self.pick_fbo.use()
        self.ctx.clear(0, 0, 0, 0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        self.pick_prog['MVP'].write(mvp.T.astype(np.float32).tobytes())
        self.pick_vao.render(moderngl.TRIANGLES)

        # Restore default framebuffer
        self.ctx.screen.use()

    def _pick_vertex(self, x, y):
        """Pick the vertex nearest to click position."""
        self._render_pick()

        # Read pixel (y is flipped in OpenGL)
        pixel = self.pick_fbo.read(viewport=(x, y, 1, 1), components=4)
        r, g, b, a = pixel[0], pixel[1], pixel[2], pixel[3]

        face_id = ((r << 16) | (g << 8) | b) - 1  # -1 because we added 1
        if face_id < 0 or face_id >= self.n_faces:
            return None

        # Find nearest vertex on this face
        face_verts = self.faces[face_id]
        # Use screen-space distance to pick closest vertex
        mvp, _ = self._get_mvp()
        best_vid = None
        best_dist = float('inf')
        for vid in face_verts:
            p = self.verts[vid]
            clip = mvp @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
            if clip[3] != 0:
                ndc = clip[:2] / clip[3]
                sx = (ndc[0] * 0.5 + 0.5) * self.width
                sy = (ndc[1] * 0.5 + 0.5) * self.height
                d = (sx - x) ** 2 + (sy - y) ** 2
                if d < best_dist:
                    best_dist = d
                    best_vid = int(vid)

        return best_vid

    def _shift_waypoints(self, dx, dy, extend=False):
        """Shift all waypoints one vertex in screen-space direction.
        If extend=True, add shifted waypoints alongside originals."""
        # Get camera right and up vectors in world space
        eye = self.camera.get_eye()
        fwd = self.camera.target - eye
        fwd = fwd / np.linalg.norm(fwd)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # World-space shift direction
        shift_dir = dx * right + dy * cam_up
        shift_dir = shift_dir / np.linalg.norm(shift_dir)

        # For each waypoint, find the neighbor most aligned with shift_dir
        new_wps = []
        for vid, w in self.waypoints:
            best_vid = vid
            best_dot = -999
            for nbr in self._neighbors[vid]:
                delta = self.verts[nbr] - self.verts[vid]
                d = np.dot(delta, shift_dir)
                if d > best_dot:
                    best_dot = d
                    best_vid = nbr
            new_wps.append((best_vid, w))

        if extend:
            # Add new positions, avoiding duplicates
            existing_vids = {v for v, _ in self.waypoints}
            for vid, w in new_wps:
                if vid not in existing_vids:
                    self.waypoints.append((vid, w))
        else:
            self.waypoints = new_wps
        self._dirty = True
        self._update_band()

    def _update_band(self):
        """Recompute muscle weights from current waypoints."""
        self._dirty = True
        if len(self.waypoints) == 0:
            self.muscle_weights[:] = 0
        elif self.click_mode == 'spread':
            # Extract just vertex IDs for geodesic computation
            vids = [wp[0] for wp in self.waypoints]
            weights_per_wp = [wp[1] for wp in self.waypoints]
            raw = geodesic_band(
                self.edge_graph, vids,
                self.band_radius, self.n_verts
            )
            # Scale by per-waypoint weights: for each vertex, find which
            # waypoint is closest and use that waypoint's weight
            if len(vids) > 1:
                dists = shortest_path(self.edge_graph, directed=False, indices=vids)
                nearest_wp = dists.argmin(axis=0)
                wp_weight_arr = np.array(weights_per_wp, dtype=np.float32)
                scale = wp_weight_arr[nearest_wp]
                self.muscle_weights = raw * scale
            else:
                self.muscle_weights = raw * weights_per_wp[0]
        else:  # 'point' mode
            self.muscle_weights = self._point_mode_weights()
        # Upload to GPU
        self.vbo_weight.write(self.muscle_weights.tobytes())
        n_active = np.sum(self.muscle_weights > 0.01)
        print(f"  Band: {n_active} active vertices, radius={self.band_radius:.4f}m, "
              f"{len(self.waypoints)} waypoints")
        self._update_hud()

    def _soften_weights(self):
        """One pass of gentle Laplacian smoothing on muscle_weights."""
        peak_before = self.muscle_weights.max()
        new_w = self.muscle_weights.copy()
        blend = 0.15  # gentle blend
        for vid in range(self.n_verts):
            nbrs = self._neighbors[vid]
            if nbrs:
                avg = np.mean(self.muscle_weights[nbrs])
                new_w[vid] = (1.0 - blend) * self.muscle_weights[vid] + blend * avg
        # Normalize to preserve peak
        peak_after = new_w.max()
        if peak_after > 1e-6 and peak_before > 1e-6:
            new_w *= peak_before / peak_after
        self.muscle_weights = new_w
        self.vbo_weight.write(self.muscle_weights.tobytes())
        self._dirty = True
        n_active = np.sum(self.muscle_weights > 0.01)
        print(f"  Softened baked weights: {n_active} active vertices")
        self._update_hud()

    def _soften_waypoint_weights(self):
        """Smooth waypoint weight values by averaging with K nearest neighbors."""
        if len(self.waypoints) < 3:
            print("  Need at least 3 waypoints to soften")
            return
        K = min(4, len(self.waypoints) - 1)  # 4 nearest neighbors
        blend = 0.3  # how much to blend toward neighbors
        vids = [wp[0] for wp in self.waypoints]
        wts = np.array([wp[1] for wp in self.waypoints], dtype=np.float32)

        # Compute pairwise distances using vertex positions
        positions = self.verts[vids]  # (N, 3)
        new_wts = wts.copy()
        for i in range(len(vids)):
            dists = np.linalg.norm(positions - positions[i], axis=1)
            dists[i] = np.inf  # exclude self
            nearest = np.argsort(dists)[:K]
            avg_w = np.mean(wts[nearest])
            new_wts[i] = (1.0 - blend) * wts[i] + blend * avg_w

        # Update waypoints with smoothed weights
        self.waypoints = [(vids[i], float(new_wts[i])) for i in range(len(vids))]
        self._dirty = True
        self._update_band()
        print(f"  Softened {len(self.waypoints)} waypoint weights (K={K}, blend={blend})")

    def _sharpen_weights(self):
        """One pass of inverse Laplacian — accentuates peaks."""
        peak_before = self.muscle_weights.max()
        new_w = self.muscle_weights.copy()
        strength = 0.15
        for vid in range(self.n_verts):
            nbrs = self._neighbors[vid]
            if nbrs:
                avg = np.mean(self.muscle_weights[nbrs])
                # Push away from neighbor average
                new_w[vid] = self.muscle_weights[vid] + strength * (self.muscle_weights[vid] - avg)
        new_w = np.clip(new_w, 0.0, None)
        # Normalize to preserve peak
        peak_after = new_w.max()
        if peak_after > 1e-6 and peak_before > 1e-6:
            new_w *= peak_before / peak_after
        self.muscle_weights = new_w.astype(np.float32)
        self.vbo_weight.write(self.muscle_weights.tobytes())
        n_active = np.sum(self.muscle_weights > 0.01)
        print(f"  Sharpened: {n_active} active vertices")
        self._update_hud()

    def _point_mode_weights(self):
        """Each waypoint gets a small local blob using Euclidean distance."""
        weights = np.zeros(self.n_verts, dtype=np.float32)
        r = self.band_radius * 0.3  # much tighter than spread mode
        falloff = r * 0.5
        for vid, w in self.waypoints:
            d = np.linalg.norm(self.verts - self.verts[vid], axis=1)
            mask = d <= r
            weights[mask] = np.maximum(weights[mask], w)
            fall_mask = (~mask) & (d < r * 3)
            weights[fall_mask] = np.maximum(
                weights[fall_mask],
                w * np.exp(-0.5 * ((d[fall_mask] - r) / falloff) ** 2)
            )
        return weights

    def _update_hud(self):
        n_active = np.sum(self.muscle_weights > 0.01)
        side_str = self.muscle_x_side or 'none'
        mode_str = self.click_mode.upper()

        # Build ref status string
        if self._coverage_mode:
            ref_str = 'ALL COVERAGE'
        elif self._pinned_refs:
            pin_names = [self.ref_muscle_names[i] for i in sorted(self._pinned_refs)
                         if i < len(self.ref_muscle_names)]
            cur_name = self.ref_muscle_names[self.ref_muscle_idx] if self.ref_muscle_names else '?'
            ref_str = f"{cur_name} + {len(pin_names)} pinned"
        else:
            ref_str = self.ref_muscle_names[self.ref_muscle_idx] if self.ref_muscle_names else 'none'

        weight_pct = int(self.paint_weight * 100)
        lines = [
            f"Muscle: {self.muscle_name}  |  Joint: {self.muscle_joint}  |  Side: {side_str}",
            f"Ref: {ref_str} (cyan)  |  [/]=cycle  P=pin  C=coverage  Tab=next",
            f"Radius: {self.band_radius:.4f}m  |  Waypoints: {len(self.waypoints)}  |  Verts: {n_active}",
            f"Mode: {mode_str}  |  Wt: {weight_pct}% (1-9)  |  F=soften D=sharpen  +/-=radius",
            f"M=mode  W=wire  V=dots  R=mirror  S=save  N=clear  Z=undo  Q=quit",
        ]
        self._hud_label.text = '\n'.join(lines)
        self._hud_label.y = self.height - 10

    # --- Event Handlers ---

    def on_mouse_press(self, x, y, button, modifiers):
        self._mouse_buttons.add(button)
        self._mouse_x = x
        self._mouse_y = y
        self._drag_total = 0  # track total drag distance

    def on_mouse_release(self, x, y, button, modifiers):
        self._mouse_buttons.discard(button)
        # Only pick on left-click if we didn't drag (orbit)
        if button == mouse.LEFT and self._drag_total < 5:
            vid = self._pick_vertex(x, y)
            if vid is not None:
                # Check if this vid already exists
                existing = [i for i, (v, w) in enumerate(self.waypoints) if v == vid]
                if existing:
                    # Toggle off
                    self.waypoints.pop(existing[0])
                    pos = self.verts[vid]
                    print(f"  Removed waypoint: vertex {vid}")
                else:
                    self.waypoints.append((vid, self.paint_weight))
                    pos = self.verts[vid]
                    print(f"  Waypoint {len(self.waypoints)}: vertex {vid} "
                          f"weight={self.paint_weight:.0%}")
                self._update_band()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self._drag_total += abs(dx) + abs(dy)
        if buttons & mouse.LEFT:
            self.camera.orbit(dx, -dy)
        if buttons & mouse.RIGHT:
            self.camera.pan(dx, dy)
        if buttons & mouse.MIDDLE:
            self.camera.zoom(dy * 0.1)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera.zoom(scroll_y)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.Q or symbol == key.ESCAPE:
            self.close()

        elif symbol in (key._1, key._2, key._3, key._4, key._5,
                        key._6, key._7, key._8, key._9):
            level = symbol - key._0  # 1..9
            self.paint_weight = level / 9.0
            print(f"  Paint weight: {level}/9 ({self.paint_weight:.1%})")
            self._update_hud()

        elif symbol == key.Z:
            if self.waypoints:
                removed = self.waypoints.pop()
                print(f"  Undo waypoint: vertex {removed}")
                self._update_band()

        elif symbol == key.EQUAL or symbol == key.PLUS:
            self.band_radius *= 1.2
            print(f"  Radius: {self.band_radius:.4f}m")
            if self.waypoints:
                self._update_band()
            else:
                self._update_hud()

        elif symbol == key.MINUS:
            self.band_radius /= 1.2
            self.band_radius = max(0.005, self.band_radius)
            print(f"  Radius: {self.band_radius:.4f}m")
            if self.waypoints:
                self._update_band()
            else:
                self._update_hud()

        elif symbol == key.W:
            self.show_wireframe = not self.show_wireframe
            print(f"  Wireframe: {'ON' if self.show_wireframe else 'OFF'}")

        elif symbol == key.V:
            self.show_dots = not self.show_dots
            print(f"  Dots: {'ON' if self.show_dots else 'OFF'}")

        elif symbol == key.P:
            self._toggle_pin_ref()

        elif symbol == key.C:
            self._toggle_coverage()

        elif symbol in (key.LEFT, key.RIGHT, key.UP, key.DOWN):
            if self.waypoints:
                dx = 1 if symbol == key.RIGHT else (-1 if symbol == key.LEFT else 0)
                dy = 1 if symbol == key.UP else (-1 if symbol == key.DOWN else 0)
                extend = bool(modifiers & key.MOD_SHIFT)
                self._shift_waypoints(dx, dy, extend=extend)
            else:
                print("  No waypoints to shift")

        elif symbol == key.F:
            if modifiers & key.MOD_SHIFT:
                # Shift+F: soften waypoint weights
                if self.waypoints:
                    self._soften_waypoint_weights()
                else:
                    print("  No waypoints to soften")
            elif np.any(self.muscle_weights > 0):
                self._soften_weights()
            else:
                print("  No weights to soften")

        elif symbol == key.D:
            if np.any(self.muscle_weights > 0):
                self._sharpen_weights()
            else:
                print("  No weights to sharpen")

        elif symbol == key.BRACKETLEFT:
            self._cycle_ref(-1)

        elif symbol == key.BRACKETRIGHT:
            self._cycle_ref(1)

        elif symbol == key.M:
            self.click_mode = 'point' if self.click_mode == 'spread' else 'spread'
            print(f"  Mode: {self.click_mode.upper()}")
            if self.waypoints:
                self._update_band()
            else:
                self._update_hud()

        elif symbol == key.R:
            self._mirror_muscle()

        elif symbol == key.S:
            self._save_muscle()

        elif symbol == key.N:
            self._new_muscle()

        elif symbol == key.TAB:
            if modifiers & key.MOD_SHIFT:
                self._goto_muscle(-1)
            else:
                self._goto_muscle(1)

    def _mirror_muscle(self):
        """Mirror current waypoints and weights to the opposite side."""
        if not self.waypoints and not np.any(self.muscle_weights > 0):
            print("  No waypoints or weights to mirror!")
            return

        # Build vertex mirror map: for each vertex, find its X-mirrored partner
        if not hasattr(self, '_mirror_map'):
            print("  Building mirror map...")
            mirrored_pos = self.verts.copy()
            mirrored_pos[:, 0] = -mirrored_pos[:, 0]
            # For each vertex, find nearest in original mesh
            from scipy.spatial import cKDTree
            tree = cKDTree(self.verts)
            _, self._mirror_map = tree.query(mirrored_pos)
            print("  Mirror map ready")

        # Mirror waypoints
        mirrored_wps = []
        for vid, w in self.waypoints:
            mirror_vid = int(self._mirror_map[vid])
            mirrored_wps.append((mirror_vid, w))

        # Mirror baked weights
        mirrored_weights = self.muscle_weights[self._mirror_map]

        # Swap name L↔R
        old_name = self.muscle_name
        if old_name.startswith('L_'):
            new_name = 'R_' + old_name[2:]
        elif old_name.startswith('R_'):
            new_name = 'L_' + old_name[2:]
        else:
            new_name = old_name + '_mirror'

        # Swap side
        if self.muscle_x_side == 'L':
            new_side = 'R'
        elif self.muscle_x_side == 'R':
            new_side = 'L'
        else:
            new_side = self.muscle_x_side

        # Swap joint
        joint_mirror = {1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7,
                        16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20,
                        13: 14, 14: 13}
        new_joint = joint_mirror.get(self.muscle_joint, self.muscle_joint)

        # Save mirrored definition to JSON
        defn = {
            'name': new_name,
            'joint': int(new_joint),
            'waypoint_vertices': [int(wp[0]) for wp in mirrored_wps],
            'waypoint_weights': [float(wp[1]) for wp in mirrored_wps],
            'radius': float(self.band_radius),
            'flex_axis': [float(x) for x in self.muscle_flex_axis],
            'has_baked_weights': True,
        }
        if new_side:
            defn['x_side'] = new_side

        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'muscle_paths.json')
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    all_defs = json.load(f)
            except json.JSONDecodeError:
                # Try auto-recovery: truncate to last valid entry
                with open(json_path) as f:
                    raw = f.read()
                last_brace = raw.rfind('}')
                if last_brace > 0:
                    try:
                        all_defs = json.loads(raw[:last_brace+1] + '\n]')
                        print(f"  WARNING: auto-recovered {len(all_defs)} entries from corrupted JSON")
                    except json.JSONDecodeError:
                        print(f"  ERROR: muscle_paths.json is corrupted and unrecoverable. Save aborted.")
                        return
                else:
                    print(f"  ERROR: muscle_paths.json is corrupted. Save aborted.")
                    return
        else:
            all_defs = []
        names = [d['name'] for d in all_defs]
        if defn['name'] in names:
            all_defs[names.index(defn['name'])] = defn
        else:
            all_defs.append(defn)
        tmp_path = json_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(all_defs, f, indent=2)
        os.replace(tmp_path, json_path)

        # Save mirrored baked weights
        baked_dir = os.path.join(script_dir, 'baked_weights')
        os.makedirs(baked_dir, exist_ok=True)
        baked_path = os.path.join(baked_dir, f"{new_name}.npy")
        np.save(baked_path, mirrored_weights.astype(np.float32))

        n_active = np.sum(mirrored_weights > 0.01)
        print(f"\n  *** Mirrored '{old_name}' → '{new_name}': "
              f"{len(mirrored_wps)} waypoints, {n_active} verts, "
              f"joint={new_joint}, side={new_side} ***")
        print(f"  JSON: {json_path}")
        print(f"  Baked: {baked_path}")

    def _save_muscle(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'muscle_paths.json')
        baked_dir = os.path.join(script_dir, 'baked_weights')
        baked_path = os.path.join(baked_dir, f"{self.muscle_name}.npy")

        # Load existing JSON
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    all_defs = json.load(f)
            except json.JSONDecodeError:
                # Try auto-recovery: truncate to last valid entry
                with open(json_path) as f:
                    raw = f.read()
                last_brace = raw.rfind('}')
                if last_brace > 0:
                    try:
                        all_defs = json.loads(raw[:last_brace+1] + '\n]')
                        print(f"  WARNING: auto-recovered {len(all_defs)} entries from corrupted JSON")
                    except json.JSONDecodeError:
                        print(f"  ERROR: muscle_paths.json is corrupted and unrecoverable. Save aborted.")
                        return
                else:
                    print(f"  ERROR: muscle_paths.json is corrupted. Save aborted.")
                    return
        else:
            all_defs = []

        if not self.waypoints and not np.any(self.muscle_weights > 0):
            # Clear: remove from JSON and delete baked file
            all_defs = [d for d in all_defs if d['name'] != self.muscle_name]
            tmp_path = json_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(all_defs, f, indent=2)
            os.replace(tmp_path, json_path)
            if os.path.exists(baked_path):
                os.remove(baked_path)
            # Also clear from in-memory cache
            self._waypoint_cache.pop(self.muscle_name, None)
            print(f"\n  *** Cleared '{self.muscle_name}': removed from JSON and baked ***")
            return

        defn = {
            'name': self.muscle_name,
            'joint': int(self.muscle_joint),
            'waypoint_vertices': [int(wp[0]) for wp in self.waypoints],
            'waypoint_weights': [float(wp[1]) for wp in self.waypoints],
            'radius': float(self.band_radius),
            'flex_axis': [float(x) for x in self.muscle_flex_axis],
            'has_baked_weights': True,
        }
        if self.muscle_x_side:
            defn['x_side'] = self.muscle_x_side

        # Replace if same name exists, otherwise append
        names = [d['name'] for d in all_defs]
        if defn['name'] in names:
            idx = names.index(defn['name'])
            all_defs[idx] = defn
        else:
            all_defs.append(defn)

        # Atomic write: write to tmp then rename
        tmp_path = json_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(all_defs, f, indent=2)
        os.replace(tmp_path, json_path)

        # Save baked weights as .npy
        os.makedirs(baked_dir, exist_ok=True)
        np.save(baked_path, self.muscle_weights)

        n_active = np.sum(self.muscle_weights > 0.01)
        print(f"\n  *** Saved '{defn['name']}': {len(self.waypoints)} waypoints, "
              f"radius={self.band_radius:.4f}m, {n_active} vertices ***")
        self._dirty = False

    def _new_muscle(self):
        """Clear waypoints for a fresh start."""
        self.waypoints = []
        self.muscle_weights[:] = 0
        self.vbo_weight.write(self.muscle_weights.tobytes())
        self._waypoint_cache.pop(self.muscle_name, None)
        self._dirty = True
        print(f"  Cleared waypoints. Use Tab to navigate muscles.")
        self._update_hud()

    def _goto_muscle(self, direction):
        """Navigate to next/prev muscle from atlas, auto-setting name/joint/side."""
        if not self.ref_muscle_names:
            print("  No atlas loaded for navigation")
            return

        # Auto-save current muscle before switching (only if edited)
        if self._dirty:
            self._save_muscle()

        # Cache waypoints in memory too
        if self.waypoints:
            self._waypoint_cache[self.muscle_name] = (self.waypoints[:], self.band_radius)

        # Advance ref index
        n = len(self.ref_muscle_names)
        self.ref_muscle_idx = (self.ref_muscle_idx + direction) % n

        # Set muscle identity from atlas metadata
        name = self.ref_muscle_names[self.ref_muscle_idx]
        self.muscle_name = name

        # Set side from metadata or infer from name
        idx = self.ref_muscle_idx
        if idx < len(self.ref_muscle_sides) and self.ref_muscle_sides[idx]:
            self.muscle_x_side = self.ref_muscle_sides[idx]
        elif name.startswith('L_'):
            self.muscle_x_side = 'L'
        elif name.startswith('R_'):
            self.muscle_x_side = 'R'
        else:
            self.muscle_x_side = None

        # Set joint from metadata
        if idx < len(self.ref_muscle_joints):
            self.muscle_joint = int(self.ref_muscle_joints[idx])

        # Set flex axis from metadata
        if idx < len(self.ref_muscle_flex_axes):
            self.muscle_flex_axis = list(self.ref_muscle_flex_axes[idx])

        # Update reference overlay
        self._set_ref_weights()
        self.vbo_ref_weight.write(self.ref_weights.tobytes())

        # Restore cached waypoints, or try loading from disk
        if name in self._waypoint_cache:
            self.waypoints, self.band_radius = self._waypoint_cache[name]
            self._update_band()
        else:
            self.waypoints = []
            self.muscle_weights[:] = 0
            self.vbo_weight.write(self.muscle_weights.tobytes())
            self._load_existing_muscle()

        # Load baked weights if available — these may include softening/sharpening
        # that _update_band would overwrite
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baked_path = os.path.join(script_dir, 'baked_weights', f"{name}.npy")
        if os.path.exists(baked_path):
            baked = np.load(baked_path).astype(np.float32)
            if len(baked) == self.n_verts:
                self.muscle_weights = baked
                self.vbo_weight.write(self.muscle_weights.tobytes())

        print(f"  → {name}  joint={self.muscle_joint}  side={self.muscle_x_side}")
        self._dirty = False  # freshly loaded, not edited yet
        self._update_hud()

    def on_resize(self, width, height):
        self.ctx.viewport = (0, 0, width, height)
        return pyglet.event.EVENT_HANDLED


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive Muscle Path Editor')
    parser.add_argument('--model_path', type=str, default='.',
                        help='Path to SMPL model directory')
    parser.add_argument('--gender', type=str, default='male',
                        choices=['male', 'female'])
    parser.add_argument('--name', type=str, default='NewMuscle',
                        help='Muscle name (e.g. L_Sartorius)')
    parser.add_argument('--joint', type=int, default=1,
                        help='Joint index (e.g. 1=L_Hip, 2=R_Hip)')
    parser.add_argument('--side', type=str, default=None,
                        choices=['L', 'R'], help='Body side filter')
    args = parser.parse_args()

    editor = MusclePathEditor(  # noqa: F841
        args.model_path, args.gender,
        muscle_name=args.name,
        muscle_joint=args.joint,
        muscle_side=args.side,
    )
    pyglet.app.run()


if __name__ == '__main__':
    main()
