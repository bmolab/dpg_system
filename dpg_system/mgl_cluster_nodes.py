"""Drawing the clusters a grid node has grouped its voxels into.

``mgl_cluster_boxes`` takes the cloud frame pc_voxel sends when it is
subdividing (the 'clusters' entry described in point_cloud_nodes) and draws one
translucent cube per box, coloured by the sum of the voxel weights that landed
in it. It is the numpy/moderngl counterpart of cVoxelMap's boxMesh in the C++
AzureKinectVoxelsApp.

The node reads the cluster frame rather than boxes specifically, so the same
node will draw whatever else comes to produce that frame — connected
components, k-means, hand-painted regions — as long as the clusters form a
regular lattice ('shape' is set). It does no summing of its own: pc_voxel has
already done that with a bincount over the voxel grid, and re-deriving it here
would mean shipping every voxel's label to the renderer each frame.
"""

import numpy as np
import moderngl

from dpg_system.node import Node
from dpg_system.moderngl_nodes import MGLNode
from dpg_system.moderngl_base import MGLContext
from dpg_system.point_cloud_nodes import CLUSTER_KEY


def register_mgl_cluster_nodes():
    Node.app.register_node('mgl_cluster_boxes', MGLClusterBoxesNode.factory)


def hsv_to_rgb(h, s, v):
    """Vectorised HSV -> RGB for (N,) arrays in 0..1, returning (N, 3) float32.

    colorsys does this one colour at a time; a 20x8x20 lattice is 3,200 of them
    every frame, which is exactly the per-element python the rest of this
    pipeline is written to avoid."""
    h = np.asarray(h, dtype=np.float32).reshape(-1) % 1.0
    s = np.clip(np.asarray(s, dtype=np.float32).reshape(-1), 0.0, 1.0)
    v = np.clip(np.asarray(v, dtype=np.float32).reshape(-1), 0.0, 1.0)
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    sector = i.astype(np.int64) % 6
    # Each sector picks a different (r, g, b) permutation of v/p/q/t.
    r = np.choose(sector, [v, q, p, p, t, v])
    g = np.choose(sector, [t, v, v, q, p, p])
    b = np.choose(sector, [p, p, t, v, v, q])
    return np.stack((r, g, b), axis=1).astype(np.float32)


# A unit cube centred on the origin, spanning -0.5..0.5, as 8 corners and the
# 12 triangles over them; the instance shader scales it to each box. Every
# triangle is wound counter-clockwise seen from outside, so back-face culling
# leaves exactly one face per cube over any pixel — without it a closed cube
# draws front and back, and under additive blending every box comes out twice
# as bright as its value.
_CUBE_CORNERS = np.array([[(i >> a) & 1 for a in range(3)] for i in range(8)],
                         dtype=np.float32) - 0.5
_CUBE_INDICES = np.array([
    0, 2, 1,  1, 2, 3,      # -z
    4, 5, 6,  5, 7, 6,      # +z
    0, 1, 4,  1, 5, 4,      # -y
    2, 6, 3,  3, 6, 7,      # +y
    0, 4, 2,  2, 4, 6,      # -x
    1, 3, 5,  3, 7, 5,      # +x
], dtype=np.int32)

# The same eight corners as twelve edges, for the frame pass.
_CUBE_EDGE_INDICES = np.array([(i, i ^ bit) for bit in (1, 2, 4)
                               for i in range(8) if not (i & bit)],
                              dtype=np.int32).reshape(-1)

# 'sensitivity' 1.0 means this much gain on the raw sum. The sums are counts of
# distance-compensated voxel weights, so their natural scale is in the tens or
# hundreds and a raw gain of 1 buries every box at full brightness. This puts a
# usable setting at 1.0 on the widget — it is the value arrived at by hand
# before the scale existed.
SENSITIVITY_SCALE = 0.015


class MGLClusterBoxesNode(MGLNode):
    """One translucent cube per cluster, coloured by that cluster's value.

    Colour is HSV. The value drives V, so a box's brightness is how much landed
    in it and an empty box disappears; 'sensitivity' is the gain on that, the
    equivalent of the C++ boxDisplayGain. 'color mode' decides where H comes
    from: 'uniform' gives every box the same hue, so the picture reads purely
    as intensity; 'per box' walks the hue by 'hue spread' per box index, the
    way cVoxelMap::SetRegionColors does, so neighbouring boxes stay
    distinguishable when you are checking which box is which.

    Blending defaults to additive because these cubes are translucent and
    unsorted: alpha blending would make the picture depend on the order the
    boxes happen to be drawn in, while additive is order-independent. Depth
    writes are off during the pass for the same reason (the test stays on, so
    solid geometry still occludes) — the pattern mgl_point_cloud already uses
    for its additive points."""

    _vert_src = '''
        #version 330
        uniform mat4 M;
        uniform mat4 V;
        uniform mat4 P;
        uniform bool frame_pass;
        uniform float alpha;
        in vec3 in_position;
        in vec3 inst_centre;
        in vec3 inst_size;
        in vec3 inst_rgb;        // the hue/saturation at full value
        in vec2 inst_levels;     // (fill, frame) intensity 0..1
        out vec4 v_color;
        void main() {
            vec3 world = inst_centre + in_position * inst_size;
            gl_Position = P * V * M * vec4(world, 1.0);
            // hsv_to_rgb is linear in v, so scaling the full-value colour here
            // is the same colour the CPU would have produced at this level —
            // and lets one instance buffer serve both passes.
            float level = frame_pass ? inst_levels.y : inst_levels.x;
            v_color = vec4(inst_rgb * level, level * alpha);
        }
    '''
    _frag_src = '''
        #version 330
        in vec4 v_color;
        out vec4 f_color;
        void main() {
            f_color = vec4(v_color.rgb * v_color.a, v_color.a);
        }
    '''

    @staticmethod
    def factory(name, data, args=None):
        return MGLClusterBoxesNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.cloud_input = self.add_input('voxel cloud', triggers_execution=True)
        self.sensitivity_input = self.add_input('sensitivity', widget_type='drag_float',
                                                default_value=1.0, min=0.0)
        self.sensitivity_input.widget.speed = 0.01
        # The frame is driven off the same sum at its own, higher gain, which
        # is what buys the visual dynamic range: the outline saturates while
        # the fill is still coming up off zero, so the bottom of the range is
        # legible and the fill carries the top. The C++ app does exactly this
        # with boxGain and boxFrameGain rather than a hard handoff, and 4x is
        # about the ratio it is usually run at.
        self.frame_sensitivity_input = self.add_input('frame sensitivity',
                                                      widget_type='drag_float',
                                                      default_value=4.0, min=0.0)
        self.frame_sensitivity_input.widget.speed = 0.01
        self.hue_input = self.add_input('hue', widget_type='drag_float',
                                        default_value=0.33, min=0.0, max=1.0)
        self.hue_input.widget.speed = 0.002
        self.saturation_input = self.add_input('saturation', widget_type='drag_float',
                                               default_value=1.0, min=0.0, max=1.0)
        self.saturation_input.widget.speed = 0.01
        self.alpha_input = self.add_input('alpha', widget_type='drag_float',
                                          default_value=1.0, min=0.0, max=1.0)
        self.alpha_input.widget.speed = 0.01
        self.show_fill_option = self.add_option('show fill', widget_type='checkbox',
                                                default_value=True)
        self.show_frames_option = self.add_option('show frames', widget_type='checkbox',
                                                  default_value=True)
        self.mode_option = self.add_option('color mode', widget_type='combo',
                                           default_value='uniform')
        self.mode_option.widget.combo_items = ['uniform', 'per box']
        # 0.13 is close to the C++ (step + 17) walk: coprime enough with the
        # wheel that a long run of boxes keeps changing hue instead of cycling.
        self.spread_option = self.add_option('hue spread', widget_type='drag_float',
                                             default_value=0.13, min=0.0, max=1.0)
        self.spread_option.widget.speed = 0.002
        self.threshold_option = self.add_option('threshold', widget_type='drag_float',
                                                default_value=0.0, min=0.0, max=1.0)
        self.threshold_option.widget.speed = 0.005
        # Cube edge as a fraction of the box, so the lattice reads as separate
        # cells rather than one solid mass.
        self.fill_option = self.add_option('fill', widget_type='drag_float',
                                           default_value=0.9, min=0.05, max=1.0)
        self.fill_option.widget.speed = 0.01
        self.blend_option = self.add_option('blend', widget_type='combo',
                                            default_value='additive')
        self.blend_option.widget.combo_items = ['additive', 'alpha']

        # Published as one reference for the render thread (see execute).
        self.frame = None
        self._centres = None        # (K, 3) box centres, cached per geometry
        self._geometry_key = None
        self._prog = None
        self._cube_vbo = None
        self._cube_ibo = None
        self._edge_ibo = None
        self._inst_vbo = None
        self._fill_vao = None
        self._frame_vao = None
        self._inst_capacity = 0

    def execute(self):
        if self.cloud_input.fresh_input:
            data = self.cloud_input()
            clusters = data.get(CLUSTER_KEY) if isinstance(data, dict) else None
            # One assignment of one immutable tuple: the sensor thread that
            # delivers the cloud and the main thread that draws it never see a
            # half-updated frame. Anything without a lattice draws nothing.
            if isinstance(clusters, dict) and clusters.get('shape') is not None:
                self.frame = (np.asarray(clusters['values'], dtype=np.float32).reshape(-1),
                              tuple(int(v) for v in clusters['shape']),
                              np.asarray(clusters['origin'], dtype=np.float32).reshape(-1)[:3],
                              np.asarray(clusters['cell'], dtype=np.float32).reshape(-1)[:3])
            else:
                self.frame = None
        super().execute()

    def _box_centres(self, shape, origin, cell):
        """Centres of every box in the lattice, (K, 3) float32. Rebuilt only
        when the lattice geometry changes, which is when the crop, the box
        count or the voxel size does — not every frame."""
        key = (shape, tuple(origin.tolist()), tuple(cell.tolist()))
        if key != self._geometry_key:
            bx, by, bz = shape
            k = np.arange(bx * by * bz, dtype=np.int64)
            # Matching pc_voxel's linear index: x + bx*y + bx*by*z.
            ijk = np.stack((k % bx, (k // bx) % by, k // (bx * by)), axis=1)
            self._centres = (origin + (ijk.astype(np.float32) + 0.5) * cell).astype(np.float32)
            self._geometry_key = key
        return self._centres

    def _instance_data(self):
        """(N, 11) float32 — centre, size, full-value rgb, and the fill and
        frame levels — for the boxes worth drawing, or None.

        Boxes below the threshold on both levels are dropped rather than drawn
        transparent: with a fine subdivision most boxes are empty most of the
        time, and they cost nothing if they never reach the GPU."""
        frame = self.frame
        if frame is None:
            return None
        values, shape, origin, cell = frame
        n_boxes = shape[0] * shape[1] * shape[2]
        if values.size != n_boxes:
            return None

        gain = max(0.0, float(self.sensitivity_input())) * SENSITIVITY_SCALE
        frame_gain = max(0.0, float(self.frame_sensitivity_input())) * SENSITIVITY_SCALE
        fill_level = np.clip(values * gain, 0.0, 1.0)
        frame_level = np.clip(values * frame_gain, 0.0, 1.0)

        threshold = max(0.0, float(self.threshold_option()))
        if not self.show_fill_option():
            fill_level = np.zeros_like(fill_level)
        if not self.show_frames_option():
            frame_level = np.zeros_like(frame_level)
        keep = np.nonzero(np.maximum(fill_level, frame_level) > threshold)[0]
        if keep.size == 0:
            return None

        saturation = float(self.saturation_input())
        hue = float(self.hue_input())
        if self.mode_option() == 'per box':
            hues = hue + keep.astype(np.float32) * float(self.spread_option())
        else:
            hues = np.full(keep.size, hue, dtype=np.float32)

        inst = np.empty((keep.size, 11), dtype=np.float32)
        inst[:, 0:3] = self._box_centres(shape, origin, cell)[keep]
        inst[:, 3:6] = cell * float(self.fill_option())
        # Full value here; the shader scales by each pass's level, which is the
        # same result because hsv_to_rgb is linear in v.
        inst[:, 6:9] = hsv_to_rgb(hues, saturation, np.ones(keep.size, dtype=np.float32))
        inst[:, 9] = fill_level[keep]
        inst[:, 10] = frame_level[keep]
        return inst

    def _ensure_gpu(self, inner_ctx, instances):
        if self._prog is None:
            self._prog = inner_ctx.program(vertex_shader=self._vert_src,
                                           fragment_shader=self._frag_src)
        if self._cube_vbo is None:
            self._cube_vbo = inner_ctx.buffer(np.ascontiguousarray(_CUBE_CORNERS).tobytes())
            self._cube_ibo = inner_ctx.buffer(_CUBE_INDICES.tobytes())
            self._edge_ibo = inner_ctx.buffer(_CUBE_EDGE_INDICES.tobytes())
            self._fill_vao = self._frame_vao = None
        if self._inst_vbo is None or self._inst_capacity < instances:
            self.ctx.defer_release(self._inst_vbo)
            # Headroom, so a cloud that breathes across a box boundary does not
            # reallocate every frame.
            self._inst_capacity = max(instances, 256)
            self._inst_vbo = inner_ctx.buffer(reserve=self._inst_capacity * 11 * 4)
            self._fill_vao = self._frame_vao = None
        if self._fill_vao is None:
            # Two vertex arrays over one instance buffer: the same cube corners
            # and the same per-box data, read as triangles for the fill and as
            # edges for the frame.
            content = [(self._cube_vbo, '3f', 'in_position'),
                       (self._inst_vbo, '3f 3f 3f 2f/i', 'inst_centre', 'inst_size',
                        'inst_rgb', 'inst_levels')]
            self._fill_vao = inner_ctx.vertex_array(self._prog, content, self._cube_ibo)
            self._frame_vao = inner_ctx.vertex_array(self._prog, content, self._edge_ibo)

    def draw(self):
        if self.ctx is None:
            return
        inst = self._instance_data()
        if inst is None:
            return
        instances = inst.shape[0]
        inner_ctx = self.ctx.ctx
        self._ensure_gpu(inner_ctx, instances)
        self._inst_vbo.write(np.ascontiguousarray(inst).tobytes())

        prog = self._prog
        prog['M'].write(self.ctx.get_model_matrix().astype('f4').T.tobytes())
        prog['V'].write(self.ctx.view_matrix.astype('f4').tobytes())
        prog['P'].write(self.ctx.projection_matrix.astype('f4').tobytes())
        prog['alpha'].value = float(self.alpha_input())

        additive = self.blend_option() == 'additive'
        fbo = inner_ctx.fbo
        # Culling stays on (the chain's default): these are closed cubes, and
        # front faces alone give one contribution per box per pixel. A camera
        # placed inside a box therefore does not see it — the right trade for
        # an overlay meant to be read from outside the volume.
        inner_ctx.enable(moderngl.BLEND | moderngl.CULL_FACE)
        if additive:
            inner_ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        fbo.depth_mask = False
        if self.show_fill_option():
            prog['frame_pass'].value = False
            self._fill_vao.render(moderngl.TRIANGLES, instances=instances)
        if self.show_frames_option():
            # After the fill, so the outline reads on top of it.
            prog['frame_pass'].value = True
            self._frame_vao.render(moderngl.LINES, instances=instances)
        fbo.depth_mask = True
        if additive:
            inner_ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

    def custom_cleanup(self):
        # Deletion runs from a DPG handler callback with no GL context current,
        # so the objects go back to the context to be released at the start of
        # its next render block.
        ctx = MGLContext._instance
        if ctx is not None:
            ctx.defer_release(self._fill_vao, self._frame_vao, self._cube_vbo,
                              self._cube_ibo, self._edge_ibo, self._inst_vbo,
                              self._prog)
        self._fill_vao = None
        self._frame_vao = None
        self._cube_vbo = None
        self._cube_ibo = None
        self._edge_ibo = None
        self._inst_vbo = None
        self._prog = None
        self._inst_capacity = 0
