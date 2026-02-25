
import numpy as np
import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.moderngl_base import MGLContext
import moderngl
import threading
import math
from ctypes import c_void_p
from dpg_system.matrix_utils import *
from dpg_system.body_base import BodyData, t_PelvisAnchor, t_ActiveJointCount, quaternion_to_R3_rotation, rotation_matrix_from_vectors
try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None
from dpg_system.body_defs import *

def register_moderngl_nodes():
    Node.app.register_node('mgl_context', MGLContextNode.factory)
    Node.app.register_node('mgl_box', MGLBoxNode.factory)
    Node.app.register_node('mgl_transform', MGLTransformationNode.factory)
    Node.app.register_node('mgl_translate', MGLTransformSingleNode.factory)
    Node.app.register_node('mgl_rotate', MGLRotationNode.factory)
    Node.app.register_node('mgl_scale', MGLScaleNode.factory)
    Node.app.register_node('mgl_camera', MGLCameraNode.factory)
    Node.app.register_node('mgl_display', MGLDisplayNode.factory)
    Node.app.register_node('mgl_sphere', MGLSphereNode.factory)
    Node.app.register_node('mgl_geo_sphere', MGLGeodesicSphereNode.factory)
    Node.app.register_node('mgl_point_cloud', MGLPointCloudNode.factory)
    Node.app.register_node('mgl_model', MGLModelNode.factory)
    Node.app.register_node('mgl_cylinder', MGLCylinderNode.factory)
    Node.app.register_node('mgl_color', MGLColorNode.factory)
    Node.app.register_node('mgl_light', MGLLightNode.factory)
    Node.app.register_node('mgl_material', MGLMaterialNode.factory)
    Node.app.register_node('mgl_texture', MGLTextureNode.factory)
    Node.app.register_node('mgl_plane', MGLPlaneNode.factory)
    Node.app.register_node('mgl_disk', MGLDiskNode.factory)
    Node.app.register_node('mgl_body', MGLBodyNode.factory)
    Node.app.register_node('mgl_surface', MGLSurfaceNode.factory)
    Node.app.register_node('mgl_line', MGLLineNode.factory)
    Node.app.register_node('mgl_text', MGLTextNode.factory)
    Node.app.register_node('mgl_line_array', MGLLineArrayNode.factory)
    Node.app.register_node('mgl_quaternion_rotate', MGLQuaternionRotateNode.factory)
    Node.app.register_node('mgl_axis_angle_rotate', MGLAxisAngleRotateNode.factory)
    Node.app.register_node('mgl_billboard', MGLBillboardNode.factory)
    Node.app.register_node('mgl_partial_disk', MGLPartialDiskNode.factory)
    Node.app.register_node('mgl_torque_arc', MGLTorqueArcNode.factory)
    Node.app.register_node('mgl_smpl_mesh', MGLSMPLMeshNode.factory)
    Node.app.register_node('mgl_smpl_heatmap', MGLSMPLHeatmapNode.factory)


class MGLNode(Node):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.mgl_input = None
        self.mgl_output = None
        self.initialize(args)

    def initialize(self, args):
        self.mgl_input = self.add_input('mgl chain in', triggers_execution=True)
        self.mgl_output = self.add_output('mgl chain out')

    def execute(self):
        if self.mgl_input.fresh_input:
            input_list = self.mgl_input()
            do_draw = False
            t = type(input_list)
            if t == list:
                if type(input_list[0]) == str:
                    if input_list[0] == 'draw':
                        do_draw = True
            elif t == str:
                if input_list == 'draw':
                    do_draw = True
            if do_draw:
                self.ctx = MGLContext.get_instance()
                self.remember_state()
                self.draw()
                self.mgl_output.send('draw')
                self.restore_state()
                self.ctx = None
            else:
                self.handle_other_messages(input_list)

    def remember_state(self):
        pass

    def restore_state(self):
        pass

    def draw(self):
        pass

    def handle_other_messages(self, message):
        pass


class MGLContextNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = MGLContextNode(name, data, args)
        return node
    
    
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.width = 640
        self.height = 480
        self.context = MGLContext.get_instance()
        self.texture_tag = None
        self.image_item = None
        self.external_window = None
        self.fullscreen_window = None
        self.render_target = None

        # PBO double-buffer state for async readback
        self._pbo = [None, None]   # two Buffer objects
        self._pbo_index = 0        # which PBO to read INTO this frame
        self._pbo_ready = False    # first frame has no previous data
        self._pbo_width = 0
        self._pbo_height = 0

        self.auto_render_input = self.add_input('auto_render', widget_type='checkbox', default_value=False, callback=self.toggle_auto_render)
        self.render_trigger = self.add_input('render', triggers_execution=True)
        self.mgl_chain_output = self.add_output('mgl_chain')
        self.texture_output = self.add_output('texture_tag')

        self.width_option = self.add_option('width', widget_type='drag_int', default_value=self.width, callback=self.resize)
        self.height_option = self.add_option('height', widget_type='drag_int', default_value=self.height, callback=self.resize)
        self.display_mode_option = self.add_option('display_mode', widget_type='combo', default_value='node')
        self.display_mode_option.widget.combo_items = ['off', 'node', 'window', 'fullscreen']
        
        self.samples_option = self.add_option('samples', widget_type='combo', default_value='4')
        self.samples_option.widget.combo_items = ['0', '2', '4', '6', '8', '16']

        self.default_light_option = self.add_option('default_light', widget_type='checkbox', default_value=True)
        self.light_pos_option = self.add_option('light_pos', widget_type='drag_float_n', default_value=[2.0, 5.0, 5.0], columns=3)
        self.light_intensity_option = self.add_option('light_intensity', widget_type='drag_float', default_value=0.8)

        self.default_camera_option = self.add_option('default_camera', widget_type='checkbox', default_value=True)
        self.camera_fov_option = self.add_option('camera_fov', widget_type='drag_float', default_value=60.0)
        self.camera_pos_option = self.add_option('camera_pos', widget_type='drag_float_n', default_value=[0.0, 0.0, 3.0], columns=3)
        self.camera_target_option = self.add_option('camera_target', widget_type='drag_float_n', default_value=[0.0, 0.0, 0.0], columns=3)
        

    def toggle_auto_render(self):
        if self.auto_render_input():
            self.add_frame_task()
        else:
            self.remove_frame_tasks()

    def custom_create(self, from_file):
        if from_file and self.auto_render_input():
            self.add_frame_task()
        self.camera_fov_option.widget.set_speed(1.0)

    def frame_task(self):
        self.execute()

    def custom_cleanup(self):
        self.remove_frame_tasks()
        if self.texture_tag:
            dpg.delete_item(self.texture_tag)
        if self.external_window:
            dpg.delete_item(self.external_window)
        if self.fullscreen_window:
            dpg.delete_item(self.fullscreen_window)
        if hasattr(self, 'render_target') and self.render_target:
            self.render_target.release()
        for pbo in self._pbo:
            if pbo is not None:
                pbo.release()

    def resize(self):
        self.width = max(1, self.width_option())
        self.height = max(1, self.height_option())

    def update_display(self, mode):
        # ... (Node display logic omitted for brevity if unchanged, but I need to include context to replace correctly)
        # Handle Node Display
        if mode == 'node':
            # Check if we need to recreate image (missing or wrong texture)
            recreate = False
            if self.image_item is None:
                recreate = True
            elif not dpg.does_item_exist(self.image_item):
                recreate = True
            else:
                # Check if texture matches
                current_conf = dpg.get_item_configuration(self.image_item)
                if current_conf.get('texture_tag') != self.texture_tag:
                    recreate = True
            
            if recreate:
                if self.image_item and dpg.does_item_exist(self.image_item):
                    dpg.delete_item(self.image_item)
                
                # We need a parent. Use static attribute.
                if not hasattr(self, 'image_attribute') or self.image_attribute is None:
                     self.image_attribute = dpg.add_node_attribute(attribute_type=dpg.mvNode_Attr_Static, parent=self.uuid)
                
                # Check if attribute exists
                if not dpg.does_item_exist(self.image_attribute):
                    self.image_attribute = dpg.add_node_attribute(attribute_type=dpg.mvNode_Attr_Static, parent=self.uuid)

                self.image_item = dpg.add_image(self.texture_tag, parent=self.image_attribute)
        else:
            if self.image_item:
                dpg.delete_item(self.image_item)
                self.image_item = None
            if hasattr(self, 'image_attribute') and self.image_attribute:
                 dpg.delete_item(self.image_attribute)
                 self.image_attribute = None

        # Handle Window Display
        if mode == 'window':
            if self.external_window is None or not dpg.does_item_exist(self.external_window):
                with dpg.window(label="MGL Output", width=self.width, height=self.height) as win:
                    self.external_window = win
                    dpg.add_image(self.texture_tag, parent=win)
            else:
                # Sync Window Size
                dpg.configure_item(self.external_window, show=True, width=self.width, height=self.height)
                
                # Check image inside window
                children = dpg.get_item_children(self.external_window, slot=1)
                image_found = False
                if children:
                    for child in children:
                        if dpg.get_item_type(child) == "mvAppItemType::mvImage":
                             if dpg.get_item_configuration(child).get('texture_tag') != self.texture_tag:
                                 dpg.delete_item(child)
                             else:
                                 image_found = True
                                 # Image should fill window?
                                 # If we just let it be, it stays texture size.
                                 # We want it to match render size, which matches window size.
                                 dpg.configure_item(child, width=self.width, height=self.height)
                
                if not image_found:
                    dpg.add_image(self.texture_tag, parent=self.external_window, width=self.width, height=self.height)
        else:
            if self.external_window:
                dpg.delete_item(self.external_window)
                self.external_window = None

        # Handle Fullscreen Display
        if mode == 'fullscreen':
            vp_width = dpg.get_viewport_width()
            vp_height = dpg.get_viewport_height()
            
            if self.fullscreen_window is None or not dpg.does_item_exist(self.fullscreen_window):
                with dpg.window(label="MGL Fullscreen", no_title_bar=True, no_resize=True, no_move=True, no_scrollbar=True, no_collapse=True, no_background=True, show=True) as win:
                    self.fullscreen_window = win
                    dpg.set_item_pos(win, [0, 0])
                    dpg.set_item_width(win, vp_width)
                    dpg.set_item_height(win, vp_height)
                    dpg.add_image(self.texture_tag, parent=win, width=vp_width, height=vp_height)
            else:
                dpg.configure_item(self.fullscreen_window, show=True)
                dpg.set_item_width(self.fullscreen_window, vp_width)
                dpg.set_item_height(self.fullscreen_window, vp_height)
                
                # Check image
                children = dpg.get_item_children(self.fullscreen_window, slot=1)
                image_found = False
                if children:
                    for child in children:
                         if dpg.get_item_type(child) == "mvAppItemType::mvImage":
                             if dpg.get_item_configuration(child).get('texture_tag') != self.texture_tag:
                                 dpg.delete_item(child)
                             else:
                                 image_found = True
                                 dpg.configure_item(child, width=vp_width, height=vp_height)
                if not image_found:
                     dpg.add_image(self.texture_tag, parent=self.fullscreen_window, width=vp_width, height=vp_height)
        else:
            if self.fullscreen_window:
                dpg.delete_item(self.fullscreen_window)
                self.fullscreen_window = None

    def execute(self):
        # Escape Exits Fullscreen
        if self.display_mode_option() == 'fullscreen' and dpg.is_key_pressed(dpg.mvKey_Escape):
            self.display_mode_option.set('window')
            dpg.set_value(self.display_mode_option.widget.uuid, 'window')
            self.update_display('window')

        # 0. Sync Window Image Size (Stretch to fit)
        if self.external_window and dpg.does_item_exist(self.external_window) and self.display_mode_option() == 'window':
             win_w = dpg.get_item_width(self.external_window)
             win_h = dpg.get_item_height(self.external_window)
             
             # Find image child and update its size to fill window
             children = dpg.get_item_children(self.external_window, slot=1)
             if children:
                 for child in children:
                     if dpg.get_item_type(child) == "mvAppItemType::mvImage":
                         # Only update if different to avoid spamming commands
                         if dpg.get_item_width(child) != win_w or dpg.get_item_height(child) != win_h:
                             dpg.configure_item(child, width=win_w, height=win_h)
        
        # Update FBO size/Check
        try:
            samples = int(self.samples_option())
        except:
            samples = 4
        
        # Scope the ModernGL Context to avoid polluting global GL state (e.g. for legacy gl_nodes)
        with self.context.ctx:
            # Create/update render target INSIDE the context scope —
            # texture/FBO creation requires an active GL context.
            if self.render_target is None or self.render_target.width != self.width or self.render_target.height != self.height or self.render_target.samples != samples:
                if self.render_target:
                    self.render_target.release()
                self.render_target = self.context.create_render_target(self.width, self.height, samples)

            # Activate Local Target
            self.context.use_render_target(self.render_target)
            
            # Reset Context State
            self.context.current_color = (1.0, 1.0, 1.0, 1.0)
            self.context.lights = []
            self.context.default_light_count = 0
            self.context.current_material = {
                'ambient': [0.1, 0.1, 0.1],
                'diffuse': [1.0, 1.0, 1.0],
                'specular': [0.5, 0.5, 0.5],
                'shininess': 32.0
            }

            # Helper to safely coerce option values to 3-element lists
            def _as_vec3(val, default):
                if np.isscalar(val):
                    return [float(val), default[1], default[2]]
                val = list(val)
                while len(val) < 3:
                    val.append(default[len(val)])
                return val[:3]

            # Inject default light (overridden if mgl_light nodes are present)
            if self.default_light_option():
                pos = _as_vec3(self.light_pos_option(), [2.0, 5.0, 5.0])
                intensity = self.light_intensity_option()
                self.context.lights.append({
                    'pos': pos,
                    'ambient': [0.15, 0.15, 0.15],
                    'diffuse': [1.0, 1.0, 1.0],
                    'specular': [0.6, 0.6, 0.6],
                    'intensity': intensity
                })
                self.context.default_light_count = 1

            # Inject default camera (overridden if mgl_camera nodes are present)
            if self.default_camera_option():
                cam_pos = _as_vec3(self.camera_pos_option(), [0.0, 0.0, 3.0])
                cam_target = _as_vec3(self.camera_target_option(), [0.0, 0.0, 0.0])
                cam_fov = self.camera_fov_option()
                aspect = self.width / self.height if self.height > 0 else 1.0
                self.context.set_projection_matrix(perspective(cam_fov, aspect, 0.1, 100.0))
                self.context.set_view_matrix(look_at(cam_pos, cam_target, [0.0, 1.0, 0.0]))
                if 'view_pos' in self.context.default_shader:
                    self.context.default_shader['view_pos'].value = tuple(cam_pos)
            
            # Sync back actual samples if fallback occurred
            if self.render_target.samples != samples:
                self.samples_option.set(str(self.render_target.samples))
            
            # Clear
            self.context.clear(0.0, 0.0, 0.0, 1.0)
            
            # Signal Downstream to Draw
            self.mgl_chain_output.send('draw')
            
            # --- PBO Double-Buffered Async Readback ---
            # Resolve MSAA if needed (same as get_pixel_data but without the sync read)
            rt = self.render_target
            if rt.samples > 0 and rt.msaa_fbo:
                self.context.ctx.copy_framebuffer(rt.fbo, rt.msaa_fbo)
            
            buf_size = self.width * self.height * 4  # RGBA uint8
            
            # Recreate PBOs if resolution changed
            if self._pbo_width != self.width or self._pbo_height != self.height:
                for i in range(2):
                    if self._pbo[i] is not None:
                        self._pbo[i].release()
                    self._pbo[i] = self.context.ctx.buffer(reserve=buf_size)
                self._pbo_width = self.width
                self._pbo_height = self.height
                self._pbo_ready = False
                self._pbo_index = 0
            
            # Determine which PBO to write into (current) and which to read from (previous)
            write_pbo = self._pbo[self._pbo_index]
            read_pbo = self._pbo[1 - self._pbo_index]
            
            # Initiate async DMA transfer: FBO -> write_pbo
            rt.fbo.read_into(write_pbo, components=4)
            
            # Read from the OTHER PBO (filled last frame)
            if self._pbo_ready:
                data = read_pbo.read()
            else:
                # First frame: fall back to synchronous read
                data = write_pbo.read()
                self._pbo_ready = True
            
            # Swap PBO index for next frame
            self._pbo_index = 1 - self._pbo_index
        
        # OpenGL framebuffer origin is bottom-left, DPG textures expect top-left.
        # Flip rows vertically to correct the orientation.
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
        pixels = np.flipud(pixels).flatten().astype(np.float32) / 255.0
        
        if self.texture_tag is None or not dpg.does_item_exist(self.texture_tag):
             with dpg.texture_registry(show=False):
                # Let DPG generate ID to avoid alias collisions
                self.texture_tag = dpg.add_dynamic_texture(self.width, self.height, pixels)
             self.texture_output.send(self.texture_tag)
        else:
             # Check size
            tex_w = dpg.get_item_width(self.texture_tag)
            tex_h = dpg.get_item_height(self.texture_tag)
            
            if tex_w != self.width or tex_h != self.height:
                 # Recreate texture
                 dpg.delete_item(self.texture_tag)
                 with dpg.texture_registry(show=False):
                    self.texture_tag = dpg.add_dynamic_texture(self.width, self.height, pixels)
                 # Note: self.texture_tag is new ID now.
            else:
                dpg.set_value(self.texture_tag, pixels)
            
            self.texture_output.send(self.texture_tag)
        
        # 2. Update Display
        mode = self.display_mode_option()
        self.update_display(mode)


class MGLDisplayNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        return MGLDisplayNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.texture_input = self.add_input('texture_tag', triggers_execution=True)
        self.width_input = self.add_input('width', widget_type='drag_int', default_value=0)
        self.height_input = self.add_input('height', widget_type='drag_int', default_value=0)
        self.fullscreen_input = self.add_input('fullscreen', widget_type='checkbox', default_value=False, callback=self.toggle_fullscreen)
        self.image_attribute = None
        self.image_item = None
        self.fullscreen = False
        self.fullscreen_window = None
        self.texture_tag = None
        
        # Permanent Handler for ESC
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=self.on_esc)

    def custom_create(self, from_file):
        self.image_attribute = dpg.add_node_attribute(attribute_type=dpg.mvNode_Attr_Static)

    def custom_cleanup(self):
        if self.fullscreen_window:
            dpg.delete_item(self.fullscreen_window)
        if self.image_item:
            dpg.delete_item(self.image_item)

    def on_esc(self, sender=None, app_data=None):
        if self.fullscreen:
            print("MGLDisplayNode: ESC detected, exiting fullscreen")
            # Explicitly update wrapper AND widget to ensure next read is False
            if hasattr(self.fullscreen_input, 'set'):
                self.fullscreen_input.set(False)
            dpg.set_value(self.fullscreen_input.widget.uuid, False)
            self.toggle_fullscreen()

    def toggle_fullscreen(self):
        self.fullscreen = self.fullscreen_input()
        if self.fullscreen:
            if not self.fullscreen_window:
                with dpg.window(label="MGL Fullscreen", no_title_bar=True, no_resize=True, no_move=True, no_scrollbar=True, no_collapse=True, no_background=True, show=True) as win:
                    self.fullscreen_window = win
            else:
                 dpg.configure_item(self.fullscreen_window, show=True)
            self.update_fullscreen_image()
        else:
            if self.fullscreen_window:
                dpg.configure_item(self.fullscreen_window, show=False)

    def update_fullscreen_image(self):
        if self.fullscreen and self.fullscreen_window and self.texture_tag:
            vp_width = dpg.get_viewport_width()
            vp_height = dpg.get_viewport_height()
            
            dpg.set_item_width(self.fullscreen_window, vp_width)
            dpg.set_item_height(self.fullscreen_window, vp_height)
            dpg.set_item_pos(self.fullscreen_window, [0, 0])
            
            dpg.delete_item(self.fullscreen_window, children_only=True)
            dpg.add_image(self.texture_tag, parent=self.fullscreen_window, width=vp_width, height=vp_height)

    def execute(self):
        self.texture_tag = self.texture_input()
        if self.texture_tag:
            if self.image_item:
                current_texture = dpg.get_item_configuration(self.image_item)['texture_tag']
                if current_texture != self.texture_tag:
                    dpg.delete_item(self.image_item)
                    self.image_item = None
            
            # Determine Size
            w = self.width_input()
            h = self.height_input()
            
            if w <= 0: w = dpg.get_item_width(self.texture_tag)
            if h <= 0: h = dpg.get_item_height(self.texture_tag)
            
            if self.image_item is None and self.image_attribute:
                self.image_item = dpg.add_image(self.texture_tag, parent=self.image_attribute, width=w, height=h)
            elif self.image_item:
                 dpg.configure_item(self.image_item, width=w, height=h)
            
            if self.fullscreen:
                self.update_fullscreen_image()

class MGLTransformNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLTransformNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.ctx = None

    def get_translation_matrix(self, t):
        m = np.identity(4, dtype=np.float32)
        m[:3, 3] = t
        return m
    
    def get_scale_matrix(self, s):
        return np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
    
    def get_rotation_matrix(self, r):
        # Euler XYZ
        # Use helper from matrix_utils or build it
        rx = rotation_matrix(r[0], [1, 0, 0])
        ry = rotation_matrix(r[1], [0, 1, 0])
        rz = rotation_matrix(r[2], [0, 0, 1])
        # Order implies T * R * S usually, or T * Rz * Ry * Rx * S
        # matrix multiplication is associative but not commutative.
        # we want R = Rz * Ry * Rx
        return np.dot(rz, np.dot(ry, rx))

    def remember_state(self):
        if self.ctx is not None:
            self.ctx.push_matrix()

    def restore_state(self):
        if self.ctx is not None:
            self.ctx.pop_matrix()

    def draw(self):
        if self.ctx is not None:
            self.perform_transformation()

    def perform_transformation(self):
        pass


class MGLTransformationNode(MGLTransformNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLTransformationNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.translate_input = self.add_input(label='translate', widget_type='drag_float_n', default_value=[0, 0, 0], speed=1.0, columns=3)
        self.rotate_input = self.add_input(label='rotate', widget_type='drag_float_n', default_value=[0, 0, 0], speed=1.0, columns=3)
        self.scale_input = self.add_input(label='scale', widget_type='drag_float_n', default_value=[1, 1, 1], speed=1.0, columns=3)

    def custom_create(self, from_file):
        dpg.configure_item(self.translate_input.widget.uuids[2], label='translate')
        dpg.configure_item(self.translate_input.widget.uuids[0], width=45)
        dpg.configure_item(self.translate_input.widget.uuids[1], width=45)
        dpg.configure_item(self.translate_input.widget.uuids[2], width=45)
        dpg.configure_item(self.rotate_input.widget.uuids[2], label='rotate')
        dpg.configure_item(self.rotate_input.widget.uuids[0], width=45)
        dpg.configure_item(self.rotate_input.widget.uuids[1], width=45)
        dpg.configure_item(self.rotate_input.widget.uuids[2], width=45)
        dpg.configure_item(self.rotate_input.widget.uuids[0], speed=1.0)
        dpg.configure_item(self.rotate_input.widget.uuids[1], speed=1.0)
        dpg.configure_item(self.rotate_input.widget.uuids[2], speed=1.0)
        dpg.configure_item(self.scale_input.widget.uuids[2], label='scale')
        dpg.configure_item(self.scale_input.widget.uuids[0], width=45)
        dpg.configure_item(self.scale_input.widget.uuids[1], width=45)
        dpg.configure_item(self.scale_input.widget.uuids[2], width=45)

    def perform_transformation(self):
        if self.ctx is not None:
            t = self.translate_input()
            self.ctx.multiply_matrix(self.get_translation_matrix(t))

            # Rotation
            r = self.rotate_input()
            self.ctx.multiply_matrix(self.get_rotation_matrix(r))

            # Scale
            s = self.scale_input()
            self.ctx.multiply_matrix(self.get_scale_matrix(s))


class MGLTransformSingleNode(MGLTransformNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLTransformSingleNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.values = [0.0, 0.0, 0.0]
        self.values[0] = self.arg_as_float(default_value=0.0)
        self.values[1] = self.arg_as_float(index=1, default_value=0.0)
        self.values[2] = self.arg_as_float(index=2, default_value=0.0)

        self.x = self.add_input('x', widget_type='drag_float', default_value=self.values[0], callback=self.receive_value)
        self.y = self.add_input('y', widget_type='drag_float', default_value=self.values[1])
        self.z = self.add_input('z', widget_type='drag_float', default_value=self.values[2])

        self.reset_button = self.add_property('reset', widget_type='button', callback=self.reset)

    def receive_value(self):
        value = self.x()
        if not np.isscalar(value):
            if isinstance(value, np.ndarray):
                if value.size > 1:
                    value = value.flatten()
                    if value.shape[0] == 3:
                        self.x.set(value[0])
                        self.y.set(value[1])
                        self.z.set(value[2])
            elif isinstance(value, list):
                if len(value) == 3:
                    self.x.set(value[0])
                    self.y.set(value[1])
                    self.z.set(value[2])
            elif self.app.torch_available and isinstance(value, torch.Tensor):
                value = value.flatten().cpu()
                if value.shape[0] == 3:
                    self.x.set(float(value[0]))
                    self.y.set(float(value[1]))
                    self.z.set(float(value[2]))

    def reset(self):
        self.x.set(0.0)
        self.y.set(0.0)
        self.z.set(0.0)

    def perform_transformation(self):
        self.values[0] = self.x()
        self.values[1] = self.y()
        self.values[2] = self.z()
        if self.ctx is not None:
            self.ctx.multiply_matrix(self.get_translation_matrix(self.values))


class MGLRotationNode(MGLTransformSingleNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLRotationNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def custom_create(self, from_file):
        dpg.configure_item(self.x.widget.uuids[0], speed=1.0)
        dpg.configure_item(self.y.widget.uuids[0], speed=1.0)
        dpg.configure_item(self.z.widget.uuids[0], speed=1.0)

    def perform_transformation(self):
        self.values[0] = self.x()
        self.values[1] = self.y()
        self.values[2] = self.z()
        if self.ctx is not None:
            self.ctx.multiply_matrix(self.get_rotation_matrix(self.values))


class MGLScaleNode(MGLTransformSingleNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLScaleNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def reset(self):
        self.x.set(1.0)
        self.y.set(1.0)
        self.z.set(1.0)

    def perform_transformation(self):
        self.values[0] = self.x()
        self.values[1] = self.y()
        self.values[2] = self.z()
        if self.ctx is not None:
            self.ctx.multiply_matrix(self.get_scale_matrix(self.values))


class MGLColorNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLColorNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.color_input = self.add_input('color', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0])
        self.prev_color = [1.0, 1.0, 1.0, 1.0]

    def remember_state(self):
        self.prev_color = self.ctx.current_color

    def restore_state(self):
        self.ctx.current_color = self.prev_color

    def draw(self):
        c = self.color_input()
        # Normalize if needed
        if len(c) > 4: c = c[:4]
        elif len(c) == 3: c = [*c, 255 if c[0] > 1.0 else 1.0]

        # Check for 0-255 range
        if max(c) > 1.0:
            c = [val / 255.0 for val in c]
            self.ctx.current_color = tuple(c)


class MGLLightNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLLightNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.pos_input = self.add_input('position', widget_type='drag_float_n', default_value=[0.0, 5.0, 5.0], columns=3)
        self.ambient_input = self.add_input('ambient', widget_type='color_picker', default_value=[0.1, 0.1, 0.1])
        self.diffuse_input = self.add_input('diffuse', widget_type='color_picker', default_value=[1.0, 1.0, 1.0])
        self.specular_input = self.add_input('specular', widget_type='color_picker', default_value=[1.0, 1.0, 1.0])
        self.intensity_input = self.add_input('intensity', widget_type='drag_float', default_value=1.0)
        # Inherits mgl_input/output

    def custom_create(self, from_file):
        dpg.configure_item(self.ambient_input.widget.uuids[0], label='ambient')
        dpg.configure_item(self.diffuse_input.widget.uuids[0], label='diffuse')
        dpg.configure_item(self.specular_input.widget.uuids[0], label='specular')
        dpg.configure_item(self.pos_input.widget.uuids[0], width=45)
        dpg.configure_item(self.pos_input.widget.uuids[1], width=45)
        dpg.configure_item(self.pos_input.widget.uuids[2], width=45)

    def draw(self):
        pos = self.pos_input()
        if len(pos) == 3: pos = [*pos, 1.0]

        if self.ctx is not None:
            # Clear default lights on first user-placed light
            if hasattr(self.ctx, 'default_light_count') and self.ctx.default_light_count > 0:
                self.ctx.lights = self.ctx.lights[self.ctx.default_light_count:]
                self.ctx.default_light_count = 0

            model = self.ctx.get_model_matrix()
            # p_world = M * p_local
            world_pos = np.dot(model, pos)

            # Normalize Colors
            def norm(c):
                if len(c) > 3: c = c[:3]
                if max(c) > 1.0:
                    return [val / 255.0 for val in c]
                return c

            # Register Light
            light_data = {
                'pos': world_pos[:3],
                'ambient': norm(self.ambient_input()),
                'diffuse': norm(self.diffuse_input()),
                'specular': norm(self.specular_input()),
                'intensity': self.intensity_input()
            }
            self.ctx.lights.append(light_data)


class MGLMaterialNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLMaterialNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.ambient_input = self.add_input('ambient', widget_type='color_picker', default_value=[0.1, 0.1, 0.1])
        self.diffuse_input = self.add_input('diffuse', widget_type='color_picker', default_value=[1.0, 1.0, 1.0])
        self.specular_input = self.add_input('specular', widget_type='color_picker', default_value=[0.5, 0.5, 0.5])
        self.shininess_input = self.add_input('shininess', widget_type='drag_float', default_value=32.0, min_value=1.0, max_value=256.0)
        self.prev_material = None

    def custom_create(self, from_file):
        dpg.configure_item(self.ambient_input.widget.uuids[0], label='ambient')
        dpg.configure_item(self.diffuse_input.widget.uuids[0], label='diffuse')
        dpg.configure_item(self.specular_input.widget.uuids[0], label='specular')

    def remember_state(self):
        self.prev_material = self.ctx.current_material.copy()

    def restore_state(self):
        self.ctx.current_material = self.prev_material

    def draw(self):
        if self.ctx is not None:

            # Normalize colors
            def norm(c):
                # if len(c) > 3: c = c[:3] # Allow alpha
                if max(c) > 1.0:
                    return [val / 255.0 for val in c]
                # Ensure 4 components (defaults to alpha 1.0 if missing)
                if len(c) == 3:
                    return list(c) + [1.0]
                return c

            self.ctx.current_material = {
                'ambient': norm(self.ambient_input()),
                'diffuse': norm(self.diffuse_input()),
                'specular': norm(self.specular_input()),
                'shininess': self.shininess_input()
            }


class MGLTextureNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLTextureNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.texture = None
        self.width = 0
        self.height = 0
        self.channels = 0

    def initialize(self, args):
        super().initialize(args)
        # Input: Numpy/Torch array
        self.source_input = self.add_input('source', triggers_execution=True)
        # Output: Texture Object
        self.texture_output = self.add_output('texture')

    def execute(self):
        # Handle Texture Update
        if self.source_input.fresh_input:
            data = self.source_input()
            if data is not None:
                self.update_texture(data)
        
        # Output Texture
        if self.texture:
            self.texture_output.send(self.texture)

        # Handle Chain Propagation
        if self.mgl_input.fresh_input:
            msg = self.mgl_input()
            self.mgl_output.send(msg)

    def update_texture(self, data):
        # Handle formats
        if isinstance(data, list):
            data = np.array(data, dtype=np.uint8)
        elif self.app.torch_available and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy().astype(np.uint8)
        
        if not isinstance(data, np.ndarray):
            return

        # data shape: [H, W, C]
        if data.ndim != 3:
            print(f"MGLTextureNode: Expected 3D array [H, W, C], got {data.shape}")
            return

        h, w, c = data.shape
        if c not in [3, 4]:
            print(f"MGLTextureNode: Expected 3 or 4 channels, got {c}")
            return

        # Check if recreate needed
        if self.texture is None or w != self.width or h != self.height or c != self.channels:
            if self.texture:
                self.texture.release()
            
            self.width = w
            self.height = h
            self.channels = c
            
            ctx = MGLContext.get_instance().ctx
            self.texture = ctx.texture((w, h), c)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR) # Smooth scaling
            # Flipping? Often textures are upside down relative to GL.
            # Usually images are top-down, GL is bottom-up.
            # Standard is data should be right for GL, or use build_mipmaps
        
        # Write data
        # Note: moderngl expects bytes
        # Provide contiguous bytes
        self.texture.write(data.tobytes())


class MGLShapeNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLShapeNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)

    def end_initialization(self):
        self.mode_input = self.add_input('mode', widget_type='combo', default_value='solid')
        self.mode_input.widget.combo_items = ['solid', 'wireframe', 'points']
        self.cull_input = self.add_input('cull', widget_type='checkbox', default_value=True)
        self.point_size_input = self.add_input('point_size', widget_type='drag_float', default_value=4.0, min_value=1.0)
        self.round_input = self.add_input('round', widget_type='checkbox', default_value=True)
        self.texture_input = self.add_input('texture')
        self.vbo = None
        self.ibo = None
        self.vao = None
        self.prog = None

    def geometry_changed(self):
        self.vao = None
        self.vbo = None

    def create_geometry(self):
        return None, None

    def render_geometry(self, vertices, indices=None):
        if self.ctx is not None and vertices is not None and len(vertices) > 0:
            inner_ctx = self.ctx.ctx
            data = np.array(vertices, dtype='f4')
            self.vbo = inner_ctx.buffer(data.tobytes())
            
            if indices is not None:
                ind_data = np.array(indices, dtype='i4')
                self.ibo = inner_ctx.buffer(ind_data.tobytes())
            else:
                self.ibo = None
                
            self.prog = self.ctx.default_shader
            
            if self.ibo is not None:
                self.vao = inner_ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord')], self.ibo)
            else:
                self.vao = inner_ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord')])

    def handle_shape_params(self):
        pass

    def draw(self):
        if self.ctx is not None:
            inner_ctx = self.ctx.ctx
            if self.vao is None:
                vertices, indices = self.create_geometry()
                self.render_geometry(vertices, indices)
            
            if self.prog is None:
                return

            self.handle_shape_params()
            if 'V' in self.prog:
                self.prog['V'].write(self.ctx.view_matrix.tobytes())
            if 'P' in self.prog:
                self.prog['P'].write(self.ctx.projection_matrix.tobytes())
            if 'color' in self.prog:
                c = self.ctx.current_color
                self.prog['color'].value = tuple(c)

            # Update Lights and Material
            # DEBUG: Check Context State
            if hasattr(self, 'debug_light_count') and self.debug_light_count < 20:
                # print(f"MGLBodyNode: Lights={len(self.ctx.lights)}, Mat={self.ctx.current_material['diffuse']}")
                # print(f"Shader has num_lights? {'num_lights' in self.prog}")
                self.debug_light_count += 1
            if not hasattr(self, 'debug_light_count'):
                self.debug_light_count = 0

            self.ctx.update_lights(self.prog)
            self.ctx.update_material(self.prog)

            # Render Mode
            mode = self.mode_input()
            cull = self.cull_input()

            # Shader Point Culling
            if 'point_culling' in self.prog:
                self.prog['point_culling'].value = (mode == 'points' and cull)
            if 'round_points' in self.prog:
                self.prog['round_points'].value = (mode == 'points' and self.round_input())

            # Texturing
            tex = self.texture_input()
            if tex is not None and isinstance(tex, moderngl.Texture):
                tex.use(location=0)
                if 'diffuse_map' in self.prog:
                    self.prog['diffuse_map'].value = 0
                if 'has_texture' in self.prog:
                    self.prog['has_texture'].value = True
            else:
                if 'has_texture' in self.prog:
                    self.prog['has_texture'].value = False

            if cull:
                inner_ctx.enable(moderngl.CULL_FACE)
            else:
                inner_ctx.disable(moderngl.CULL_FACE)

            if mode == 'wireframe':
                inner_ctx.wireframe = True
                self.vao.render()
                inner_ctx.wireframe = False
            elif mode == 'points':
                if 'point_size' in self.prog:
                    self.prog['point_size'].value = self.point_size_input()
                self.vao.render(mode=moderngl.POINTS)
            else:
                self.vao.render()


class MGLBoxNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLBoxNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.size_input = self.add_input('size', widget_type='drag_float_n', default_value=[1.0, 1.0, 1.0], columns=3,
                                         speed=0.01)
        self.end_initialization()

    def custom_create(self, from_file):
        dpg.configure_item(self.size_input.widget.uuids[2], label='size')
        dpg.configure_item(self.size_input.widget.uuids[0], width=45)
        dpg.configure_item(self.size_input.widget.uuids[1], width=45)
        dpg.configure_item(self.size_input.widget.uuids[2], width=45)

    def create_geometry(self):
        # Front face
        # ...
        vertices = [
            # Front face
            -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0,

            # Back face
            -0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 0.0,
            -0.5, 0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 1.0,
            0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 0.0,

            # Top face
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0,

            # Bottom face
            -0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 0.0, 1.0,
            0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 1.0, 1.0,
            0.5, -0.5, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0,
            -0.5, -0.5, 0.5, 0.0, -1.0, 0.0, 0.0, 0.0,

            # Right face
            0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,

            # Left face
            -0.5, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0,
            -0.5, -0.5, 0.5, -1.0, 0.0, 0.0, 1.0, 0.0,
            -0.5, 0.5, 0.5, -1.0, 0.0, 0.0, 1.0, 1.0,
            -0.5, 0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 1.0,
        ]

        indices = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ]

        return vertices, indices

    def handle_shape_params(self):
        if self.ctx is not None:
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                s = self.size_input()
                scale_mat = np.diag([s[0], s[1], s[2], 1.0]).astype('f4')
                # Pre-multiply scale for local transformation
                # Post-multiply scale for local transformation
                model = np.dot(model, scale_mat)
                self.prog['M'].write(model.astype('f4').T.tobytes())

class MGLSphereNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLSphereNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.radius_input = self.add_input('radius', widget_type='drag_float', default_value=0.5, speed=0.01)
        self.stacks_input = self.add_input('stacks', widget_type='drag_int', default_value=16, min_value=3, callback=self.geometry_changed)
        self.sectors_input = self.add_input('sectors', widget_type='drag_int', default_value=32, min_value=3, callback=self.geometry_changed)
        self.end_initialization()

    def custom_create(self, from_file):
        dpg.configure_item(self.stacks_input.widget.uuids[0], width=100)
        dpg.configure_item(self.sectors_input.widget.uuids[0], width=100)

    def create_geometry(self):
        import math
        vertices = []
        indices = []

        stacks = max(3, self.stacks_input())
        sectors = max(3, self.sectors_input())

        # Generate Vertices
        for i in range(stacks + 1):
            lat = math.pi * (-0.5 + float(i) / stacks)
            y = math.sin(lat)
            zr = math.cos(lat)
            
            for j in range(sectors + 1):
                lng = 2 * math.pi * float(j) / sectors
                x = zr * math.cos(lng)
                z = zr * math.sin(lng)
                
                # Pos (3f) + Normal (3f) + UV (2f)
                u = float(j) / sectors
                v = float(i) / stacks
                vertices.extend([x, y, z, x, y, z, u, v])
                
        # Generate Indices
        # k1--k1+1
        # |  / |
        # | /  |
        # k2--k2+1
        for i in range(stacks):
            k1 = i * (sectors + 1)
            k2 = k1 + sectors + 1
            
            for j in range(sectors):
                indices.extend([k1, k2, k1 + 1])
                indices.extend([k1 + 1, k2, k2 + 1])
                k1 += 1
                k2 += 1

        return vertices, indices

    def handle_shape_params(self):
        if self.ctx is not None:
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                r = self.radius_input()
                scale_mat = np.diag([r, r, r, 1.0]).astype('f4')
                # Post-multiply scale for local transformation
                model = np.dot(model, scale_mat)
                self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLCylinderNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLCylinderNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.radius_input = self.add_input('radius', widget_type='drag_float', default_value=0.5, speed=0.01)
        self.height_input = self.add_input('height', widget_type='drag_float', default_value=1.0, speed=0.01)
        self.slices_input = self.add_input('slices', widget_type='drag_int', default_value=32, min_value=3, callback=self.geometry_changed)
        self.end_initialization()

    def custom_create(self, from_file):
        dpg.configure_item(self.slices_input.widget.uuids[0], width=100)

    def create_geometry(self):
        import math
        vertices = []
        indices = []
        
        slices = max(3, self.slices_input())
        
        # --- Top Cap ---
        # Center Top (UV 0.5, 0.5)
        base_top_center = len(vertices) // 8
        vertices.extend([0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
        
        # Circle Top
        base_top_circle = len(vertices) // 8
        for i in range(slices + 1):
            theta = 2 * math.pi * float(i) / slices
            x = math.cos(theta)
            z = math.sin(theta)
            u = 0.5 + 0.5 * x
            v = 0.5 + 0.5 * z
            vertices.extend([x, 0.5, z, 0.0, 1.0, 0.0, u, v])
            
        # Indices Top (Fan)
        # Center -> i+1 -> i (Standard CCW for Up Normal)
        for i in range(slices):
            indices.extend([base_top_center, base_top_circle + i + 1, base_top_circle + i])


        # --- Bottom Cap ---
        # Center Bottom (UV 0.5, 0.5)
        base_bot_center = len(vertices) // 8
        vertices.extend([0.0, -0.5, 0.0, 0.0, -1.0, 0.0, 0.5, 0.5])
        
        # Circle Bottom
        base_bot_circle = len(vertices) // 8
        for i in range(slices + 1):
            theta = 2 * math.pi * float(i) / slices
            x = math.cos(theta)
            z = math.sin(theta)
            u = 0.5 + 0.5 * x
            v = 0.5 + 0.5 * z
            vertices.extend([x, -0.5, z, 0.0, -1.0, 0.0, u, v])
            
        # Indices Bottom (Fan)
        # Center -> i -> i+1 (Standard CCW for Down Normal)
        for i in range(slices):
            indices.extend([base_bot_center, base_bot_circle + i, base_bot_circle + i + 1])


        # --- Side ---
        base_side = len(vertices) // 8
        # Top Row (v=1)
        for i in range(slices + 1):
            theta = 2 * math.pi * float(i) / slices
            x = math.cos(theta)
            z = math.sin(theta)
            u = float(i) / slices
            vertices.extend([x, 0.5, z, x, 0.0, z, u, 1.0])

        # Bottom Row (v=0)
        for i in range(slices + 1):
            theta = 2 * math.pi * float(i) / slices
            x = math.cos(theta)
            z = math.sin(theta)
            u = float(i) / slices
            vertices.extend([x, -0.5, z, x, 0.0, z, u, 0.0])

        # Indices Side
        # k (Top) -- k+1
        # |          |
        # k+N (Bot) -- k+N+1
        offset = slices + 1
        for i in range(slices):
            k = base_side + i
            # Tri 1: Top-Right, Top-Left, Bot-Right (CCW)
            indices.extend([k, k + 1, k + offset])
            # Tri 2: Top-Left, Bot-Left, Bot-Right (CCW)
            indices.extend([k + 1, k + offset + 1, k + offset])
            
        return vertices, indices

    def handle_shape_params(self):
        if self.ctx is not None:
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                r = self.radius_input()
                h = self.height_input()
                scale_mat = np.diag([r, h, r, 1.0]).astype('f4')
                # Post-multiply scale for local transformation
                model = np.dot(model, scale_mat)
                self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLGeodesicSphereNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLGeodesicSphereNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.radius_input = self.add_input('radius', widget_type='drag_float', default_value=0.5, speed=0.01)
        self.subdivisions_input = self.add_input('subdivisions', widget_type='drag_int', default_value=2, min_value=0, max_value=5, callback=self.geometry_changed)
        self.end_initialization()

    def create_geometry(self):
        import math
        
        radius = 1.0 # Base radius, scaled by model matrix later
        t = (1.0 + math.sqrt(5.0)) / 2.0

        # Base Icosahedron Vertices
        verts = [
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
        ]
        
        # Normalize to radius 1
        def normalize(v):
            l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            return [v[0]/l, v[1]/l, v[2]/l]
            
        verts = [normalize(v) for v in verts]

        # Base Icosahedron Indices (Triangles)
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        # Subdivision Cache to avoid duplicate vertices
        midpoint_cache = {}

        def get_midpoint(i1, i2):
            # Key is sorted tuple ensuring 1-2 is same as 2-1
            key = tuple(sorted((i1, i2)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            
            v1 = verts[i1]
            v2 = verts[i2]
            mid = [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]
            verts.append(normalize(mid))
            
            idx = len(verts) - 1
            midpoint_cache[key] = idx
            return idx

        # Subdivide
        subs = self.subdivisions_input()
        for _ in range(subs):
            new_faces = []
            for tri in faces:
                v1, v2, v3 = tri
                a = get_midpoint(v1, v2)
                b = get_midpoint(v2, v3)
                c = get_midpoint(v3, v1)
                
                new_faces.append([v1, a, c])
                new_faces.append([v2, b, a])
                new_faces.append([v3, c, b])
                new_faces.append([a, b, c])
            faces = new_faces

        # Prepare Buffers
        final_vertices = []
        final_indices = []

        # For smooth shading, we need normals per vertex.
        # For a sphere, Normal == Position (normalized).
        # We can just reuse the vertex list since it's already unique and normalized.
        
        # MGLShapeNode expects interleaved [x, y, z, nx, ny, nz, u, v]
        for v in verts:
            final_vertices.extend(v) # Pos
            final_vertices.extend(v) # Normal (Same as Pos)
            
            # UV (Spherical Projection)
            # v is normalized
            u = 0.5 + math.atan2(v[2], v[0]) / (2 * math.pi)
            v_coord = 0.5 - math.asin(v[1]) / math.pi
            final_vertices.extend([u, v_coord])
            
        # Flatten indices
        for tri in faces:
            final_indices.extend(tri)

        return final_vertices, final_indices

    def handle_shape_params(self):
        if self.ctx is not None:
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                r = self.radius_input()
                scale_mat = np.diag([r, r, r, 1.0]).astype('f4')
                model = np.dot(model, scale_mat)
                self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLPlaneNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLPlaneNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.width_input = self.add_input('width', widget_type='drag_float', default_value=1.0, speed=0.01)
        self.depth_input = self.add_input('depth', widget_type='drag_float', default_value=1.0, speed=0.01)
        self.subdivisions_input = self.add_input('subdivisions', widget_type='drag_int', default_value=1, min_value=1, max_value=256, callback=self.geometry_changed)
        self.end_initialization()

    def custom_create(self, from_file):
        dpg.configure_item(self.subdivisions_input.widget.uuids[0], width=100)

    def create_geometry(self):
        vertices = []
        indices = []

        subs = max(1, self.subdivisions_input())
        verts_per_side = (subs + 1) * (subs + 1)

        # --- Top face (normal = +Y) ---
        for iz in range(subs + 1):
            for ix in range(subs + 1):
                x = -0.5 + float(ix) / subs
                z = -0.5 + float(iz) / subs
                u = float(ix) / subs
                v = float(iz) / subs
                vertices.extend([x, 0.0, z, 0.0, 1.0, 0.0, u, v])

        for iz in range(subs):
            for ix in range(subs):
                bl = iz * (subs + 1) + ix
                br = bl + 1
                tl = (iz + 1) * (subs + 1) + ix
                tr = tl + 1
                indices.extend([bl, tl, br])
                indices.extend([br, tl, tr])

        # --- Bottom face (normal = -Y, reversed winding) ---
        for iz in range(subs + 1):
            for ix in range(subs + 1):
                x = -0.5 + float(ix) / subs
                z = -0.5 + float(iz) / subs
                u = float(ix) / subs
                v = float(iz) / subs
                vertices.extend([x, 0.0, z, 0.0, -1.0, 0.0, u, v])

        for iz in range(subs):
            for ix in range(subs):
                bl = verts_per_side + iz * (subs + 1) + ix
                br = bl + 1
                tl = verts_per_side + (iz + 1) * (subs + 1) + ix
                tr = tl + 1
                # Reversed winding for -Y face
                indices.extend([bl, br, tl])
                indices.extend([br, tr, tl])

        return vertices, indices

    def handle_shape_params(self):
        if self.ctx is not None:
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                w = self.width_input()
                d = self.depth_input()
                scale_mat = np.diag([w, 1.0, d, 1.0]).astype('f4')
                model = np.dot(model, scale_mat)
                self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLDiskNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLDiskNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.radius_input = self.add_input('radius', widget_type='drag_float', default_value=0.5, speed=0.01)
        self.hole_ratio_input = self.add_input('hole_ratio', widget_type='drag_float', default_value=0.0, speed=0.01, min_value=0.0, max_value=0.99, callback=self.geometry_changed)
        self.segments_input = self.add_input('segments', widget_type='drag_int', default_value=32, min_value=3, max_value=256, callback=self.geometry_changed)
        self.rings_input = self.add_input('rings', widget_type='drag_int', default_value=1, min_value=1, max_value=64, callback=self.geometry_changed)
        self.end_initialization()

    def custom_create(self, from_file):
        dpg.configure_item(self.segments_input.widget.uuids[0], width=100)
        dpg.configure_item(self.rings_input.widget.uuids[0], width=100)

    def create_geometry(self):
        import math
        vertices = []
        indices = []

        segments = max(3, self.segments_input())
        rings = max(1, self.rings_input())
        inner_frac = max(0.0, min(self.hole_ratio_input(), 0.99))
        has_hole = inner_frac > 0.0

        def build_side(normal_y):
            base = len(vertices) // 8
            ny = normal_y

            if not has_hole:
                # Center vertex
                vertices.extend([0.0, 0.0, 0.0, 0.0, ny, 0.0, 0.5, 0.5])

                # Concentric rings from inner to outer
                for r in range(1, rings + 1):
                    frac = float(r) / rings
                    for s in range(segments):
                        theta = 2.0 * math.pi * float(s) / segments
                        x = frac * math.cos(theta)
                        z = frac * math.sin(theta)
                        u = 0.5 + 0.5 * x
                        v = 0.5 + 0.5 * z
                        vertices.extend([x, 0.0, z, 0.0, ny, 0.0, u, v])

                # Inner fan
                if ny > 0:
                    for s in range(segments):
                        s_next = (s + 1) % segments
                        indices.extend([base, base + 1 + s_next, base + 1 + s])
                else:
                    for s in range(segments):
                        s_next = (s + 1) % segments
                        indices.extend([base, base + 1 + s, base + 1 + s_next])

                # Ring quads
                for r in range(1, rings):
                    inner_base = base + 1 + (r - 1) * segments
                    outer_base = base + 1 + r * segments
                    for s in range(segments):
                        s_next = (s + 1) % segments
                        i0 = inner_base + s
                        i1 = inner_base + s_next
                        i2 = outer_base + s
                        i3 = outer_base + s_next
                        if ny > 0:
                            indices.extend([i0, i1, i2])
                            indices.extend([i1, i3, i2])
                        else:
                            indices.extend([i0, i2, i1])
                            indices.extend([i1, i2, i3])
            else:
                # Annular disk: rings+1 concentric rings from inner_frac to 1.0
                for r in range(rings + 1):
                    frac = inner_frac + (1.0 - inner_frac) * float(r) / rings
                    for s in range(segments):
                        theta = 2.0 * math.pi * float(s) / segments
                        x = frac * math.cos(theta)
                        z = frac * math.sin(theta)
                        u = 0.5 + 0.5 * x
                        v = 0.5 + 0.5 * z
                        vertices.extend([x, 0.0, z, 0.0, ny, 0.0, u, v])

                # Ring quads between concentric rings
                for r in range(rings):
                    inner_base = base + r * segments
                    outer_base = base + (r + 1) * segments
                    for s in range(segments):
                        s_next = (s + 1) % segments
                        i0 = inner_base + s
                        i1 = inner_base + s_next
                        i2 = outer_base + s
                        i3 = outer_base + s_next
                        if ny > 0:
                            indices.extend([i0, i1, i2])
                            indices.extend([i1, i3, i2])
                        else:
                            indices.extend([i0, i2, i1])
                            indices.extend([i1, i2, i3])

        # Top face (+Y)
        build_side(1.0)
        # Bottom face (-Y)
        build_side(-1.0)

        return vertices, indices

    def handle_shape_params(self):
        if self.ctx is not None:
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                r = self.radius_input()
                scale_mat = np.diag([r, 1.0, r, 1.0]).astype('f4')
                model = np.dot(model, scale_mat)
                self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLPointCloudNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLPointCloudNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.points_input = self.add_input('points', triggers_execution=True)
        self.end_initialization()
        self.points_data = None
        self.dirty = False

    def custom_create(self, from_file):
        if not from_file:
            self.mode_input.set('points')
            self.mode_input.widget.set('points')

    def execute(self):
        if self.points_input.fresh_input:
            data = self.points_input()
            if data is not None:
                # Convert to numpy float32
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                elif isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                elif self.app.torch_available and isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy().astype(np.float32)
                
                # Reshape if flat [N*3] -> [N, 3]
                if data.ndim == 1 and data.size % 3 == 0:
                    data = data.reshape(-1, 3)
                
                if data.ndim == 2 and data.shape[1] == 3:
                     self.points_data = data
                     self.dirty = True
        
        super().execute()

    def draw(self):
        if self.ctx and self.dirty and self.points_data is not None:
            # Prepare interleaved data [x,y,z, nx,ny,nz, u,v]
            count = self.points_data.shape[0]
            
            # Dummy Normals (0, 1, 0)
            normals = np.tile([0.0, 1.0, 0.0], (count, 1)).astype(np.float32)
            
            # Dummy UVs (0, 0)
            uvs = np.zeros((count, 2), dtype=np.float32)
            
            # Combine
            # [N, 8]
            vertices = np.hstack([self.points_data, normals, uvs]).flatten().astype(np.float32)
            
            # Update Geometry
            self.render_geometry(vertices, indices=None)
            self.dirty = False
            
        super().draw()

    def create_geometry(self):
        # Fallback if draw called without data
        return [], None

    def handle_shape_params(self):
        if self.ctx is not None and 'M' in self.prog:
            model = self.ctx.get_model_matrix()
            # Debug: Check translation
            # print(f"PC Model Trans: {model[3, :3]}")
            self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLSurfaceNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLSurfaceNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.points_input = self.add_input('points', triggers_execution=True)
        self.end_initialization()
        self.points_data = None
        self.dirty = False

    def execute(self):
        if self.points_input.fresh_input:
            data = self.points_input()
            if data is not None:
                # Convert to numpy float32
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                elif isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                elif self.app.torch_available and isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy().astype(np.float32)

                # Expect (rows, cols, 3)
                if data.ndim == 3 and data.shape[2] == 3:
                    self.points_data = data
                    self.dirty = True
                elif data.ndim == 2 and data.shape[1] == 3:
                    # Flat (N, 3) — treat as single row
                    self.points_data = data.reshape(1, -1, 3)
                    self.dirty = True

        super().execute()

    def compute_vertex_normals(self, grid):
        """Compute per-vertex normals by averaging adjacent face normals.
        grid: (rows, cols, 3)
        Returns: (rows, cols, 3) normals
        """
        rows, cols, _ = grid.shape
        normals = np.zeros_like(grid)

        # Compute face normals for each quad (two triangles per quad)
        # For each quad at (iy, ix), vertices are:
        #   bl = grid[iy, ix], br = grid[iy, ix+1]
        #   tl = grid[iy+1, ix], tr = grid[iy+1, ix+1]
        # Triangle 1: bl, tl, br  -> edges: tl-bl, br-bl
        # Triangle 2: br, tl, tr  -> edges: tl-br, tr-br

        for iy in range(rows - 1):
            for ix in range(cols - 1):
                bl = grid[iy, ix]
                br = grid[iy, ix + 1]
                tl = grid[iy + 1, ix]
                tr = grid[iy + 1, ix + 1]

                # Triangle 1: bl, tl, br
                n1 = np.cross(tl - bl, br - bl)
                len1 = np.linalg.norm(n1)
                if len1 > 1e-10:
                    n1 /= len1

                # Triangle 2: br, tl, tr
                n2 = np.cross(tl - br, tr - br)
                len2 = np.linalg.norm(n2)
                if len2 > 1e-10:
                    n2 /= len2

                # Accumulate to each vertex
                normals[iy, ix] += n1
                normals[iy + 1, ix] += n1 + n2
                normals[iy, ix + 1] += n1 + n2
                normals[iy + 1, ix + 1] += n2

        # Normalize
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        normals /= lengths

        return normals

    def draw(self):
        if self.ctx and self.dirty and self.points_data is not None:
            grid = self.points_data
            rows, cols, _ = grid.shape

            if rows < 2 or cols < 2:
                self.dirty = False
                super().draw()
                return

            # Compute per-vertex normals
            normals = self.compute_vertex_normals(grid)

            # Generate UVs
            us = np.linspace(0.0, 1.0, cols, dtype=np.float32)
            vs = np.linspace(0.0, 1.0, rows, dtype=np.float32)
            uu, vv = np.meshgrid(us, vs)
            uvs = np.stack([uu, vv], axis=-1)  # (rows, cols, 2)

            # --- Top face ---
            # Interleave: [x,y,z, nx,ny,nz, u,v] per vertex
            top_verts = np.concatenate([grid, normals, uvs], axis=-1)  # (rows, cols, 8)
            top_verts_flat = top_verts.reshape(-1, 8)

            # Triangle indices for top face
            top_indices = []
            for iy in range(rows - 1):
                for ix in range(cols - 1):
                    bl = iy * cols + ix
                    br = bl + 1
                    tl = (iy + 1) * cols + ix
                    tr = tl + 1
                    top_indices.extend([bl, tl, br])
                    top_indices.extend([br, tl, tr])

            # --- Bottom face (reversed winding, negated normals) ---
            neg_normals = -normals
            bot_verts = np.concatenate([grid, neg_normals, uvs], axis=-1)
            bot_verts_flat = bot_verts.reshape(-1, 8)

            bot_indices = []
            offset = rows * cols
            for iy in range(rows - 1):
                for ix in range(cols - 1):
                    bl = offset + iy * cols + ix
                    br = bl + 1
                    tl = offset + (iy + 1) * cols + ix
                    tr = tl + 1
                    # Reversed winding
                    bot_indices.extend([bl, br, tl])
                    bot_indices.extend([br, tr, tl])

            # Combine
            all_verts = np.vstack([top_verts_flat, bot_verts_flat]).flatten().astype(np.float32)
            all_indices = top_indices + bot_indices

            self.render_geometry(all_verts, all_indices)
            self.dirty = False

        super().draw()

    def create_geometry(self):
        # Fallback if draw called without data
        return [], None

    def handle_shape_params(self):
        if self.ctx is not None and 'M' in self.prog:
            model = self.ctx.get_model_matrix()
            self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLLineNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLLineNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.vector_input = self.add_input('vector', triggers_execution=True)
        self.vector_data = None
        self.vbo = None
        self.vao = None
        self.prog = None

    def execute(self):
        if self.vector_input.fresh_input:
            data = self.vector_input()
            if data is not None:
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                elif isinstance(data, np.ndarray):
                    data = data.astype(np.float32).flatten()
                elif self.app.torch_available and isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy().astype(np.float32).flatten()

                if data.size == 3:
                    self.vector_data = data
                elif data.size == 6:
                    # start + end packed as 6 values
                    self.vector_data = data

        super().execute()

    def draw(self):
        if self.ctx is not None and self.vector_data is not None:
            inner_ctx = self.ctx.ctx

            if self.vector_data.size == 3:
                # Line from origin to vector
                start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                end = self.vector_data
            else:
                # 6 values: start xyz, end xyz
                start = self.vector_data[:3]
                end = self.vector_data[3:6]

            # Direction for dummy normal
            direction = end - start
            length = np.linalg.norm(direction)
            if length > 1e-10:
                normal = direction / length
            else:
                normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            # Build two vertices: [x,y,z, nx,ny,nz, u,v]
            vertices = np.array([
                *start, *normal, 0.0, 0.0,
                *end,   *normal, 1.0, 0.0,
            ], dtype='f4')

            self.vbo = inner_ctx.buffer(vertices.tobytes())
            self.prog = self.ctx.default_shader
            self.vao = inner_ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord')])

            # Set uniforms
            if 'M' in self.prog:
                model = self.ctx.get_model_matrix()
                self.prog['M'].write(model.astype('f4').T.tobytes())
            if 'V' in self.prog:
                self.prog['V'].write(self.ctx.view_matrix.tobytes())
            if 'P' in self.prog:
                self.prog['P'].write(self.ctx.projection_matrix.tobytes())
            if 'color' in self.prog:
                c = self.ctx.current_color
                self.prog['color'].value = tuple(c)

            self.ctx.update_lights(self.prog)
            self.ctx.update_material(self.prog)

            if 'has_texture' in self.prog:
                self.prog['has_texture'].value = False

            inner_ctx.disable(moderngl.CULL_FACE)
            self.vao.render(mode=moderngl.LINES)


class MGLTextNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLTextNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.font_size = 48
        self.font_path = 'Inconsolata-g.otf'
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [float, int]:
                self.font_size = int(v)
            elif t == str:
                self.font_path = v

        self.text_input = self.add_input('text', triggers_execution=True)
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=1.0, speed=0.01)
        self.text_font = self.add_option('font', widget_type='text_input', default_value=self.font_path, callback=self.font_changed)
        self.text_size = self.add_option('size', widget_type='drag_int', default_value=self.font_size, callback=self.size_changed)

        self.face = None
        self.characters = {}
        self.glyph_shape = [0, 0]
        self.atlas_texture = None
        self.initialized = False
        self.text_data = None
        self.dirty = False
        self.vbo = None
        self.vao = None
        self.prog = None
        self.text_shader = None

    def font_changed(self):
        path = self.text_font()
        if path != self.font_path:
            self.font_path = path
            self.initialized = False
            self.dirty = True

    def size_changed(self):
        size = self.text_size()
        if size != self.font_size:
            self.font_size = size
            self.initialized = False
            self.dirty = True

    def build_atlas(self):
        """Build a glyph atlas texture using FreeType and moderngl."""
        try:
            import freetype
        except ImportError:
            print('mgl_text: freetype-py not installed')
            return False

        try:
            if self.face is not None:
                del self.face
            self.face = freetype.Face(self.font_path)
            self.face.set_char_size(int(self.font_size * 64))
        except Exception as e:
            print(f'mgl_text: failed to load font {self.font_path}: {e}')
            return False

        self.characters = {}

        # Measure glyph extents
        bottom = 1000
        top = 0
        left = 1000
        right = 0

        for i in range(32, 128):
            c = chr(i)
            if c.isprintable() or c == ' ':
                self.face.load_char(c)
                glyph = self.face.glyph
                if glyph.bitmap_top > top:
                    top = glyph.bitmap_top
                if glyph.bitmap_left < left:
                    left = glyph.bitmap_left
                bw = glyph.bitmap.width + glyph.bitmap_left
                if bw > right:
                    right = bw
                bb = glyph.bitmap_top - glyph.bitmap.rows
                if bb < bottom:
                    bottom = bb

        self.glyph_shape = [max(1, right - left), max(1, top - bottom)]
        gw, gh = self.glyph_shape
        atlas_w = gw * 16
        atlas_h = gh * 8  # 128 - 32 = 96 chars, fits in 16 x 6, use 16 x 8
        atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.float32)

        for i in range(32, 128):
            c = chr(i)
            if c.isprintable() or c == ' ':
                self.face.load_char(c)
                glyph = self.face.glyph
                bm = glyph.bitmap

                cell_x = (i - 32) % 16
                cell_y = (i - 32) // 16
                ox = cell_x * gw + glyph.bitmap_left
                oy = cell_y * gh + (gh - glyph.bitmap_top + bottom)

                for row in range(bm.rows):
                    for col in range(bm.width):
                        py = oy + row
                        px = ox + col
                        if 0 <= py < atlas_h and 0 <= px < atlas_w:
                            alpha = bm.buffer[row * bm.width + col] / 255.0
                            atlas[py, px] = [1.0, 1.0, 1.0, alpha]

                # Store texture coords and metrics
                tc_left = (cell_x * gw) / atlas_w
                tc_right = ((cell_x + 1) * gw) / atlas_w
                tc_top = (cell_y * gh) / atlas_h
                tc_bottom = ((cell_y + 1) * gh) / atlas_h

                self.characters[c] = {
                    'tc': [tc_left, tc_top, tc_right, tc_bottom],
                    'bearing': (glyph.bitmap_left, glyph.bitmap_top),
                    'advance': glyph.advance.x >> 6,
                }

        # Upload to moderngl texture
        if self.atlas_texture is not None:
            self.atlas_texture.release()

        ctx = MGLContext.get_instance().ctx
        # Flip vertically for OpenGL convention
        atlas_flipped = np.flipud(atlas).copy()
        self.atlas_texture = ctx.texture((atlas_w, atlas_h), 4, atlas_flipped.tobytes(), dtype='f4')
        self.atlas_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.initialized = True
        return True

    def execute(self):
        if self.text_input.fresh_input:
            data = self.text_input()
            if data is not None:
                if not isinstance(data, str):
                    data = str(data)
                self.text_data = data
                self.dirty = True

        super().execute()

    def get_text_shader(self):
        """Create a simple text shader that discards transparent fragments."""
        if self.text_shader is None:
            ctx = MGLContext.get_instance().ctx
            self.text_shader = ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 M;
                    uniform mat4 V;
                    uniform mat4 P;
                    in vec3 in_position;
                    in vec3 in_normal;
                    in vec2 in_texcoord;
                    out vec2 v_texcoord;
                    void main() {
                        gl_Position = P * V * M * vec4(in_position, 1.0);
                        v_texcoord = in_texcoord;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform vec4 color;
                    uniform sampler2D diffuse_map;
                    in vec2 v_texcoord;
                    out vec4 f_color;
                    void main() {
                        vec4 texColor = texture(diffuse_map, v_texcoord);
                        if (texColor.a < 0.01) discard;
                        f_color = vec4(color.rgb * texColor.a, texColor.a * color.a);
                    }
                '''
            )
        return self.text_shader

    def draw(self):
        if self.ctx is None:
            return
        if not self.initialized:
            if not self.build_atlas():
                return
            self.dirty = True

        if self.text_data is None or len(self.text_data) == 0:
            return

        inner_ctx = self.ctx.ctx
        scale = self.scale_input() / 100.0
        gw, gh = self.glyph_shape

        # Build quad vertices for each character
        vertices = []
        x_cursor = 0.0

        for c in self.text_data:
            if c not in self.characters:
                continue
            ch = self.characters[c]
            tc = ch['tc']  # [left, top, right, bottom]
            width = gw * scale
            height = gh * scale

            # Quad as two triangles in XY plane (z=0)
            x0 = x_cursor
            x1 = x_cursor + width
            y0 = 0.0
            y1 = height

            # tc: [left, top, right, bottom]
            u0, v0_top, u1, v1_bot = tc
            v0 = 1.0 - v1_bot
            v1 = 1.0 - v0_top

            nx, ny, nz = 0.0, 0.0, 1.0

            # Triangle 1: bottom-left, top-left, top-right
            vertices.extend([x0, y0, 0.0, nx, ny, nz, u0, v0])
            vertices.extend([x0, y1, 0.0, nx, ny, nz, u0, v1])
            vertices.extend([x1, y1, 0.0, nx, ny, nz, u1, v1])
            # Triangle 2: bottom-left, top-right, bottom-right
            vertices.extend([x0, y0, 0.0, nx, ny, nz, u0, v0])
            vertices.extend([x1, y1, 0.0, nx, ny, nz, u1, v1])
            vertices.extend([x1, y0, 0.0, nx, ny, nz, u1, v0])

            x_cursor += ch['advance'] * scale

        if len(vertices) == 0:
            return

        prog = self.get_text_shader()

        vert_data = np.array(vertices, dtype='f4')
        self.vbo = inner_ctx.buffer(vert_data.tobytes())
        self.vao = inner_ctx.vertex_array(prog, [(self.vbo, '3f 12x 2f', 'in_position', 'in_texcoord')])

        # Set uniforms
        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())
        if 'color' in prog:
            c = self.ctx.current_color
            prog['color'].value = tuple(c)

        # Enable texturing with the atlas
        self.atlas_texture.use(location=0)
        if 'diffuse_map' in prog:
            prog['diffuse_map'].value = 0

        # Disable culling and depth for text overlay
        inner_ctx.disable(moderngl.CULL_FACE)
        inner_ctx.disable(moderngl.DEPTH_TEST)
        self.vao.render()
        inner_ctx.enable(moderngl.DEPTH_TEST)


class MGLLineArrayNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLLineArrayNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.array_input = self.add_input('array', triggers_execution=True)
        self.line_width_input = self.add_input('line_width', widget_type='drag_float', default_value=1.0, min_value=1.0)
        self.line_array = None
        self.vbo = None
        self.vao = None
        self.prog = None
        self.line_shader = None

    def get_line_shader(self):
        """Simple unlit shader for colored lines."""
        if self.line_shader is None:
            ctx = MGLContext.get_instance().ctx
            self.line_shader = ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 M;
                    uniform mat4 V;
                    uniform mat4 P;
                    in vec3 in_position;
                    in vec4 in_color;
                    out vec4 v_color;
                    void main() {
                        gl_Position = P * V * M * vec4(in_position, 1.0);
                        v_color = in_color;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    in vec4 v_color;
                    out vec4 f_color;
                    void main() {
                        f_color = v_color;
                    }
                '''
            )
        return self.line_shader

    def execute(self):
        if self.array_input.fresh_input:
            incoming = self.array_input()
            if incoming is not None:
                if isinstance(incoming, list):
                    incoming = np.array(incoming, dtype=np.float32)
                elif isinstance(incoming, np.ndarray):
                    incoming = incoming.astype(np.float32)
                elif self.app.torch_available and isinstance(incoming, torch.Tensor):
                    incoming = incoming.detach().cpu().numpy().astype(np.float32)

                if incoming.ndim == 2 and incoming.shape[1] == 3:
                    # Single line strip: (N, 3) -> (N, 1, 3)
                    incoming = incoming[:, np.newaxis, :]
                if incoming.ndim == 3 and incoming.shape[2] == 3:
                    self.line_array = incoming

        super().execute()

    def draw(self):
        if self.ctx is None or self.line_array is None:
            return

        inner_ctx = self.ctx.ctx
        prog = self.get_line_shader()
        color = self.ctx.current_color
        num_points, num_lines, _ = self.line_array.shape

        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())

        inner_ctx.disable(moderngl.CULL_FACE)

        for i in range(num_lines):
            line = self.line_array[:, i, :]  # (N, 3)
            n = line.shape[0]
            # Build vertices with per-vertex color: [x,y,z, r,g,b,a]
            colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
            verts = np.hstack([line, colors]).flatten().astype(np.float32)

            vbo = inner_ctx.buffer(verts.tobytes())
            vao = inner_ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
            vao.render(mode=moderngl.LINE_STRIP)
            vbo.release()
            vao.release()


class MGLQuaternionRotateNode(MGLTransformNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLQuaternionRotateNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.quat_input = self.add_input('quaternion')

    def perform_transformation(self):
        if self.ctx is None:
            return
        input_ = self.quat_input()
        if input_ is None:
            return

        if isinstance(input_, list):
            input_ = np.array(input_, dtype=np.float32)
        if isinstance(input_, np.ndarray):
            input_ = input_.flatten()
            if input_.size == 4:
                # Build 4x4 rotation matrix from quaternion
                transform = quaternion_to_R3_rotation(input_)
                # quaternion_to_R3_rotation returns a flat 16-element list (row-major for legacy GL)
                mat = np.array(transform, dtype=np.float32).reshape(4, 4).T  # transpose for our column-major convention
                self.ctx.multiply_matrix(mat)


class MGLAxisAngleRotateNode(MGLTransformNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLAxisAngleRotateNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.rotvec_input = self.add_input('rotation vector')

    def perform_transformation(self):
        if self.ctx is None:
            return
        input_ = self.rotvec_input()
        if input_ is None:
            return

        if isinstance(input_, list):
            input_ = np.array(input_, dtype=np.float32)
        if isinstance(input_, np.ndarray):
            input_ = input_.flatten()
            if input_.size >= 3:
                axis = input_[:3]
                up_vector = np.array([0.0, 0.0, 1.0])
                # rotation_matrix_from_vectors returns flat 16-element array (already transposed for GL)
                alignment_flat = rotation_matrix_from_vectors(up_vector, axis)
                mat = np.array(alignment_flat, dtype=np.float32).reshape(4, 4)
                self.ctx.multiply_matrix(mat)


class MGLBillboardNode(MGLTransformNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLBillboardNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        # No extra inputs needed - billboard just cancels view rotation

    def perform_transformation(self):
        if self.ctx is None:
            return
        # Extract the rotation part of the view matrix and invert it
        # This makes child geometry always face the camera
        view = self.ctx.view_matrix
        # The upper-left 3x3 of the view matrix is the rotation
        rot3x3 = view[:3, :3].copy()
        # Inverse of a rotation matrix is its transpose
        inv_rot = np.identity(4, dtype=np.float32)
        inv_rot[:3, :3] = rot3x3.T
        self.ctx.multiply_matrix(inv_rot)


class MGLPartialDiskNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLPartialDiskNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        inner_r = 0.0
        outer_r = 0.5
        slices = 32
        start = 0.0
        sweep = 90.0

        self.outer_radius = self.add_input('outer radius', widget_type='drag_float', default_value=outer_r, callback=self.geometry_changed)
        self.inner_radius = self.add_input('inner radius', widget_type='drag_float', default_value=inner_r, callback=self.geometry_changed)
        self.slices_input = self.add_input('slices', widget_type='drag_int', default_value=slices, callback=self.geometry_changed)
        self.start_angle = self.add_input('start angle', widget_type='drag_float', default_value=start, callback=self.geometry_changed)
        self.start_angle.widget.speed = 1
        self.sweep_angle = self.add_input('sweep angle', widget_type='drag_float', default_value=sweep, callback=self.geometry_changed)
        self.sweep_angle.widget.speed = 1
        self.end_initialization()

    def create_geometry(self):
        vertices = []
        indices = []

        inner_r = max(0.0, self.inner_radius())
        outer_r = max(inner_r + 0.001, self.outer_radius())
        slices = max(3, self.slices_input())
        start_deg = self.start_angle()
        sweep_deg = self.sweep_angle()

        start_rad = math.radians(start_deg)
        sweep_rad = math.radians(sweep_deg)

        # Normal facing +Y (disk lies in XZ plane)
        nx, ny, nz = 0.0, 1.0, 0.0

        # Generate vertices: two rings (inner and outer) with slices+1 points each
        # Top face
        for i in range(slices + 1):
            angle = start_rad + sweep_rad * i / slices
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            u = i / slices

            # Inner vertex
            x_in = inner_r * cos_a
            z_in = inner_r * sin_a
            vertices.extend([x_in, 0.0, z_in, nx, ny, nz, u, 0.0])

            # Outer vertex
            x_out = outer_r * cos_a
            z_out = outer_r * sin_a
            vertices.extend([x_out, 0.0, z_out, nx, ny, nz, u, 1.0])

        # Top face indices
        for i in range(slices):
            inner_idx = i * 2
            outer_idx = i * 2 + 1
            next_inner = (i + 1) * 2
            next_outer = (i + 1) * 2 + 1
            indices.extend([inner_idx, next_inner, outer_idx])
            indices.extend([outer_idx, next_inner, next_outer])

        # Bottom face (reversed winding, negated normals)
        offset = (slices + 1) * 2
        for i in range(slices + 1):
            angle = start_rad + sweep_rad * i / slices
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            u = i / slices

            x_in = inner_r * cos_a
            z_in = inner_r * sin_a
            vertices.extend([x_in, 0.0, z_in, -nx, -ny, -nz, u, 0.0])

            x_out = outer_r * cos_a
            z_out = outer_r * sin_a
            vertices.extend([x_out, 0.0, z_out, -nx, -ny, -nz, u, 1.0])

        for i in range(slices):
            inner_idx = offset + i * 2
            outer_idx = offset + i * 2 + 1
            next_inner = offset + (i + 1) * 2
            next_outer = offset + (i + 1) * 2 + 1
            # Reversed winding
            indices.extend([inner_idx, outer_idx, next_inner])
            indices.extend([outer_idx, next_outer, next_inner])

        return vertices, indices

    def handle_shape_params(self):
        if self.ctx is not None and 'M' in self.prog:
            model = self.ctx.get_model_matrix()
            self.prog['M'].write(model.astype('f4').T.tobytes())


class MGLTorqueArcNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLTorqueArcNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.torques_input = self.add_input('torques', triggers_execution=True)
        self.positions_input = self.add_input('positions')
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=0.01, speed=0.001)
        self.threshold_input = self.add_input('threshold', widget_type='drag_float', default_value=0.1, speed=0.01)
        self.sweep_input = self.add_input('sweep', widget_type='drag_float', default_value=270.0, speed=1.0)
        self.segments_input = self.add_input('segments', widget_type='drag_int', default_value=24, min_value=6, max_value=64)
        self.arrow_size_input = self.add_input('arrow size', widget_type='drag_float', default_value=0.3, speed=0.01)

        self.torques_data = None
        self.positions_data = None
        self.arc_shader = None

    def get_arc_shader(self):
        if self.arc_shader is None:
            ctx = MGLContext.get_instance().ctx
            self.arc_shader = ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 M;
                    uniform mat4 V;
                    uniform mat4 P;
                    in vec3 in_position;
                    in vec4 in_color;
                    out vec4 v_color;
                    void main() {
                        gl_Position = P * V * M * vec4(in_position, 1.0);
                        v_color = in_color;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    in vec4 v_color;
                    out vec4 f_color;
                    void main() {
                        f_color = v_color;
                    }
                '''
            )
        return self.arc_shader

    def execute(self):
        if self.torques_input.fresh_input:
            data = self.torques_input()
            if data is not None:
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                elif isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                if data.ndim == 2 and data.shape[1] == 3:
                    self.torques_data = data

        if self.positions_input.fresh_input:
            data = self.positions_input()
            if data is not None:
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                elif isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                if data.ndim == 2 and data.shape[1] == 3:
                    self.positions_data = data

        super().execute()

    def _build_arc_geometry(self, torque_vec, position, scale, threshold, sweep_deg, segments, arrow_size, color):
        """Build arc + arrowhead vertices for a single joint torque.
        Returns list of (verts, mode) tuples where mode is moderngl.LINE_STRIP or moderngl.TRIANGLES.
        """
        magnitude = np.linalg.norm(torque_vec)
        if magnitude < threshold:
            return []

        radius = magnitude * scale
        axis = torque_vec / magnitude  # Normalized torque axis

        # Build a coordinate frame perpendicular to the torque axis
        # Choose a reference vector not parallel to axis
        if abs(axis[1]) < 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Perpendicular vectors (standard right-hand rule basis)
        u = np.cross(axis, ref)
        u /= np.linalg.norm(u) + 1e-10
        v = np.cross(axis, u)
        v /= np.linalg.norm(v) + 1e-10

        # Negate sweep to reverse arc + arrowhead direction (show resistive torque)
        sweep_rad = -math.radians(sweep_deg)
        result = []

        # --- Arc line strip ---
        arc_verts = []
        for i in range(segments + 1):
            angle = sweep_rad * i / segments
            point = position + radius * (math.cos(angle) * u + math.sin(angle) * v)
            arc_verts.extend([point[0], point[1], point[2], color[0], color[1], color[2], color[3]])
        result.append((np.array(arc_verts, dtype='f4'), moderngl.LINE_STRIP))

        # --- Arrowhead triangle at the end of the arc ---
        end_angle = sweep_rad
        end_point = position + radius * (math.cos(end_angle) * u + math.sin(end_angle) * v)

        # Tangent direction at the arc endpoint (derivative of the arc parametric equation)
        tangent = radius * (-math.sin(end_angle) * u + math.cos(end_angle) * v)
        tangent_norm = tangent / (np.linalg.norm(tangent) + 1e-10)

        # Radial direction at the endpoint (outward from center)
        radial = (end_point - position)
        radial_norm = radial / (np.linalg.norm(radial) + 1e-10)

        # Arrow size proportional to radius
        arrow_len = radius * arrow_size

        # Three points of the arrowhead triangle
        tip = end_point - tangent_norm * arrow_len
        left = end_point + radial_norm * arrow_len * 0.4
        right = end_point - radial_norm * arrow_len * 0.4

        arrow_verts = []
        for p in [tip, left, right]:
            arrow_verts.extend([p[0], p[1], p[2], color[0], color[1], color[2], color[3]])
        result.append((np.array(arrow_verts, dtype='f4'), moderngl.TRIANGLES))

        return result

    def draw(self):
        if self.ctx is None or self.torques_data is None or self.positions_data is None:
            return

        inner_ctx = self.ctx.ctx
        prog = self.get_arc_shader()
        color = self.ctx.current_color

        scale = self.scale_input()
        threshold = self.threshold_input()
        sweep_deg = self.sweep_input()
        segments = self.segments_input()
        arrow_size = self.arrow_size_input()

        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())

        inner_ctx.disable(moderngl.CULL_FACE)

        num_joints = min(self.torques_data.shape[0], self.positions_data.shape[0])

        for j in range(num_joints):
            pieces = self._build_arc_geometry(
                self.torques_data[j],
                self.positions_data[j],
                scale, threshold, sweep_deg, segments, arrow_size, color
            )
            for verts, mode in pieces:
                vbo = inner_ctx.buffer(verts.tobytes())
                vao = inner_ctx.vertex_array(prog, [(vbo, '3f 4f', 'in_position', 'in_color')])
                vao.render(mode=mode)
                vbo.release()
                vao.release()


class MGLModelNode(MGLShapeNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLModelNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.file_input = self.add_input('file_path', widget_type='text_input', default_value='', callback=self.reload_model)
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=1.0)
        self.center_input = self.add_input('center', widget_type='checkbox', default_value=True)
        
        # UV Controls
        self.uv_mode_input = self.add_input('uv_mode', widget_type='combo', default_value='original')
        self.uv_mode_input.widget.combo_items = ['original', 'sphere', 'cylinder', 'plane_xy', 'plane_xz', 'box']
        self.uv_scale_input = self.add_input('uv_scale', widget_type='drag_float', default_value=1.0)
        self.generate_uv_button = self.add_property('generate_uv', widget_type='button', callback=self.reload_model)
        
        self.loaded_geometry = None
        self.end_initialization()

    def reload_model(self):
        self.dirty = True
        self.vao = None # Force recreation
        self.vbo = None

    def create_geometry(self):
        import trimesh
        path = self.file_input()
        if not path:
            return [], None

        try:
            mesh = trimesh.load(path)
            
            # Handle Scene (multiple meshes)
            if isinstance(mesh, trimesh.Scene):
                # Concatenate all geometries in the scene
                # This works if they are compatible
                if len(mesh.geometry) > 0:
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
                else:
                    print(f"MGLModelNode: Scene at {path} is empty.")
                    return [], None

            # Process Mesh
            if self.center_input():
                mesh.vertices -= mesh.center_mass
            
            # Scale is applied in handle_shape_params via Matrix
            # scale = self.scale_input()
            # if scale != 1.0:
            #    mesh.apply_scale(scale)

            # Ensure Normals
            if mesh.vertex_normals is None or mesh.vertex_normals.shape != mesh.vertices.shape:
                mesh.compute_vertex_normals()

            # Extract UVs
            uvs = None
            uv_mode = self.uv_mode_input()
            uv_scale = self.uv_scale_input()

            if uv_mode == 'original':
                # Try to get existing UVs
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    uvs = mesh.visual.uv
                
                # If plain ColorVisuals or None, try converting/accessing
                if uvs is None:
                    try:
                        # visual.to_texture() might create dummy UVs or simple projection
                        pass 
                    except:
                        pass
                
                if uvs is not None:
                    # Match vertex count
                    if uvs.shape[0] != mesh.vertices.shape[0]:
                        print(f"MGLModelNode: UV count mismatch ({uvs.shape[0]} != {mesh.vertices.shape[0]}). Ignoring UVs.")
                        uvs = None
                
                if uvs is None:
                     print(f"MGLModelNode: No UVs found for {path}. using zeros.")
                     uvs = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)

            else:
                # Generate UVs based on geometry
                verts = mesh.vertices
                if uv_mode == 'sphere':
                    # Spherical Projection
                    # u = 0.5 + atan2(z, x) / 2pi
                    # v = 0.5 - asin(y / R) / pi
                    # Assume centered at 0,0,0 (already centered if center_input is True)
                    # Normalize first
                    norms = np.linalg.norm(verts, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    v_norm = verts / norms
                    
                    u = 0.5 + np.arctan2(v_norm[:, 2], v_norm[:, 0]) / (2 * np.pi)
                    v = 0.5 - np.arcsin(v_norm[:, 1]) / np.pi
                    uvs = np.stack([u, v], axis=1)

                elif uv_mode == 'cylinder':
                    # Cylindrical Projection
                    # u = 0.5 + atan2(z, x) / 2pi
                    # v = (y - ymin) / (ymax - ymin)
                    norms_xz = np.linalg.norm(verts[:, [0, 2]], axis=1, keepdims=True)
                    norms_xz[norms_xz == 0] = 1.0
                    
                    u = 0.5 + np.arctan2(verts[:, 2], verts[:, 0]) / (2 * np.pi)
                    
                    y = verts[:, 1]
                    ymin, ymax = y.min(), y.max()
                    h = ymax - ymin
                    if h == 0: h = 1.0
                    v = (y - ymin) / h
                    uvs = np.stack([u, v], axis=1)

                elif uv_mode == 'plane_xy':
                    # Planar XY
                    # u = (x - xmin) / w
                    # v = (y - ymin) / h
                    x = verts[:, 0]
                    y = verts[:, 1]
                    xmin, xmax = x.min(), x.max()
                    ymin, ymax = y.min(), y.max()
                    w, h = xmax - xmin, ymax - ymin
                    if w == 0: w = 1.0
                    if h == 0: h = 1.0
                    
                    u = (x - xmin) / w
                    v = (y - ymin) / h
                    uvs = np.stack([u, v], axis=1)

                elif uv_mode == 'plane_xz':
                    # Planar XZ
                    x = verts[:, 0]
                    z = verts[:, 2]
                    xmin, xmax = x.min(), x.max()
                    zmin, zmax = z.min(), z.max()
                    w, h = xmax - xmin, zmax - zmin
                    if w == 0: w = 1.0
                    if h == 0: h = 1.0
                    
                    u = (x - xmin) / w
                    v = (z - zmin) / h
                    uvs = np.stack([u, v], axis=1)
                
                elif uv_mode == 'box':
                    # Tri-planar / Cube Mapping simplified
                    # Naively project based on dominant normal?
                    # Too complex for this simple block?
                    # Let's do simple bounding box normalize for now (XYZ -> UVW)
                    # Use XY for Front/Back, XZ for Top/Bot, YZ for Left/Right?
                    # This requires per-face processing which we don't easily have here (shared vertices).
                    # Fallback to Sphere for now or just generic normalize.
                    # Let's implement normalized Position 3D (maybe for 3D textures later? No, we need 2D).
                    # Let's do a simple "Unwrapped Box" approximation -> Sphere?
                    # Let's default to Sphere for Box for now to avoid complexity.
                    print("MGLModelNode: Box mapping complex on shared vertices. Using Sphere.")
                    norms = np.linalg.norm(verts, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    v_norm = verts / norms
                    u = 0.5 + np.arctan2(v_norm[:, 2], v_norm[:, 0]) / (2 * np.pi)
                    v = 0.5 - np.arcsin(v_norm[:, 1]) / np.pi
                    uvs = np.stack([u, v], axis=1)
            
            # Apply Scale
            if uvs is not None:
                uvs = uvs.astype(np.float32) * uv_scale

            # Interleave Data [Pos, Normal, UV]
            vertices = mesh.vertices.astype(np.float32)
            normals = mesh.vertex_normals.astype(np.float32)
            
            # Combine [N, 8] -> Flatten
            vertex_data = np.hstack([vertices, normals, uvs]).flatten()
            
            # Indices
            indices = mesh.faces.flatten().astype(np.int32)
            
            print(f"MGLModelNode: Loaded {path} ({len(vertices)} verts, {len(mesh.faces)} faces)")
            return vertex_data, indices

        except Exception as e:
            print(f"MGLModelNode: Failed to load {path}: {e}")
            return [], None

    def handle_shape_params(self):
        if self.ctx is not None and 'M' in self.prog:
            model = self.ctx.get_model_matrix()
            
            # Apply dynamic scaling
            s = self.scale_input()
            if s != 1.0:
                scale_mat = np.diag([s, s, s, 1.0]).astype(np.float32)
                model = np.dot(model, scale_mat)
                
            self.prog['M'].write(model.astype('f4').T.tobytes())



class MGLCameraNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLCameraNode(name, data, args)
        
    def __init__(self, label, data, args):
        super().__init__(label, data, args)

    def initialize(self, args):
        super().initialize(args)
        self.fov = self.add_input('fov', widget_type='drag_float', default_value=60.0)
        self.pos = self.add_input('pos', widget_type='drag_float_n', default_value=[0.0, 0.0, 3.0], speed=0.1, columns=3)
        self.target = self.add_input('target', widget_type='drag_float_n', default_value=[0.0, 0.0, 0.0], speed=0.1, columns=3)
        self.up = self.add_input('up', widget_type='drag_float_n', default_value=[0.0, 1.0, 0.0], columns=3)

    def custom_create(self, from_file):
        dpg.configure_item(self.fov.widget.uuids[0], speed=1.0)
        dpg.configure_item(self.pos.widget.uuids[2], label='pos')
        dpg.configure_item(self.pos.widget.uuids[0], width=45)
        dpg.configure_item(self.pos.widget.uuids[1], width=45)
        dpg.configure_item(self.pos.widget.uuids[2], width=45)
        dpg.configure_item(self.target.widget.uuids[2], label='target')
        dpg.configure_item(self.target.widget.uuids[0], width=45)
        dpg.configure_item(self.target.widget.uuids[1], width=45)
        dpg.configure_item(self.target.widget.uuids[2], width=45)
        dpg.configure_item(self.up.widget.uuids[2], label='up')
        dpg.configure_item(self.up.widget.uuids[0], width=45)
        dpg.configure_item(self.up.widget.uuids[1], width=45)
        dpg.configure_item(self.up.widget.uuids[2], width=45)

    def draw(self):
        # Set View and Projection
        if self.ctx is not None:

            # Projection
            aspect = self.ctx.width / self.ctx.height
            if aspect == 0: aspect = 1.0
            p = perspective(self.fov(), aspect, 0.1, 100.0)
            self.ctx.set_projection_matrix(p)

            # View
            v = look_at(self.pos(), self.target(), self.up())
            self.ctx.set_view_matrix(v)

            # Update view_pos in default shader if available
            if 'view_pos' in self.ctx.default_shader:
                self.ctx.default_shader['view_pos'].value = tuple(self.pos())

            # Just pass signal? Actually Camera should usually be BEFORE transform/geometry
            # in the chain for View/Proj to be set for them.
            pass

from dpg_system.mgl_body_node import MGLBodyNode
from dpg_system.mgl_smpl_mesh_node import MGLSMPLMeshNode
from dpg_system.mgl_smpl_heatmap_node import MGLSMPLHeatmapNode
