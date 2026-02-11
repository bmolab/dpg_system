
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
from dpg_system.body_base import BodyData, t_PelvisAnchor, t_ActiveJointCount
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
    Node.app.register_node('mgl_body', MGLBodyNode.factory)


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

        self.render_trigger = self.add_input('render', triggers_execution=True)
        self.mgl_chain_output = self.add_output('mgl_chain')
        self.texture_output = self.add_output('texture_tag')

        self.width_option = self.add_option('width', widget_type='drag_int', default_value=self.width, callback=self.resize)
        self.height_option = self.add_option('height', widget_type='drag_int', default_value=self.height, callback=self.resize)
        self.display_mode_option = self.add_option('display_mode', widget_type='combo', default_value='node')
        self.display_mode_option.widget.combo_items = ['off', 'node', 'window', 'fullscreen']
        
        self.samples_option = self.add_option('samples', widget_type='combo', default_value='4')
        self.samples_option.widget.combo_items = ['0', '2', '4', '6', '8', '16']
        

    def custom_cleanup(self):
        if self.texture_tag:
            dpg.delete_item(self.texture_tag)
        if self.external_window:
            dpg.delete_item(self.external_window)
        if self.fullscreen_window:
            dpg.delete_item(self.fullscreen_window)
        if hasattr(self, 'render_target') and self.render_target:
            self.render_target.release()

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
        
        if self.render_target is None or self.render_target.width != self.width or self.render_target.height != self.height or self.render_target.samples != samples:
            if self.render_target:
                self.render_target.release()
            self.render_target = self.context.create_render_target(self.width, self.height, samples)

        # Scope the ModernGL Context to avoid polluting global GL state (e.g. for legacy gl_nodes)
        with self.context.ctx:
            # Activate Local Target
            self.context.use_render_target(self.render_target)
            
            # Reset Context State
            self.context.current_color = (1.0, 1.0, 1.0, 1.0)
            self.context.lights = []
            self.context.current_material = {
                'ambient': [0.1, 0.1, 0.1],
                'diffuse': [1.0, 1.0, 1.0],
                'specular': [0.5, 0.5, 0.5],
                'shininess': 32.0
            }
            
            # Sync back actual samples if fallback occurred
            if self.render_target.samples != samples:
                self.samples_option.set(str(self.render_target.samples))
            
            # Clear
            self.context.clear(0.0, 0.0, 0.0, 1.0)
            
            # Signal Downstream to Draw
            self.mgl_chain_output.send('draw')
            
            # Read Pixels and Update DPG Texture
            data = self.context.get_pixel_data()
        
        # 1. Create Texture if needed
        pixels = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
        
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
                print(f"MGLBodyNode: Lights={len(self.ctx.lights)}, Mat={self.ctx.current_material['diffuse']}")
                print(f"Shader has num_lights? {'num_lights' in self.prog}")
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

from dpg_system.body_base import BodyData, t_PelvisAnchor, t_ActiveJointCount
from dpg_system.body_defs import *

class MGLBodyNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLBodyNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.body = BodyData()
        self.body.node = self
        
        # Manually initialize joint hierarchy because standard BodyData lacks connect_limbs
        self.build_hierarchy(self.body.joints)
        self.body.create_limbs()

        self.pose_input = self.add_input('pose', triggers_execution=True)

        self.enable_callbacks_input = self.add_input('enable_callbacks', widget_type='checkbox', default_value=True)
        self.display_mode_input = self.add_input('display_mode', widget_type='combo', default_value='lines')
        self.display_mode_input.widget.combo_items = ['lines', 'solid']
        
        self.scale_input = self.add_input('scale', widget_type='drag_float', default_value=1.0)
        self.joint_radius_input = self.add_input('joint_radius', widget_type='drag_float', default_value=0.1)
        self.joint_data_input = self.add_input('joint_data')
        self.color_input = self.add_input('color', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0])
        self.instanced_display_mode_input = self.add_input('instanced_mode', widget_type='combo', default_value='solid')
        self.instanced_display_mode_input.widget.combo_items = ['solid', 'wireframe', 'points']
        
        self.joint_callback = self.add_output('joint_callback')
        self.joint_id_out = self.add_output('joint_id')

    def initialize_instanced_gl(self):
        ctx = self.ctx.ctx
        # 1. Sphere Geometry (Unit Sphere)
        # Reusing similar logic to MGLSphereNode but simplified
        sphere_res = 16
        # Parametric sphere generation... or just reuse cube for testing?
        # Let's generate a proper sphere
        
        verts = []
        norms = []
        indices = []
        
        import math
        
        for i in range(sphere_res + 1):
            lat = math.pi * i / sphere_res
            for j in range(sphere_res + 1):
                lon = 2 * math.pi * j / sphere_res
                x = math.sin(lat) * math.cos(lon)
                y = math.sin(lat) * math.sin(lon)
                z = math.cos(lat)
                verts.extend([x, y, z])
                norms.extend([x, y, z])
                
        for i in range(sphere_res):
            for j in range(sphere_res):
                p1 = i * (sphere_res + 1) + j
                p2 = p1 + 1
                p3 = (i + 1) * (sphere_res + 1) + j
                p4 = p3 + 1
                indices.extend([p1, p2, p3, p2, p4, p3])
        
        self.instanced_sphere_vbo = ctx.buffer(np.array(verts, dtype='f4').tobytes())
        self.instanced_sphere_ibo = ctx.buffer(np.array(indices, dtype='i4').tobytes())
        self.instanced_sphere_norms = ctx.buffer(np.array(norms, dtype='f4').tobytes())
        
        # 2. Instance Data Buffers (Dynamic)
        # We need to pre-calculate the "valid" bone indices to skip duplicates
        valid_indices = []
        for i, joint in enumerate(self.body.joints):
             if i in [t_PelvisAnchor, t_LeftHip, t_RightHip, t_LeftShoulderBladeBase, t_RightShoulderBladeBase]:
                continue
             if joint is None:
                continue
             valid_indices.append(i)
        
        self.instanced_bone_indices = np.array(valid_indices, dtype='i4')
        # Buffer for bone indices (static enough unless body changes)
        self.instanced_bone_idx_buffer = ctx.buffer(self.instanced_bone_indices.tobytes())
        
        # Buffer for radii (dynamic) - Initial size matching valid count
        self.instanced_radius_buffer = ctx.buffer(reserve=len(valid_indices) * 4) # float32
        
        # 3. VAO
        # Attributes: in_pos, in_norm, in_radius, in_bone_idx
        # Radius and BoneIdx are per-instance (divisor=1)
        
        self.instanced_vao = ctx.vertex_array(self.ctx.mgl_instanced_joint_shader, [
            (self.instanced_sphere_vbo, '3f', 'in_pos'),
            (self.instanced_sphere_norms, '3f', 'in_norm'),
            (self.instanced_radius_buffer, '1f/i', 'in_radius'),
            (self.instanced_bone_idx_buffer, '1i/i', 'in_bone_idx'),
        ], index_buffer=self.instanced_sphere_ibo)
        
    def draw_instanced(self, joint_data):
        # DEBUG PRINTS FOR USER
        if not hasattr(self, 'instanced_vao'):
            # print("MGLBodyNode: No instanced_vao in draw_instanced - skipping")
            return
            
        ctx = self.ctx.ctx
        num_joints = len(self.body.joints)
        
        # Robust Data Handling
        if not isinstance(joint_data, np.ndarray):
            joint_data = np.array(joint_data, dtype=np.float32)
            
        joint_data = joint_data.flatten()
        
        # DEBUG
        # print(f"MGLBodyNode: draw_instanced entered. Data len: {len(joint_data)}/{num_joints}. Global Mats: {len(self.global_matrices)}")
        
        # Ensure data size matches full body first
        if len(joint_data) < num_joints:
             # Pad with zeros
             padded = np.zeros(num_joints, dtype=np.float32)
             padded[:len(joint_data)] = joint_data
             full_data = padded
        else:
             full_data = joint_data[:num_joints].astype(np.float32)

        # Filter to Valid Indices (Remove duplicates)
        # self.instanced_bone_indices contains indices of joints we actually draw
        # We need to extract data for these indices
        valid_data = full_data[self.instanced_bone_indices]
        
        # Apply base radius multiplier
        base_r = self.joint_radius_input()
        if base_r is None: base_r = 0.1 # Default or prevent crash
        
        # Multiply valid data by base radius
        scaled_data = valid_data * base_r
        
        self.instanced_radius_buffer.write(scaled_data.tobytes())
        
        # Setup Shader Uniforms (VP)
        if 'VP' in self.ctx.mgl_instanced_joint_shader:
            # Match MGLBodyNode.draw logic: View @ Projection likely
            # Though standard is Projection @ View, it depends on conventions
            # Let's align with working limb code
            vp = self.ctx.view_matrix @ self.ctx.projection_matrix
            self.ctx.mgl_instanced_joint_shader['VP'].write(vp.flatten().tobytes())
            
        # Upload Bones to Shader
        max_bones = 50
        bones_bytes = bytearray(max_bones * 64) 
        
        for i in range(min(num_joints, max_bones)):
            if i in self.global_matrices:
                mat = self.global_matrices[i]
                # Transpose for OpenGL (Column-Major)
                bones_bytes[i*64 : (i+1)*64] = mat.T.astype('f4').flatten().tobytes()
        
        if 'bones' in self.ctx.mgl_instanced_joint_shader:
            self.ctx.mgl_instanced_joint_shader['bones'].write(bones_bytes)
            
        # Note: Shader does NOT have base_radius uniform, so we handle it on CPU above.
        
        
        # Set Color
        # Use node input if available, otherwise context
        sphere_color = self.color_input()
        if sphere_color is None:
             sphere_color = self.ctx.current_color
             
        # Normalize colors (handle 0-255 vs 0-1)
        # Identical logic to MGLMaterialNode.norm(c)
        def norm(c):
             if c is None or len(c) == 0: return [1.0, 1.0, 1.0, 1.0] # Fallback
             
             # Handle list/tuple to list
             c_list = list(c)
             
             if max(c_list) > 1.0:
                 c_list = [val / 255.0 for val in c_list]
                 
             # Ensure 4 components (defaults to alpha 1.0 if missing)
             if len(c_list) == 3:
                 return c_list + [1.0]
             return c_list
             
        final_color = norm(sphere_color)
        
        if 'color' in self.ctx.mgl_instanced_joint_shader:
            self.ctx.mgl_instanced_joint_shader['color'].value = tuple(final_color)
             
        # Handle Display Mode
        mode = self.instanced_display_mode_input()
        
        # Disable culling
        self.ctx.ctx.disable(moderngl.CULL_FACE) 
        # Enable Standard Alpha Blending
        self.ctx.ctx.enable(moderngl.BLEND)
        self.ctx.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        count = len(scaled_data)
        
        if mode == 'wireframe':
            self.ctx.ctx.wireframe = True
            self.instanced_vao.render(instances=count)
            self.ctx.ctx.wireframe = False
        elif mode == 'points':
            self.instanced_vao.render(instances=count, mode=moderngl.POINTS)
        else:
            self.instanced_vao.render(instances=count)
            
        self.ctx.ctx.disable(moderngl.BLEND)
        self.ctx.ctx.enable(moderngl.CULL_FACE)
        
    def build_hierarchy(self, joints):
        # Manually connect limbs (Hierarchy for traversal)
        # Based on AlternateBodyData.connect_humanoid
        
        # Initialize empty lists first
        for joint in joints:
            joint.immed_children = []
            
        # Helper to safely append
        def add_child(parent_idx, child_idx):
            if 0 <= parent_idx < len(joints) and 0 <= child_idx < len(joints):
                joints[parent_idx].immed_children.append(child_idx)

        # Spine
        add_child(t_PelvisAnchor, t_SpinePelvis)
        add_child(t_SpinePelvis, t_LowerVertebrae)
        add_child(t_LowerVertebrae, t_MidVertebrae)
        add_child(t_MidVertebrae, t_UpperVertebrae)
        add_child(t_UpperVertebrae, t_BaseOfSkull)
        # Connect extremities (Indices 20+)
        add_child(t_BaseOfSkull, t_TopOfHead) 
        
        # Legs
        add_child(t_PelvisAnchor, t_LeftHip)
        add_child(t_PelvisAnchor, t_RightHip)
        
        add_child(t_LeftHip, t_LeftKnee)
        add_child(t_LeftKnee, t_LeftAnkle)
        add_child(t_LeftAnkle, t_LeftBallOfFoot)
        if t_LeftHeel < len(joints):
            add_child(t_LeftAnkle, t_LeftHeel)
        if t_LeftBallOfFoot < len(joints) and t_LeftToeTip < len(joints):
             add_child(t_LeftBallOfFoot, t_LeftToeTip)
        
        add_child(t_RightHip, t_RightKnee)
        add_child(t_RightKnee, t_RightAnkle)
        add_child(t_RightAnkle, t_RightBallOfFoot)
        if t_RightHeel < len(joints):
            add_child(t_RightAnkle, t_RightHeel)
        if t_RightBallOfFoot < len(joints) and t_RightToeTip < len(joints):
             add_child(t_RightBallOfFoot, t_RightToeTip)
        
        # Arms (Collar/ShoulderBlade attached to MidVertebrae usually, or Spine?)
        # AlternateBodyData: MidVertebrae -> ShoulderBlade -> Shoulder
        add_child(t_MidVertebrae, t_LeftShoulderBladeBase)
        add_child(t_LeftShoulderBladeBase, t_LeftShoulder)
        add_child(t_LeftShoulder, t_LeftElbow)
        add_child(t_LeftElbow, t_LeftWrist)
        add_child(t_LeftWrist, t_LeftKnuckle)
        if t_LeftKnuckle < len(joints) and t_LeftFingerTip < len(joints):
             add_child(t_LeftKnuckle, t_LeftFingerTip)
        
        add_child(t_MidVertebrae, t_RightShoulderBladeBase)
        add_child(t_RightShoulderBladeBase, t_RightShoulder)
        add_child(t_RightShoulder, t_RightElbow)
        add_child(t_RightElbow, t_RightWrist)
        add_child(t_RightWrist, t_RightKnuckle)
        if t_RightKnuckle < len(joints) and t_RightFingerTip < len(joints):
             add_child(t_RightKnuckle, t_RightFingerTip)
        
        self.cube_vbo = None
        self.cube_vao = None
        self.prog = None
        self.instance_vbo = None
        self.max_limbs = 64

    def initialize_gl(self):
        if self.ctx.ctx is None: return
        
        # Simple Cube Geometry (centered at 0,0,0, size 1x1x1)
        # But body_base cubes are complex. Let's use simple unit cube valid for scaling.
        vertices = np.array([
            # Front
            -0.5, -0.5, 0.5, 0, 0, 1, 0, 0,
             0.5, -0.5, 0.5, 0, 0, 1, 1, 0,
             0.5,  0.5, 0.5, 0, 0, 1, 1, 1,
            -0.5,  0.5, 0.5, 0, 0, 1, 0, 1,
            # Back
            -0.5, -0.5, -0.5, 0, 0, -1, 0, 0,
             0.5, -0.5, -0.5, 0, 0, -1, 1, 0,
             0.5,  0.5, -0.5, 0, 0, -1, 1, 1,
            -0.5,  0.5, -0.5, 0, 0, -1, 0, 1,
            # Top
            -0.5,  0.5, -0.5, 0, 1, 0, 0, 0,
            -0.5,  0.5,  0.5, 0, 1, 0, 0, 1,
             0.5,  0.5,  0.5, 0, 1, 0, 1, 1,
             0.5,  0.5, -0.5, 0, 1, 0, 1, 0,
            # Bottom
            -0.5, -0.5, -0.5, 0, -1, 0, 0, 0,
             0.5, -0.5, -0.5, 0, -1, 0, 1, 0,
             0.5, -0.5,  0.5, 0, -1, 0, 1, 1,
            -0.5, -0.5,  0.5, 0, -1, 0, 0, 1,
            # Right
             0.5, -0.5, -0.5, 1, 0, 0, 0, 0,
             0.5,  0.5, -0.5, 1, 0, 0, 1, 0,
             0.5,  0.5,  0.5, 1, 0, 0, 1, 1,
             0.5, -0.5,  0.5, 1, 0, 0, 0, 1,
            # Left
            -0.5, -0.5, -0.5, -1, 0, 0, 0, 0,
            -0.5, -0.5,  0.5, -1, 0, 0, 1, 0,
            -0.5,  0.5,  0.5, -1, 0, 0, 1, 1,
            -0.5,  0.5, -0.5, -1, 0, 0, 0, 1,
        ], dtype='f4')
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 7, 6, 6, 5, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ], dtype='i4')

        ctx = self.ctx.ctx
        self.cube_vbo = ctx.buffer(vertices)
        self.cube_ibo = ctx.buffer(indices)
        
        # Shader for Skinned Rendering (Batched Limbs)
        self.prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 VP; // ViewProjection
                
                #define MAX_BONES 50
                uniform mat4 bones[MAX_BONES];
                
                in vec3 in_pos;
                in vec3 in_norm;
                in float in_bone_index;
                
                // Material Color
                uniform vec4 color;
                
                out vec3 v_norm;
                out vec3 v_pos;
                out vec4 v_color;
                
                void main() {
                    int b_idx = int(in_bone_index);
                    mat4 bone_mat = bones[b_idx];
                    
                    vec4 world_pos = bone_mat * vec4(in_pos, 1.0);
                    gl_Position = VP * world_pos;
                    v_pos = world_pos.xyz;
                    
                    // Normal Matrix
                    mat3 normal_matrix = transpose(inverse(mat3(bone_mat)));
                    v_norm = normal_matrix * in_norm;
                    
                    v_color = color; 
                }
            ''',
            fragment_shader='''
                #version 330
                #define MAX_LIGHTS 8
                
                in vec3 v_norm;
                in vec3 v_pos;
                in vec4 v_color;
                
                uniform vec3 view_pos;
                
                // Lighting Uniforms
                struct Light {
                    vec3 pos;
                    vec3 ambient;
                    vec3 diffuse;
                    vec3 specular;
                    float intensity;
                };
                
                uniform int num_lights;
                uniform Light lights[MAX_LIGHTS];
                
                // Material Uniforms
                uniform vec4 material_ambient;
                uniform vec4 material_diffuse;
                uniform vec4 material_specular;
                uniform float material_shininess;
                
                out vec4 f_color;
                
                void main() {
                    vec3 norm = normalize(v_norm);
                    vec3 viewDir = normalize(view_pos - v_pos);
                    vec3 result = vec3(0.0);
                    
                    // If no lights, use simple ambient fallback based on material
                    if (num_lights == 0) {
                        result = material_ambient.rgb * v_color.rgb + material_diffuse.rgb * v_color.rgb * 0.5;
                    } else {
                        vec3 total_ambient = vec3(0.0);
                        vec3 total_diffuse = vec3(0.0);
                        vec3 total_specular = vec3(0.0);
                        
                        for(int i = 0; i < MAX_LIGHTS; i++) {
                            if (i >= num_lights) break;
                            
                            // Ambient
                            total_ambient += lights[i].ambient * material_ambient.rgb;
                            
                            // Diffuse
                            vec3 lightDir = normalize(lights[i].pos - v_pos);
                            float diff = max(dot(norm, lightDir), 0.0);
                            
                            total_diffuse += diff * lights[i].diffuse * material_diffuse.rgb * lights[i].intensity;
                            
                            // Specular (Blinn-Phong)
                            vec3 halfwayDir = normalize(lightDir + viewDir);
                            float spec = pow(max(dot(norm, halfwayDir), 0.0), material_shininess);
                            total_specular += spec * lights[i].specular * material_specular.rgb * lights[i].intensity;
                        }
                        
                        // Premultiplied Alpha Output
                        // Body Color = (Ambient + Diffuse) * Alpha
                        // Specular = Specular (Additive)
                        float alpha = v_color.a * material_diffuse.a * material_ambient.a;
                        result = (total_ambient + total_diffuse) * v_color.rgb * alpha + total_specular;
                        
                        f_color = vec4(result, alpha);
                    }
                }
            '''
        )
        
        # Build Limb Geometry
        all_verts = []
        all_norms = []
        all_indices = []
        
        # Skeleton Lines
        line_verts = []
        line_norms = []
        line_indices = []
        line_bone_indices = []
        line_vert_count = 0
        
        def_points = [
            (-.5, -.5, 0), (-1.0, -1.0, 0.5), (.5, -.5, 0), (1.0, -1.0, 0.5),
            (.5, .5, 0), (1.0, 1.0, 0.5), (-.5, .5, 0),(-1.0, 1.0, .5),
            (-.5, -.5, 1.0), (.5, -.5, 1.0), (.5, .5, 1.0), (-.5, .5, 1.0)
        ]

        for i in range(len(self.body.joints)):
            if i == t_PelvisAnchor: continue
            if self.body.joints[i] is None or not self.body.joints[i].do_draw:
                continue
                
            pts = def_points
            if i < len(self.body.limb_vertices) and self.body.limb_vertices[i] is not None:
                pts = self.body.limb_vertices[i]
            
            # Simple scaling logic based on dims or defaults
            scale_vec = np.array([0.05, 0.05, 1.0])
            if i < len(self.body.joints) and self.body.joints[i] is not None:
                j = self.body.joints[i]
                if hasattr(j, 'dims'):
                     dz = j.dims[0]
                     dx = j.dims[1] / 2.0
                     dy = j.dims[2] / 2.0
                     scale_vec = np.array([dx, dy, dz])

            def add_tri(p1, p2, p3, n):
                all_verts.extend(p1); all_norms.extend(n); all_indices.append(float(i))
                all_verts.extend(p2); all_norms.extend(n); all_indices.append(float(i))
                all_verts.extend(p3); all_norms.extend(n); all_indices.append(float(i))

            def calc_n(p1, p2, p3):
                v1 = np.array(p2) - np.array(p1)
                v2 = np.array(p3) - np.array(p1)
                c = np.cross(v2, v1)
                l = np.linalg.norm(c)
                return c / l if l > 0 else c
                
            # Add Skeleton Line Segment (0,0,0 -> 0,0,dz)
            l_start = [0.0, 0.0, 0.0]
            l_end = [0.0, 0.0, scale_vec[2]]
            line_verts.extend(l_start); line_verts.extend(l_end)
            line_norms.extend([0,0,1]); line_norms.extend([0,0,1]) # Dummy normals
            line_bone_indices.append(float(i)); line_bone_indices.append(float(i))
            line_indices.append(line_vert_count); line_indices.append(line_vert_count+1)
            line_vert_count += 2
            
            # Apply scale FIRST
            pts_np = [np.array(p) * scale_vec for p in pts]
            
            if len(pts) == 8:
                n0 = calc_n(pts_np[0], pts_np[1], pts_np[2])
                add_tri(pts_np[0], pts_np[1], pts_np[2], n0)
                add_tri(pts_np[2], pts_np[1], pts_np[3], n0)
                n1 = calc_n(pts_np[2], pts_np[3], pts_np[4])
                add_tri(pts_np[2], pts_np[3], pts_np[4], n1)
                add_tri(pts_np[4], pts_np[3], pts_np[5], n1)
                n2 = calc_n(pts_np[4], pts_np[5], pts_np[6])
                add_tri(pts_np[4], pts_np[5], pts_np[6], n2)
                add_tri(pts_np[6], pts_np[5], pts_np[7], n2)
                n3 = calc_n(pts_np[6], pts_np[7], pts_np[0])
                add_tri(pts_np[6], pts_np[7], pts_np[0], n3)
                add_tri(pts_np[0], pts_np[7], pts_np[1], n3)
                n4 = calc_n(pts_np[1], pts_np[3], pts_np[7])
                add_tri(pts_np[1], pts_np[3], pts_np[7], n4); add_tri(pts_np[7], pts_np[3], pts_np[5], n4)
                n5 = calc_n(pts_np[0], pts_np[2], pts_np[6])
                add_tri(pts_np[0], pts_np[2], pts_np[6], n5); add_tri(pts_np[6], pts_np[2], pts_np[4], n5)
            else:
                n = calc_n(pts_np[0], pts_np[1], pts_np[2])
                add_tri(pts_np[0], pts_np[1], pts_np[2], n); add_tri(pts_np[2], pts_np[1], pts_np[3], n)
                n = calc_n(pts_np[2], pts_np[3], pts_np[4])
                add_tri(pts_np[2], pts_np[3], pts_np[4], n); add_tri(pts_np[4], pts_np[3], pts_np[5], n)
                n = calc_n(pts_np[4], pts_np[5], pts_np[6])
                add_tri(pts_np[4], pts_np[5], pts_np[6], n); add_tri(pts_np[6], pts_np[5], pts_np[7], n)
                n = calc_n(pts_np[6], pts_np[7], pts_np[0])
                add_tri(pts_np[6], pts_np[7], pts_np[0], n); add_tri(pts_np[0], pts_np[7], pts_np[1], n)
                n = calc_n(pts_np[1], pts_np[8], pts_np[3])
                add_tri(pts_np[1], pts_np[8], pts_np[3], n); add_tri(pts_np[3], pts_np[8], pts_np[9], n)
                n = calc_n(pts_np[3], pts_np[9], pts_np[5])
                add_tri(pts_np[3], pts_np[9], pts_np[5], n); add_tri(pts_np[5], pts_np[9], pts_np[10], n)
                n = calc_n(pts_np[5], pts_np[10], pts_np[7])
                add_tri(pts_np[5], pts_np[10], pts_np[7], n); add_tri(pts_np[7], pts_np[10], pts_np[11], n)
                n = calc_n(pts_np[7], pts_np[11], pts_np[1])
                add_tri(pts_np[7], pts_np[11], pts_np[1], n); add_tri(pts_np[1], pts_np[11], pts_np[8], n)
                # TOP CAP Normal (should be +Z/Outward, but standard calc gives -Z/Inward)
                # So we swap arguments to flip it back
                n = calc_n(pts_np[8], pts_np[11], pts_np[9])
                add_tri(pts_np[8], pts_np[9], pts_np[11], n); add_tri(pts_np[11], pts_np[9], pts_np[10], n)
                n = calc_n(pts_np[0], pts_np[2], pts_np[6])
                add_tri(pts_np[0], pts_np[2], pts_np[6], n); add_tri(pts_np[6], pts_np[2], pts_np[4], n)

        
        v_data = np.array(all_verts, dtype='f4')
        n_data = np.array(all_norms, dtype='f4')
        i_data = np.array(all_indices, dtype='f4')
        
        self.limb_top_fix_vbo = ctx.buffer(np.hstack([v_data.reshape(-1,3), n_data.reshape(-1,3), i_data.reshape(-1,1)]).flatten().tobytes())
        self.limb_top_fix_vao = ctx.vertex_array(self.prog, [
            (self.limb_top_fix_vbo, '3f 3f 1f', 'in_pos', 'in_norm', 'in_bone_index')
        ])

        # Create Skeleton VBO/VAO
        l_v_data = np.array(line_verts, dtype='f4')
        l_n_data = np.array(line_norms, dtype='f4')
        l_b_data = np.array(line_bone_indices, dtype='f4')
        l_i_data = np.array(line_indices, dtype='i4') # Indices are int
        
        # Interleave
        # buffer data: pos(3), norm(3), bone(1)
        self.skeleton_vbo = ctx.buffer(np.hstack([l_v_data.reshape(-1,3), l_n_data.reshape(-1,3), l_b_data.reshape(-1,1)]).flatten().tobytes())
        self.skeleton_index_buffer = ctx.buffer(l_i_data.tobytes())
        
        self.skeleton_vao = ctx.vertex_array(self.prog, [
            (self.skeleton_vbo, '3f 3f 1f', 'in_pos', 'in_norm', 'in_bone_index')
        ], index_buffer=self.skeleton_index_buffer)

        # Unlit Shader for Optimized Lines
        self.unlit_prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 VP;
                #define MAX_BONES 50
                uniform mat4 bones[MAX_BONES];
                in vec3 in_pos;
                // Skipped Normals
                in float in_bone_index;
                uniform vec4 color;
                out vec4 v_color;
                void main() {
                    int b_idx = int(in_bone_index);
                    mat4 bone_mat = bones[b_idx];
                    vec4 world_pos = bone_mat * vec4(in_pos, 1.0);
                    gl_Position = VP * world_pos;
                    v_color = color; 
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
        self.skeleton_unlit_vao = ctx.vertex_array(self.unlit_prog, [
            (self.skeleton_vbo, '3f 12x 1f', 'in_pos', 'in_bone_index')
        ], index_buffer=self.skeleton_index_buffer)

        # Initialize Instanced Geometry (Spheres)
        self.initialize_instanced_gl()

    def execute(self):
        if self.pose_input.fresh_input:
            data = self.pose_input()
            if data is not None:
                self.body.update_quats(data)

        if self.joint_data_input.fresh_input:
            self.latest_joint_data = self.joint_data_input()
    
        if self.mgl_input.fresh_input:
            msg = self.mgl_input()
            if msg == 'draw':
                self.ctx = MGLContext.get_instance()
                self.draw()
                self.ctx = None
            self.mgl_output.send(msg)

    def draw(self):
        if not self.ctx or not self.ctx.ctx: return
        
        # 5. Instanced Visualization (Optional) - MOVED TO END
        # if hasattr(self, 'latest_joint_data') and self.latest_joint_data is not None:
        #      self.draw_instanced(self.latest_joint_data)
        
        if not getattr(self, 'limb_top_fix_vao', None): 
             print("MGLBodyNode: initializing GL for top cap normal fix")
             self.initialize_gl()

        
        # 1. Compute Global Matrices for all joints
        self.global_matrices = {}
        
        if len(self.ctx.model_matrix_stack) > 0:
            world_mat = self.ctx.model_matrix_stack[-1]
        else:
            world_mat = np.identity(4, dtype=np.float32)

        s = self.scale_input()
        scale_mat = np.diag([s, s, s, 1.0])
        rot_flip = rotation_matrix(180, [0, 0, 1]).T
        
        root_mat = world_mat @ rot_flip @ scale_mat
        
        self.traverse_matrices(t_PelvisAnchor, root_mat)
        
        # 2. Upload Bone Matrices
        bones_bytes = bytearray()
        MAX_BONES = 50
        
        for i in range(MAX_BONES):
            if i in self.global_matrices:
                m = self.global_matrices[i]
            else:
                m = np.identity(4, dtype='f4')
            bones_bytes.extend(m.T.astype('f4').flatten().tobytes())

        # Determine Render Mode
        mode = self.display_mode_input()
        use_unlit = (mode == 'lines')
        
        # Select Program
        target_prog = self.unlit_prog if use_unlit and hasattr(self, 'unlit_prog') else self.prog
            
        if 'bones' in target_prog:
            target_prog['bones'].write(bones_bytes)
            
        # 3. Uniforms
        vp = self.ctx.view_matrix @ self.ctx.projection_matrix
        if 'VP' in target_prog:
            target_prog['VP'].write(vp.flatten().tobytes())
        
        # Lit-only uniforms
        if not use_unlit:
            try:
                inv_view = np.linalg.inv(self.ctx.view_matrix)
                cam_pos = inv_view[3, :3]
                if 'view_pos' in target_prog:
                    target_prog['view_pos'].value = tuple(cam_pos)
            except:
                pass
            
            self.ctx.update_lights(target_prog)
            self.ctx.update_material(target_prog)

        if 'color' in target_prog:
            c = self.ctx.current_color
            target_prog['color'].value = tuple(c)
        
        # 4. Draw
        if use_unlit:
            if hasattr(self, 'skeleton_unlit_vao'):
                self.ctx.ctx.disable(moderngl.CULL_FACE)
                self.skeleton_unlit_vao.render(mode=moderngl.LINES)
                self.ctx.ctx.enable(moderngl.CULL_FACE)
        else:
            if getattr(self, 'limb_top_fix_vao', None):
                self.ctx.ctx.disable(moderngl.CULL_FACE)
                self.limb_top_fix_vao.render(mode=moderngl.TRIANGLES)
                self.ctx.ctx.enable(moderngl.CULL_FACE)

        # 5. Joint Callbacks (Optional)
        if self.enable_callbacks_input():
            # Iterate all joints and trigger callback with their global matrix
            for i, joint in enumerate(self.body.joints):
                # Filter duplicates (e.g. 4=6=9 and 1=12=16 are coincident)
                if i in [t_PelvisAnchor, t_LeftHip, t_RightHip, t_LeftShoulderBladeBase, t_RightShoulderBladeBase]:
                    continue
                if joint is None:
                    continue
                
                if i in self.global_matrices:
                    if not hasattr(joint, 'do_draw') or joint.do_draw:
                        mat = self.global_matrices[i]
                        self.ctx.model_matrix_stack.append(mat)
                        self.joint_id_out.send(i)
                        self.joint_callback.send('draw')
                        self.ctx.model_matrix_stack.pop()
        
        # 6. Instanced Visualization (Final Step)
        if hasattr(self, 'latest_joint_data') and self.latest_joint_data is not None:
             self.draw_instanced(self.latest_joint_data)

    def traverse_matrices(self, joint_index, current_mat):
        if joint_index == t_PelvisAnchor:
             joint = self.body.joints[joint_index]
             trans = np.identity(4, dtype='f4')
             t_vec = joint.bone_translation * 0.01 
             trans[:3, 3] = t_vec
             
             anim_rot = np.identity(4, dtype='f4')
             if self.body.joint_matrices is not None and -1 < joint_index < self.body.joint_matrices.shape[1]: 
                  anim_rot = self.body.joint_matrices[self.body.current_body, joint_index].reshape((4,4)).T
                  if np.abs(anim_rot).sum() < 0.1: anim_rot = np.identity(4, dtype='f4')

             current_mat = current_mat @ trans @ anim_rot
        
        # self.global_matrices[joint_index] = current_mat # DO NOT OVERWRITE! This destroys limb alignment
        
        children = self.body.joints[joint_index].immed_children
        for child_idx in children:
             self.process_child_matrices(child_idx, current_mat)

    def process_child_matrices(self, joint_index, parent_mat):
        joint = self.body.joints[joint_index]
        if joint is None: return

        rest_rot = np.identity(4, dtype='f4')
        trans_bone = np.identity(4, dtype='f4')
        trans_bone[:3, 3] = joint.bone_translation 
        
        anim_rot = np.identity(4, dtype='f4')
        if self.body.joint_matrices is not None and -1 < joint_index < self.body.joint_matrices.shape[1]: 
             anim_matrix = self.body.joint_matrices[self.body.current_body, joint_index].reshape((4,4)).T
             if np.abs(anim_matrix).sum() > 0.1: 
                  anim_rot = anim_matrix
        
        # Calculate Limb Matrix
        b_vec = joint.bone_translation
        b_len = np.linalg.norm(b_vec)
        
        target_axis = b_vec / (b_len + 1e-6)
        up = np.array([0.0, 0.0, 1.0], dtype='f4')
        
        rot_align = np.identity(4, dtype='f4')
        dot_p = np.dot(up, target_axis)
        
        if np.abs(dot_p) < 0.99:
             axis = np.cross(up, target_axis)
             angle = math.acos(dot_p)
             rot_align = rotation_matrix(math.degrees(angle), axis).T
        elif dot_p < -0.99:
             rot_align = rotation_matrix(180, [1, 0, 0]).T
        
        limb_model = parent_mat @ rest_rot @ rot_align
        self.global_matrices[joint_index] = limb_model
        
        child_global = parent_mat @ rest_rot @ trans_bone @ anim_rot
        self.traverse_matrices(joint_index, child_global)

        

