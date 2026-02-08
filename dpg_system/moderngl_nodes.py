
import dearpygui.dearpygui as dpg
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import numpy as np
from dpg_system.moderngl_base import MGLContext
import moderngl
from dpg_system.matrix_utils import *

def register_moderngl_nodes():
    Node.app.register_node('mgl_context', MGLContextNode.factory)
    Node.app.register_node('mgl_box', MGLBoxNode.factory)
    Node.app.register_node('mgl_translate', MGLTransformNode.factory)
    Node.app.register_node('mgl_rotate', MGLTransformNode.factory)
    Node.app.register_node('mgl_scale', MGLTransformNode.factory)
    Node.app.register_node('mgl_scale', MGLTransformNode.factory)
    Node.app.register_node('mgl_camera', MGLCameraNode.factory)
    Node.app.register_node('mgl_display', MGLDisplayNode.factory)
    Node.app.register_node('mgl_sphere', MGLSphereNode.factory)
    Node.app.register_node('mgl_cylinder', MGLCylinderNode.factory)
    Node.app.register_node('mgl_color', MGLColorNode.factory)
    Node.app.register_node('mgl_light', MGLLightNode.factory)
    Node.app.register_node('mgl_material', MGLMaterialNode.factory)

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
            input_data = self.mgl_input()
            if input_data == 'draw':
                self.draw()
                self.mgl_output.send('draw')

    def draw(self):
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
        
        self.width_input = self.add_input('width', widget_type='drag_int', default_value=self.width, callback=self.resize)
        self.height_input = self.add_input('height', widget_type='drag_int', default_value=self.height, callback=self.resize)
        self.display_mode_input = self.add_input('display_mode', widget_type='combo', default_value='node')
        self.display_mode_input.widget.combo_items = ['off', 'node', 'window', 'fullscreen']
        
        self.samples_input = self.add_input('samples', widget_type='combo', default_value='4')
        self.samples_input.widget.combo_items = ['0', '2', '4', '6', '8', '16']
        
        self.render_trigger = self.add_input('render', triggers_execution=True)
        self.mgl_chain_output = self.add_output('mgl_chain')
        self.texture_output = self.add_output('texture_tag')

    def custom_cleanup(self):
        if self.texture_tag:
            dpg.delete_item(self.texture_tag)
        if self.external_window:
            dpg.delete_item(self.external_window)
        if self.fullscreen_window:
            dpg.delete_item(self.fullscreen_window)

    def resize(self):
        self.width = max(1, self.width_input())
        self.height = max(1, self.height_input())

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
        if self.display_mode_input() == 'fullscreen' and dpg.is_key_pressed(dpg.mvKey_Escape):
            self.display_mode_input.set('window')
            dpg.set_value(self.display_mode_input.widget.uuid, 'window')
            self.update_display('window')

        # 0. Sync Window Image Size (Stretch to fit)
        if self.external_window and dpg.does_item_exist(self.external_window) and self.display_mode_input() == 'window':
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
        
        # Update FBO size
        try:
            samples = int(self.samples_input())
        except:
            samples = 4
            
        # Scope the ModernGL Context to avoid polluting global GL state (e.g. for legacy gl_nodes)
        with self.context.ctx:
            self.context.update_framebuffer(self.width, self.height, samples=samples)
            
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
            if self.context.samples != samples:
                self.samples_input.set(str(self.context.samples))
            
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
        mode = self.display_mode_input()
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

    def get_translation_matrix(self, t):
        m = np.identity(4, dtype=np.float32)
        m[3, :3] = t
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

    def execute(self):
        # Override execute to wrap the draw call
        if self.mgl_input.fresh_input:
            if self.mgl_input() == 'draw':
                ctx = MGLContext.get_instance()
                ctx.push_matrix()
                
                # Translation
                t = self.translate_input()
                ctx.multiply_matrix(self.get_translation_matrix(t))
                
                # Rotation
                r = self.rotate_input()
                ctx.multiply_matrix(self.get_rotation_matrix(r))
                
                # Scale
                s = self.scale_input()
                ctx.multiply_matrix(self.get_scale_matrix(s))
                
                self.mgl_output.send('draw')
                ctx.pop_matrix()


class MGLColorNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLColorNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.color_input = self.add_input('color', widget_type='color_picker', default_value=[1.0, 1.0, 1.0, 1.0])
        # Base class MGLNode already adds mgl_input ('mgl chain in') and mgl_output ('mgl chain out')

    def execute(self):
        if self.mgl_input.fresh_input:
            if self.mgl_input() == 'draw':
                c = self.color_input()
                # Normalize if needed
                if len(c) > 4: c = c[:4]
                elif len(c) == 3: c = [*c, 255 if c[0] > 1.0 else 1.0]
                
                # Check for 0-255 range
                if max(c) > 1.0:
                    c = [val / 255.0 for val in c]
                
                ctx = MGLContext.get_instance()
                prev_color = ctx.current_color
                try:
                    ctx.current_color = tuple(c)
                    self.mgl_output.send('draw')
                finally:
                    ctx.current_color = prev_color

class MGLLightNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLLightNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
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

    def execute(self):
        if self.mgl_input.fresh_input:
            if self.mgl_input() == 'draw':
                ctx = MGLContext.get_instance()
                
                # Transform Light Position by current Model Matrix
                # So lights can be attached to moving objects
                # pos is (x, y, z, 1)
                pos = self.pos_input()
                if len(pos) == 3: pos = [*pos, 1.0]
                
                model = ctx.get_model_matrix()
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
                ctx.lights.append(light_data)
                
                self.mgl_output.send('draw')


class MGLMaterialNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLMaterialNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.ambient_input = self.add_input('ambient', widget_type='color_picker', default_value=[0.1, 0.1, 0.1])
        self.diffuse_input = self.add_input('diffuse', widget_type='color_picker', default_value=[1.0, 1.0, 1.0])
        self.specular_input = self.add_input('specular', widget_type='color_picker', default_value=[0.5, 0.5, 0.5])
        self.shininess_input = self.add_input('shininess', widget_type='drag_float', default_value=32.0, min_value=1.0, max_value=256.0)

    def custom_create(self, from_file):
        dpg.configure_item(self.ambient_input.widget.uuids[0], label='ambient')
        dpg.configure_item(self.diffuse_input.widget.uuids[0], label='diffuse')
        dpg.configure_item(self.specular_input.widget.uuids[0], label='specular')

    def execute(self):
        if self.mgl_input.fresh_input:
            if self.mgl_input() == 'draw':
                ctx = MGLContext.get_instance()
                
                # Save previous material
                prev_material = ctx.current_material.copy()
                
                # Normalize colors
                def norm(c):
                    if len(c) > 3: c = c[:3]
                    if max(c) > 1.0:
                        return [val / 255.0 for val in c]
                    return c

                # Set new material
                ctx.current_material = {
                    'ambient': norm(self.ambient_input()),
                    'diffuse': norm(self.diffuse_input()),
                    'specular': norm(self.specular_input()),
                    'shininess': self.shininess_input()
                }
                
                try:
                    self.mgl_output.send('draw')
                finally:
                    # Restore
                    ctx.current_material = prev_material


class MGLBoxNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLBoxNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.vbo = None
        self.ibo = None
        self.vao = None
        self.size_input = self.add_input('size', widget_type='drag_float_n', default_value=[1.0, 1.0, 1.0], columns=3, speed=0.01)
        self.mode_input = self.add_input('mode', widget_type='combo', default_value='solid')
        self.mode_input.widget.combo_items = ['solid', 'wireframe', 'points']
        self.cull_input = self.add_input('cull', widget_type='checkbox', default_value=True)
        self.point_size_input = self.add_input('point_size', widget_type='drag_float', default_value=4.0, min_value=1.0)
        self.round_input = self.add_input('round', widget_type='checkbox', default_value=True)
        # Lazy init

    def custom_create(self, from_file):
        dpg.configure_item(self.size_input.widget.uuids[2], label='size')
        dpg.configure_item(self.size_input.widget.uuids[0], width=45)
        dpg.configure_item(self.size_input.widget.uuids[1], width=45)
        dpg.configure_item(self.size_input.widget.uuids[2], width=45)

    def create_geometry(self):
        ctx_wrapper = MGLContext.get_instance()
        ctx = ctx_wrapper.ctx
        # Force context activation
        if ctx_wrapper.fbo:
            ctx_wrapper.fbo.use()
        
        vertices = [
            # Front face
            -0.5, -0.5, 0.5,  0.0, 0.0, 1.0,
             0.5, -0.5, 0.5,  0.0, 0.0, 1.0,
             0.5,  0.5, 0.5,  0.0, 0.0, 1.0,
            -0.5,  0.5, 0.5,  0.0, 0.0, 1.0,
            
            # Back face
            -0.5, -0.5, -0.5, 0.0, 0.0, -1.0,
            -0.5,  0.5, -0.5, 0.0, 0.0, -1.0,
             0.5,  0.5, -0.5, 0.0, 0.0, -1.0,
             0.5, -0.5, -0.5, 0.0, 0.0, -1.0,
             
            # Top face
            -0.5,  0.5, -0.5, 0.0, 1.0, 0.0,
            -0.5,  0.5,  0.5, 0.0, 1.0, 0.0,
             0.5,  0.5,  0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, -0.5, 0.0, 1.0, 0.0,
             
            # Bottom face
            -0.5, -0.5, -0.5, 0.0, -1.0, 0.0,
             0.5, -0.5, -0.5, 0.0, -1.0, 0.0,
             0.5, -0.5,  0.5, 0.0, -1.0, 0.0,
            -0.5, -0.5,  0.5, 0.0, -1.0, 0.0,
            
            # Right face
             0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
             0.5,  0.5, -0.5, 1.0, 0.0, 0.0,
             0.5,  0.5,  0.5, 1.0, 0.0, 0.0,
             0.5, -0.5,  0.5, 1.0, 0.0, 0.0,
             
            # Left face
            -0.5, -0.5, -0.5, -1.0, 0.0, 0.0,
            -0.5, -0.5,  0.5, -1.0, 0.0, 0.0,
            -0.5,  0.5,  0.5, -1.0, 0.0, 0.0,
            -0.5,  0.5, -0.5, -1.0, 0.0, 0.0,
        ]
        
        indices = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ]
        
        vertices = np.array(vertices, dtype='f4')
        indices = np.array(indices, dtype='i4')
        
        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())
        self.prog = MGLContext.get_instance().default_shader
        
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '3f 3f', 'in_position', 'in_normal')], self.ibo)

    def draw(self):
        if self.vao is None:
            self.create_geometry()
            
        ctx = MGLContext.get_instance()
        
        # Uniforms
        # Note: MGL writes matrices as bytes directly
        # We need to make sure arrays are C-contiguous/float32
        
        if 'M' in self.prog:
            model = ctx.get_model_matrix()
            s = self.size_input()
            scale_mat = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
            # Pre-multiply scale for local transformation
            # M_gl = M_parent_gl @ S_gl => M_np = S_np @ M_parent_np
            model = np.dot(scale_mat, model)
            self.prog['M'].write(model.tobytes())
        if 'V' in self.prog:
            self.prog['V'].write(ctx.view_matrix.tobytes())
        if 'P' in self.prog:
            self.prog['P'].write(ctx.projection_matrix.tobytes())
        if 'color' in self.prog:
            c = ctx.current_color
            self.prog['color'].value = tuple(c)
        # Lights
        ctx.update_lights(self.prog)
        ctx.update_material(self.prog)

        # Render Mode
        # Render Mode
        mode = self.mode_input()
        cull = self.cull_input()
        
        # Shader Point Culling
        if 'point_culling' in self.prog:
            self.prog['point_culling'].value = (mode == 'points' and cull)
        if 'round_points' in self.prog:
            self.prog['round_points'].value = (mode == 'points' and self.round_input())

        if cull:
            ctx.ctx.enable(moderngl.CULL_FACE)
        else:
            ctx.ctx.disable(moderngl.CULL_FACE)

        if mode == 'wireframe':
            ctx.ctx.wireframe = True
            self.vao.render()
            ctx.ctx.wireframe = False
        elif mode == 'points':
            if 'point_size' in self.prog:
                self.prog['point_size'].value = self.point_size_input()
            self.vao.render(mode=moderngl.POINTS)
        else:
            self.vao.render()
        
        # Restore CULL_FACE default (Enabled) if changed? 
        # Better to just set it every time or rely on MGLContextNode to reset.
        # MGLContextNode resets to enable(CULL_FACE).

class MGLSphereNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLSphereNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.vbo = None
        self.vao = None
        self.radius_input = self.add_input('radius', widget_type='drag_float', default_value=0.5, speed=0.01)
        self.mode_input = self.add_input('mode', widget_type='combo', default_value='solid')
        self.mode_input.widget.combo_items = ['solid', 'wireframe', 'points']
        self.cull_input = self.add_input('cull', widget_type='checkbox', default_value=True)
        self.point_size_input = self.add_input('point_size', widget_type='drag_float', default_value=4.0, min_value=1.0)
        self.round_input = self.add_input('round', widget_type='checkbox', default_value=True)
        self.stacks_input = self.add_input('stacks', widget_type='drag_int', default_value=16, min_value=3, callback=self.geometry_changed)
        self.sectors_input = self.add_input('sectors', widget_type='drag_int', default_value=32, min_value=3, callback=self.geometry_changed)
        # Lazy init

    def geometry_changed(self):
        self.vao = None
        self.vbo = None

    def custom_create(self, from_file):
        dpg.configure_item(self.stacks_input.widget.uuids[0], width=100)
        dpg.configure_item(self.sectors_input.widget.uuids[0], width=100)

    def create_geometry(self):
        ctx = MGLContext.get_instance().ctx
        import math
        vertices = []
        
        stacks = max(3, self.stacks_input())
        sectors = max(3, self.sectors_input())
        
        # Helper function
        def get_vert(lat_sin, lat_cos, lng):
            x = lat_cos * math.cos(lng)
            y = lat_sin
            z = lat_cos * math.sin(lng)
            return [x, y, z, x, y, z]

        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i+1) / stacks)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            
            for j in range(sectors):
                lng0 = 2 * math.pi * float(j) / sectors
                lng1 = 2 * math.pi * float(j+1) / sectors
                
                v00 = get_vert(z0, zr0, lng0)
                v10 = get_vert(z0, zr0, lng1)
                v01 = get_vert(z1, zr1, lng0)
                v11 = get_vert(z1, zr1, lng1)
                
                # Triangle 1
                vertices.extend(v00)
                vertices.extend(v11)
                vertices.extend(v10)
                
                # Triangle 2
                vertices.extend(v00)
                vertices.extend(v01)
                vertices.extend(v11)

        data = np.array(vertices, dtype='f4')
        self.vbo = ctx.buffer(data.tobytes())
        self.prog = MGLContext.get_instance().default_shader
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '3f 3f', 'in_position', 'in_normal')])
    
    def draw(self):
        if self.vao is None:
            self.create_geometry()
            
        ctx = MGLContext.get_instance()
        if 'M' in self.prog:
            model = ctx.get_model_matrix()
            r = self.radius_input()
            scale_mat = np.diag([r, r, r, 1.0]).astype(np.float32)
            # Pre-multiply scale for local transformation
            model = np.dot(scale_mat, model)
            self.prog['M'].write(model.tobytes())
        if 'V' in self.prog:
            self.prog['V'].write(ctx.view_matrix.tobytes())
        if 'P' in self.prog:
            self.prog['P'].write(ctx.projection_matrix.tobytes())
        if 'color' in self.prog:
            c = ctx.current_color
            self.prog['color'].value = tuple(c)
        # Lights
        ctx.update_lights(self.prog)
        ctx.update_material(self.prog)


        # Render Mode
        mode = self.mode_input()
        cull = self.cull_input()
        
        # Shader Point Culling
        if 'point_culling' in self.prog:
            self.prog['point_culling'].value = (mode == 'points' and cull)
        if 'round_points' in self.prog:
            self.prog['round_points'].value = (mode == 'points' and self.round_input())

        if cull:
            ctx.ctx.enable(moderngl.CULL_FACE)
        else:
            ctx.ctx.disable(moderngl.CULL_FACE)

        if mode == 'wireframe':
            ctx.ctx.wireframe = True
            self.vao.render()
            ctx.ctx.wireframe = False
        elif mode == 'points':
            if 'point_size' in self.prog:
                self.prog['point_size'].value = self.point_size_input()
            self.vao.render(mode=moderngl.POINTS)
        else:
            self.vao.render()


class MGLCylinderNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLCylinderNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.vbo = None
        self.vao = None
        self.radius_input = self.add_input('radius', widget_type='drag_float', default_value=0.5, speed=0.01)
        self.height_input = self.add_input('height', widget_type='drag_float', default_value=1.0, speed=0.01)
        self.mode_input = self.add_input('mode', widget_type='combo', default_value='solid')
        self.mode_input.widget.combo_items = ['solid', 'wireframe', 'points']
        self.cull_input = self.add_input('cull', widget_type='checkbox', default_value=True)
        self.point_size_input = self.add_input('point_size', widget_type='drag_float', default_value=4.0, min_value=1.0)
        self.round_input = self.add_input('round', widget_type='checkbox', default_value=True)
        self.slices_input = self.add_input('slices', widget_type='drag_int', default_value=32, min_value=3, callback=self.geometry_changed)
        # Removed flip_winding
        # Lazy init

    def geometry_changed(self):
        self.vao = None
        self.vbo = None

    def custom_create(self, from_file):
        dpg.configure_item(self.slices_input.widget.uuids[0], width=100)

    def create_geometry(self):
        ctx = MGLContext.get_instance().ctx
        import math
        
        vertices = []
        slices = max(3, self.slices_input())
        # Removed flip = self.flip_input()
        
        # Unit cylinder: radius 1, height 1 (from -0.5 to 0.5 on Y)
        # Side
        for i in range(slices):
            theta0 = 2 * math.pi * float(i) / slices
            theta1 = 2 * math.pi * float(i+1) / slices
            
            x0 = math.cos(theta0); z0 = math.sin(theta0)
            x1 = math.cos(theta1); z1 = math.sin(theta1)
            
            # Bottom (-0.5) and Top (0.5)
            y_bot = -0.5
            y_top = 0.5
            
            # Normals (x, 0, z)
            n0 = [x0, 0, z0]
            n1 = [x1, 0, z1]
            
            # Side Quads -> Triangles
            # v0 (bot, t0), v1 (top, t0), v2 (top, t1), v3 (bot, t1)
            
            p0 = [x0, y_bot, z0]; p1 = [x0, y_top, z0]
            p2 = [x1, y_top, z1]; p3 = [x1, y_bot, z1]
            
            # Default to CCW (Consistent with Caps)
            # Triangle 1: p0, p1, p2
            t1 = [*p0, *n0, *p1, *n0, *p2, *n1]
            # Triangle 2: p0, p2, p3
            t2 = [*p0, *n0, *p2, *n1, *p3, *n1]
            
            # Caps
            # Top Cap (Normal 0, 1, 0)
            nt = [0, 1, 0]
            c_top = [0, 0.5, 0]
            # c_top, p2, p1
            t_top = [*c_top, *nt, *p2, *nt, *p1, *nt]
            
            # Bottom Cap (Normal 0, -1, 0)
            nb = [0, -1, 0]
            c_bot = [0, -0.5, 0]
            # c_bot, p0, p3
            t_bot = [*c_bot, *nb, *p0, *nb, *p3, *nb]
            
            vertices.extend(t1)
            vertices.extend(t2)
            vertices.extend(t_top)
            vertices.extend(t_bot)

        data = np.array(vertices, dtype='f4')
        self.vbo = ctx.buffer(data.tobytes())
        self.prog = MGLContext.get_instance().default_shader
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '3f 3f', 'in_position', 'in_normal')])
    
    def draw(self):
        if self.vao is None:
            self.create_geometry()
            
        ctx = MGLContext.get_instance()
        if 'M' in self.prog:
            model = ctx.get_model_matrix()
            r = self.radius_input()
            h = self.height_input()
            scale_mat = np.diag([r, h, r, 1.0]).astype(np.float32)
            # Pre-multiply scale for local transformation
            model = np.dot(scale_mat, model)
            self.prog['M'].write(model.tobytes())
        if 'V' in self.prog:
            self.prog['V'].write(ctx.view_matrix.tobytes())
        if 'P' in self.prog:
            self.prog['P'].write(ctx.projection_matrix.tobytes())
        if 'color' in self.prog:
            c = ctx.current_color
            self.prog['color'].value = tuple(c)
        # Lights
        ctx.update_lights(self.prog)
        ctx.update_material(self.prog)
        if 'view_pos' in self.prog:
            # We need camera position for culling.
            # Get it from MGLContext or passed uniform?
            # MGLCameraNode updates it?
            # Let's rely on MGLContext storing it, or just use View Matrix inverse?
            # For now, let's assume MGLCameraNode sets 'view_pos' uniform on default shader.
            pass

        # Render Mode
        mode = self.mode_input()
        cull = self.cull_input()
        
        # Shader Point Culling
        if 'point_culling' in self.prog:
            self.prog['point_culling'].value = (mode == 'points' and cull)
        if 'round_points' in self.prog:
            self.prog['round_points'].value = (mode == 'points' and self.round_input())

        if cull:
            ctx.ctx.enable(moderngl.CULL_FACE)
        else:
            ctx.ctx.disable(moderngl.CULL_FACE)

        if mode == 'wireframe':
            ctx.ctx.wireframe = True
            self.vao.render()
            ctx.ctx.wireframe = False
        elif mode == 'points':
            if 'point_size' in self.prog:
                self.prog['point_size'].value = self.point_size_input()
            self.vao.render(mode=moderngl.POINTS)
        else:
            self.vao.render()

class MGLCameraNode(MGLNode):
    @staticmethod
    def factory(name, data, args=None):
        return MGLCameraNode(name, data, args)
        
    def __init__(self, label, data, args):
        super().__init__(label, data, args)
        self.fov = self.add_input('fov', widget_type='drag_float', default_value=60.0, speed=1.0)
        self.pos = self.add_input('pos', widget_type='drag_float_n', default_value=[3, 3, 3], speed=0.1, columns=3)
        self.target = self.add_input('target', widget_type='drag_float_n', default_value=[0, 0, 0], speed=0.1, columns=3)
        self.up = self.add_input('up', widget_type='drag_float_n', default_value=[0, 1, 0], columns=3)

    def custom_create(self, from_file):
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
        ctx = MGLContext.get_instance()
        
        # Projection
        aspect = ctx.width / ctx.height
        if aspect == 0: aspect = 1.0
        p = perspective(self.fov(), aspect, 0.1, 100.0)
        ctx.set_projection_matrix(p)
        
        # View
        v = look_at(self.pos(), self.target(), self.up())
        ctx.set_view_matrix(v)
        
        # Update view_pos in default shader if available
        if 'view_pos' in ctx.default_shader:
            ctx.default_shader['view_pos'].value = tuple(self.pos())
        
        # Just pass signal? Actually Camera should usually be BEFORE transform/geometry 
        # in the chain for View/Proj to be set for them.
        pass
