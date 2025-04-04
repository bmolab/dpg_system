import dearpygui.dearpygui as dpg
import time

import numpy as np
import torch

from dpg_system.node import Node
import threading
from dpg_system.conversion_utils import *
from dpg_system.matrix_nodes import RollingBuffer


def register_plot_nodes():
    Node.app.register_node("plot", PlotNode.factory)
    Node.app.register_node("heat_map", HeatMapNode.factory)
    Node.app.register_node("heat_scroll", HeatScrollNode.factory)
    Node.app.register_node("profile", ProfileNode.factory)


class BasePlotNode(Node):
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.x_axis = dpg.generate_uuid()
        self.y_axis = dpg.generate_uuid()
        self.plot_data_tag = []
        self.plot_data_tag.append(dpg.generate_uuid())
        self.plot_tag = dpg.generate_uuid()
        self.heat_map_colour_property = None
        self.update_style = 'input is stream of samples'
        self.sample_count = 200
        self.pending_sample_count = self.sample_count
        self.rows = 1
        self.x_data = np.linspace(0, self.sample_count, self.sample_count)
        self.roll_along_x = False
        self.format = ''

        self.width = 300
        self.height = 128
        self.min_y = -1.0
        self.max_y = 1.0
        self.min_x = 0
        self.max_x = self.sample_count

        self.x_axis_scaler = 1

        self.range = 1.0
        self.offset = 0.0

        default_color_map = 'viridis'

        self.lock = threading.Lock()
        self.plotter = None
        self.width_option = None
        self.height_option = None
        self.min_y_option = None
        self.max_y_option = None
        self.min_x_option = None
        self.max_x_option = None
        self.sample_count_option = None

        self.input = self.add_input('y', triggers_execution=True)

        self.output = self.add_output('')
        self.plot_display = self.add_display('')
        self.plot_display.submit_callback = self.submit_display

        self.heat_map_colour_property = self.add_option('color', widget_type='combo', default_value=default_color_map, callback=self.change_colormap)
        self.heat_map_colour_property.widget.combo_items = ['none', 'deep', 'dark', 'pastel', 'paired', 'viridis', 'plasma', 'hot', 'cool', 'pink', 'jet', 'twilight', 'red-blue', 'brown-bluegreen', 'pink-yellowgreen', 'spectral', 'greys']
        self.width_option = self.add_option(label='width', widget_type='drag_int', default_value=self.width, max=3840, callback=self.change_size)
        self.height_option = self.add_option(label='height', widget_type='drag_int', default_value=self.height, max=3840, callback=self.change_size)

    def add_min_and_max_y_options(self):
        self.min_y_option = self.add_option(label='min y', widget_type='drag_float', default_value=self.min_y, callback=self.change_range)
        self.min_y_option.widget.speed = .01

        self.max_y_option = self.add_option(label='max y', widget_type='drag_float', default_value=self.max_y, callback=self.change_range)
        self.max_y_option.widget.speed = .01

    def add_min_and_max_x_options(self):
        self.min_x_option = self.add_option(label='min x', widget_type='drag_float', default_value=self.min_x, max=3840, callback=self.change_range)
        self.min_x_option.widget.speed = .01

        self.max_x_option = self.add_option(label='max x', widget_type='drag_float', default_value=self.max_x, max=100000, callback=self.change_range)
        self.max_x_option.widget.speed = .01

    def add_sample_count_option(self):
        self.sample_count_option = self.add_option(label='sample count', widget_type='drag_int', default_value=self.sample_count, max=100000, callback=self.change_sample_count)

    def set_custom_visibility(self):
        if self.visibility == 'show_all':
            dpg.bind_item_theme(self.plot_tag, self.app.global_theme)
            for i in range(len(self.plot_data_tag)):
                if dpg.does_item_exist(self.plot_data_tag[i]):
                    dpg.bind_item_theme(self.plot_data_tag[i], self.app.global_theme)
            dpg.bind_item_theme(self.y_axis, self.app.global_theme)
            dpg.bind_item_theme(self.x_axis, self.app.global_theme)
            dpg.configure_item(self.plot_tag, show=True)
            self.change_colormap()
        elif self.visibility == 'widgets_only':
            dpg.bind_item_theme(self.plot_tag, self.app.global_theme)
            for i in range(len(self.plot_data_tag)):
                dpg.bind_item_theme(self.plot_data_tag[i], self.app.global_theme)
            dpg.bind_item_theme(self.y_axis, self.app.global_theme)
            dpg.bind_item_theme(self.x_axis, self.app.global_theme)
            dpg.configure_item(self.plot_tag, show=True)
            self.change_colormap()
        else:
            for i in range(len(self.plot_data_tag)):
                dpg.bind_item_theme(self.plot_data_tag[0], self.app.invisible_theme)
            dpg.bind_item_theme(self.plot_tag, self.app.invisible_theme)
            dpg.bind_item_theme(self.y_axis, self.app.invisible_theme)
            dpg.bind_item_theme(self.x_axis, self.app.invisible_theme)
            dpg.configure_item(self.plot_tag, show=False)

    def change_colormap(self):
        if self.heat_map_colour_property is not None:
            colormap = self.heat_map_colour_property()
            if colormap == 'none':
                dpg.bind_colormap(self.plot_tag, None)
            if colormap == 'deep':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Deep)
            elif colormap == 'dark':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Dark)
            elif colormap == 'pastel':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Pastel)
            elif colormap == 'paired':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Paired)
            elif colormap == 'viridis':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
            elif colormap == 'plasma':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Plasma)
            elif colormap == 'hot':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Hot)
            elif colormap == 'cool':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Cool)
            elif colormap == 'pink':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Pink)
            elif colormap == 'jet':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Jet)
            elif colormap == 'twilight':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Twilight)
            elif colormap == 'red-blue':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_RdBu)
            elif colormap == 'brown-bluegreen':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_BrBG)
            elif colormap == 'pink-yellowgreen':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_PiYG)
            elif colormap == 'spectral':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Spectral)
            elif colormap == 'greys':
                dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Greys)

    def reallocate_buffer(self):
        self.y_data = RollingBuffer((self.sample_count, self.rows), roll_along_x=False)
        self.y_data.owner = self
        self.y_data.buffer_changed_callback = self.buffer_changed
        self.y_data.set_update_style(self.update_style)

    def buffer_changed(self, buffer):
        try:
            if self.sample_count is not None:
                self.sample_count_option.set(self.sample_count)

            if self.min_x is not None:
                self.min_x_option.set(0)
            self.min_x = 0
            if self.max_x_option is not None:
                self.max_x_option.set(self.sample_count)
            self.max_x = self.sample_count

            dpg.set_axis_limits(self.x_axis, self.min_x / self.x_axis_scaler, self.max_x / self.x_axis_scaler)

            for i in range(len(self.plot_data_tag)):
                if dpg.does_item_exist(self.plot_data_tag[i]):
                    dpg.configure_item(self.plot_data_tag[i], rows=1, cols=self.y_data.sample_count)

            self.change_range()
        except Exception as e:
            pass

    def submit_display(self):
        with dpg.plot(label='', tag=self.plot_tag, height=self.height, width=self.width, no_title=True) as self.plotter:
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis, no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label="", tag=self.y_axis, no_tick_labels=True)

    def update_plot(self):
        buffer = self.y_data.get_buffer()
        if buffer is not None:
            dpg.set_value(self.plot_data_tag[0], [self.x_data, buffer.ravel()])
            self.y_data.release_buffer()

    def change_size(self):
        if self.width_option is not None:
            dpg.set_item_width(self.plot_tag, self.width_option())
        if self.height_option is not None:
            dpg.set_item_height(self.plot_tag, self.height_option())

    def change_range(self):
        self.max_y = self.max_y_option()
        self.min_y = self.min_y_option()
        self.range = self.max_y - self.min_y
        self.offset = - self.min_y
        if self.min_x_option is not None:
            self.min_x = self.min_x_option()
            self.max_x = self.max_x_option()
        dpg.set_axis_limits(self.x_axis, self.min_x / self.x_axis_scaler,
                            self.max_x / self.x_axis_scaler)
        dpg.set_axis_limits(self.y_axis, self.min_y, self.max_y)

    def change_sample_count(self):
        self.lock.acquire(blocking=True)
        if self.sample_count_option is not None:
            self.sample_count = self.sample_count_option()
        if self.sample_count < 1:
            self.sample_count = 1
            if self.sample_count_option is not None:
                self.sample_count_option.set(self.sample_count)
        del self.x_data
        del self.y_data
        self.x_data = np.linspace(0, self.sample_count, self.sample_count)
        self.reallocate_buffer()

        if self.min_x_option is not None:
            self.min_x_option.set(0)
        self.min_x = 0
        if self.max_x_option is not None:
            self.max_x_option.set(self.sample_count)
        self.max_x = self.sample_count

        for i in range(len(self.plot_data_tag)):
            if dpg.does_item_exist(self.plot_data_tag[i]):
                dpg.delete_item(self.plot_data_tag[i])

        self.adjust_to_sample_count_change()

        self.pending_sample_count = self.sample_count
        self.change_range()
        self.lock.release()

    def adjust_to_sample_count_change(self):
        pass

class PlotNode(BasePlotNode):
    @staticmethod
    def factory(name, data, args=None):
        node = PlotNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.style_type = 'line'
        self.style = 0
        self.update_style = 'input is stream of samples'

        default_color_map = 'none'

        self.style_property = self.add_option('style', widget_type='combo', default_value=self.style_type, callback=self.change_style_property)
        self.style_property.widget.combo_items = ['line', 'scatter', 'stair', 'stem', 'bar']

        self.update_style_property = self.add_option('update style', widget_type='combo', width=240, default_value='input is stream of samples', callback=self.update_style_changed)
        self.update_style_property.widget.combo_items = ['input is stream of samples', 'input is multi-channel sample', 'buffer holds one sample of input']

        self.add_sample_count_option()
        self.add_min_and_max_x_options()
        self.add_min_and_max_y_options()

        self.pending_breadth = 0

        with dpg.theme() as self.line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 200, 0), category=dpg.mvThemeCat_Plots)

    def update_style_changed(self):
        self.update_style = self.update_style_property()
        self.y_data.set_update_style(self.update_style)

    def custom_create(self, from_file):
        self.reallocate_buffer()

        if self.style_type == 'bar':
            self.min_y = 0.0
        else:
            self.min_y = -1.0

        self.min_y_option.set(self.min_y)
        self.max_y = 1.0
        self.max_y_option.set(self.max_y)
        dpg.set_axis_limits(self.y_axis, self.min_y, self.max_y)
        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            dpg.add_line_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
            dpg.bind_item_theme(self.plot_data_tag[0], self.line_theme)
            self.y_data.release_buffer()
        self.change_style_property()
        self.change_colormap()

    # def buffer_changed(self, buffer):
    #     try:
    #         self.sample_count_option.set(self.sample_count)
    #         self.min_x_option.set(0)
    #         self.max_x_option.set(self.sample_count)
    #         dpg.set_axis_limits(self.x_axis, self.min_x, self.max_x)
    #         for i in range(len(self.plot_data_tag)):
    #             if dpg.does_item_exist(self.plot_data_tag[i]):
    #                 dpg.configure_item(self.plot_data_tag[i], rows=1, cols=self.y_data.sample_count)
    #
    #         self.change_range()
    #     except Exception as e:
    #         print(e)

    def change_style_property(self):
        self.style_type = self.style_property()
        self.heat_map_colour_property.set('none')
        self.update_style = 'input is stream of samples'
        self.y_data.set_update_style(self.update_style)
        if self.sample_count_option() == 1:
            self.sample_count_option.set(200)
        self.change_sample_count()

    def adjust_to_sample_count_change(self):
        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            if self.y_data.breadth > 1:
                if len(self.plot_data_tag) < self.y_data.breadth:
                    for i in range(len(self.plot_data_tag), self.y_data.breadth):
                        self.plot_data_tag.append(dpg.generate_uuid())
                for i in range(self.y_data.breadth):
                    this_buffer = np.ascontiguousarray(np.expand_dims(buffer[i, :], axis=0))
                    if self.style_type == 'line':
                        dpg.add_line_series(self.x_data, this_buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[i])
                    elif self.style_type == 'scatter':
                        dpg.add_scatter_series(self.x_data, this_buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[i])
                    elif self.style_type == 'stair':
                        dpg.add_stair_series(self.x_data, this_buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[i])
                    elif self.style_type == 'stem':
                        dpg.add_stem_series(self.x_data, this_buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[i])
                    elif self.style_type == 'bar':
                        dpg.add_bar_series(self.x_data, this_buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[i])
                self.change_colormap()
            else:
                if self.style_type == 'line':
                    dpg.add_line_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
                elif self.style_type == 'scatter':
                    dpg.add_scatter_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
                elif self.style_type == 'stair':
                    dpg.add_stair_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
                elif self.style_type == 'stem':
                    dpg.add_stem_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
                elif self.style_type == 'bar':
                    dpg.add_bar_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
                self.change_colormap()
            self.y_data.release_buffer()

    # def change_range(self):
    #     self.max_y = self.max_y_option()
    #     self.min_y = self.min_y_option()
    #     self.range = self.max_y - self.min_y
    #     self.offset = - self.min_y
    #     dpg.set_axis_limits(self.y_axis, self.min_y, self.max_y)
    #     dpg.set_axis_limits(self.x_axis, self.min_x_option(), self.max_x_option())

    def execute(self):
        if self.pending_sample_count != self.sample_count:
            self.sample_count_option.set(self.pending_sample_count)
            self.change_sample_count()

        if self.pending_breadth != 0:
            self.change_sample_count()
            self.pending_breadth = 0

        self.lock.acquire(blocking=True)
        if self.input.fresh_input:   # standard plot
            data = self.input()

            t = type(data)
            if self.app.torch_available and t == torch.Tensor:
                data = any_to_array(data)
                t = np.ndarray
            if t == str:
                data = any_to_array(data)
                t = np.ndarray
            if t not in [list, np.ndarray, torch.Tensor]:
                ii = any_to_array(float(data))
                self.y_data.update(ii)
            elif t == torch.Tensor:
                data = tensor_to_array(data)
                t = np.ndarray
            elif t == list:
                if len(data) == 1:
                    ii = any_to_array(float(data[0]))
                    self.y_data.update(ii)
                else:
                    data = list_to_array(data, validate=True)
                    if data is None:
                        return
                    t = np.ndarray
            if t == np.ndarray:
                if data.dtype in [np.csingle, np.cdouble, np.clongdouble]:
                    data = data.real
                if data.dtype in [float, np.float32, np.double, int, np.int64, np.uint8, np.bool_]:
                    if self.update_style == 'input is stream of samples':
                        if self.rows > 1:
                            self.rows = 1
                            self.pending_breadth = self.rows
                        if self.rows != self.y_data.breadth:
                            self.lock.release()
                            return
                        self.y_data.set_update_style(self.update_style)

                        if len(data.shape) == 1:
                            if data.shape[0] > self.sample_count:
                                self.pending_sample_count = data.shape[0]
                            else:
                                self.y_data.update(data)
                        elif len(data.shape) == 0:
                            self.y_data.update(np.expand_dims(data, axis=0))
                    elif self.update_style == 'input is multi-channel sample':
                        if len(data.shape) == 2 and data.shape[1] > 1 and data.shape[0] > 1:
                            rows = data.shape[0]
                            self.pending_breadth = rows
                            self.pending_sample_count = data.shape[1]
                        else:
                            rows = data.size
                        if rows != self.rows:
                            self.rows = rows
                            self.pending_breadth = self.rows

                        if len(self.plot_data_tag) < self.rows:
                            for i in range(len(self.plot_data_tag), rows):
                                self.plot_data_tag.append(dpg.generate_uuid())
                        if self.rows != self.y_data.breadth or self.sample_count != self.pending_sample_count:
                            self.lock.release()
                            return

                        self.y_data.set_update_style(self.update_style)
                        if len(data.shape) == 2 and data.shape[1] > 1 and data.shape[0] > 1:
                            ii = data.reshape((rows, self.sample_count))
                        else:
                            ii = data.reshape((rows, 1))

                        self.y_data.update(ii)
                    elif self.update_style == 'buffer holds one sample of input' and len(data.shape) > 0:
                        if len(data.shape) == 1:
                            if data.shape[0] == 1:
                                self.y_data.update(data)
                            else:
                                if self.sample_count != data.shape[0]:
                                    self.pending_sample_count = data.shape[0]
                                    self.lock.release()
                                    return
                                else:
                                    self.y_data.update(data)
                    else:
                        if len(data.shape) == 1:
                            if data.shape[0] > self.sample_count:
                                self.pending_sample_count = data.shape[0]
                            else:
                                self.y_data.update(data)
                        elif len(data.shape) == 0:
                            self.y_data.update(np.expand_dims(data, axis=0))
        buffer = self.y_data.get_buffer()
        if buffer is not None:
            if self.y_data.breadth > 1:
                for i in range(self.y_data.breadth):
                    if dpg.does_item_exist(self.plot_data_tag[i]):
                        dpg.set_value(self.plot_data_tag[i], [self.x_data, buffer[i].ravel()])
            else:
                dpg.set_value(self.plot_data_tag[0], [self.x_data, buffer.ravel()])
            self.y_data.release_buffer()
        self.lock.release()
        self.send_all()

    def update_plot(self):
        buffer = self.y_data.get_buffer()
        if buffer is not None:
            for i in range(self.y_data.breadth):
                dpg.set_value(self.plot_data_tag[i], [self.x_data, buffer[i].ravel()])
            self.y_data.release_buffer()


class HeatMapNode(BasePlotNode):
    @staticmethod
    def factory(name, data, args=None):
        node = HeatMapNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.add_min_and_max_y_options()

        self.format = '%.3f'
        self.format_option = self.add_option(label='number format', widget_type='text_input', default_value='', callback=self.change_format)
        self.hold_format = self.format
        self.x_axis_scaler = self.sample_count
        self.pending_rows = None

    def submit_display(self):
        with dpg.plot(label='', tag=self.plot_tag, height=self.height, width=self.width, no_title=True) as self.plotter:
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis, no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label="", tag=self.y_axis, no_tick_labels=True)

    def frame_task(self):
        if self.pending_sample_count != self.sample_count or (self.pending_rows is not None and self.pending_rows != self.rows):
            if self.sample_count_option is not None:
                self.sample_count_option.set(self.pending_sample_count)
            else:
                self.sample_count = self.pending_sample_count
            self.change_sample_count()
            self.input.fresh_input = True
            self.execute()
            self.remove_frame_tasks()

    def custom_create(self, from_file):
        self.reallocate_buffer()

        self.min_y = 0.0
        self.min_y_option.set(0.0)
        self.max_y = 1.0
        self.max_y_option.set(1.0)
        self.format = '%.3f'
        dpg.set_axis_limits(self.y_axis, 0, 1)
        dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)
        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            dpg.add_heat_series(x=buffer.ravel(), rows=self.y_data.breadth, cols=self.y_data.sample_count, parent=self.y_axis,
                                tag=self.plot_data_tag[0], format=self.format, scale_min=self.min_y, scale_max=self.max_y)
            self.y_data.release_buffer()

        dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
        self.change_range()
        self.format_option.set(self.format)

    def reallocate_buffer(self):
        self.y_data = RollingBuffer((self.sample_count, self.rows), roll_along_x=False)
        self.y_data.owner = self
        self.y_data.buffer_changed_callback = self.buffer_changed
        self.y_data.set_update_style('buffer holds one sample of input')

    def change_range(self):
        super().change_range()
        dpg.set_axis_limits(self.y_axis, 0.0, 1.0)

    def adjust_to_sample_count_change(self):
        if self.rows != self.pending_rows:
            self.rows = self.pending_rows
            self.reallocate_buffer()
        self.x_axis_scaler = self.sample_count
        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            dpg.set_axis_limits(self.y_axis, self.min_y, self.max_y)
            dpg.set_axis_limits(self.x_axis, self.min_x / self.x_axis_scaler, self.max_x / self.x_axis_scaler)
            dpg.add_heat_series(x=buffer.ravel(), rows=self.y_data.breadth, cols=self.y_data.sample_count, parent=self.y_axis,
                                tag=self.plot_data_tag[0], format=self.format, scale_min=self.min_y, scale_max=self.max_y)
            self.change_colormap()
            self.y_data.release_buffer()

    def change_format(self):
        self.format = self.format_option()
        if self.format != '':
            self.hold_format = self.format
        dpg.configure_item(self.plot_data_tag[0], format=self.format)

    def execute(self):
        if self.pending_sample_count != self.sample_count or (self.pending_rows is not None and self.pending_rows != self.rows):
            if self.sample_count_option is not None:
                self.sample_count_option.set(self.pending_sample_count)
            else:
                self.sample_count = self.pending_sample_count
            self.change_sample_count()

        self.lock.acquire(blocking=True)
        if self.input.fresh_input:   # standard plot
            data = self.input()

            t = type(data)
            if self.app.torch_available and t == torch.Tensor:
                data = any_to_array(data)
                t = np.ndarray
            if t == str:
                if data == 'dump':
                    self.output.send(self.y_data.get_buffer()[0])
                    self.y_data.release_buffer()
                else:
                    data = any_to_array(data)
                    t = np.ndarray

            if t not in [list, np.ndarray, torch.Tensor]:
                ii = any_to_array(data)
                if self.range != 1.0 or self.offset != 0:
                    ii = (ii + self.offset) / self.range
                rows = 1
                sample_count = 1
                if rows != self.rows or sample_count != self.sample_count:
                    self.pending_rows = rows
                    self.pending_sample_count = sample_count
                    self.lock.release()
                    self.add_frame_task()
                    return
                self.y_data.update(ii)

            elif t == list:
                if len(data) == 1:
                    ii = any_to_array(data[0])
                    if self.range != 1.0 or self.offset != 0:
                        ii = (ii + self.offset) / self.range
                    rows = 1
                    sample_count = 1
                    if rows != self.rows or sample_count != self.sample_count:
                        self.pending_rows = rows
                        self.pending_sample_count = sample_count
                        self.lock.release()
                        self.add_frame_task()
                        return
                    self.y_data.update(ii)
                else:
                    data = list_to_array(data, validate=True)
                    if data is None:
                        return
                    t = np.ndarray
            elif t == torch.Tensor:
                data = tensor_to_array(data)
                t = np.ndarray
            if t == np.ndarray:
                if data.dtype in [float, np.float32, np.double, int, np.int64, np.uint8, np.bool_]:
                    dims = len(data.shape)
                    rows = 1
                    sample_count = data.shape[0]
                    if dims > 1:
                        rows = data.shape[1]
                    if rows != self.rows or self.sample_count != sample_count:
                        self.pending_rows = rows
                        self.pending_sample_count = sample_count
                        self.lock.release()
                        self.add_frame_task()
                        return
                    if self.range != 1.0 or self.offset != 0:
                        ii = (data + self.offset) / self.range
                        self.y_data.update(ii)
                    else:
                        self.y_data.update(data)

        buffer = self.y_data.get_buffer()
        forced_format = False
        if len(buffer.shape) == 1:
            if self.width / self.rows < 40:
                forced_format = True
                if len(self.format) > 0:
                    self.hold_format = self.format
                    self.format_option.set('')
                    self.format = ''
                    self.change_format()
        else:
            if self.sample_count == 1:
                if self.width / self.rows < 40:
                    forced_format = True
                    if len(self.format) > 0:
                        self.hold_format = self.format
                        self.format_option.set('')
                        self.format = ''
                        self.change_format()
            elif self.width / self.rows < 40:
                forced_format = True
                if len(self.format) > 0:
                    self.hold_format = self.format
                    self.format_option.set('')
                    self.format = ''
                    self.change_format()

            elif len(buffer.shape) > 1 and (self.height / buffer.shape[0]) < 16:
                forced_format = True
                if len(self.format) > 0:
                    self.hold_format = self.format
                    self.format_option.set('')
                    self.format = ''
                    self.change_format()

        if not forced_format and self.hold_format != self.format:
            self.format = self.hold_format
            self.format_option.set(self.hold_format)
            self.change_format()

        if buffer is not None:
            dpg.set_value(self.plot_data_tag[0], [buffer.ravel(), self.x_data])
            self.y_data.release_buffer()

        self.lock.release()
        self.send_all()


class HeatScrollNode(BasePlotNode):
    @staticmethod
    def factory(name, data, args=None):
        node = HeatScrollNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.style_type = label
        self.update_style = 'input is multi-channel sample'

        self.add_sample_count_option()
        self.add_min_and_max_y_options()

        self.format_option = self.add_option(label='number format', widget_type='text_input', default_value='', callback=self.change_format)

        self.hold_format = self.format
        self.x_axis = self.sample_count

    def submit_display(self):
        with dpg.plot(label='', tag=self.plot_tag, height=self.height, width=self.width, no_title=True) as self.plotter:
            dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag=self.x_axis, no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label="", tag=self.y_axis, no_tick_labels=True)

    def custom_create(self, from_file):
        self.reallocate_buffer()
        self.min_y = 0.0
        self.min_y_option.set(0.0)
        dpg.set_axis_limits(self.y_axis, 0, 1)
        dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)

        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            dpg.add_heat_series(x=buffer.ravel(), rows=self.y_data.breadth, cols=self.y_data.sample_count, parent=self.y_axis,
                            tag=self.plot_data_tag[0], format=self.format, scale_min=0, scale_max=1)
            self.y_data.release_buffer()
        dpg.bind_colormap(self.plot_tag, dpg.mvPlotColormap_Viridis)
        self.change_range()

    # def buffer_changed(self, buffer):
    #     self.sample_count_option.set(self.sample_count)
    #     self.min_x = 0
    #     self.max_x = self.sample_count
    #     dpg.set_axis_limits(self.x_axis, self.min_x / self.sample_count, self.max_x / self.sample_count)
    #     for i in range(len(self.plot_data_tag)):
    #         if dpg.does_item_exist(self.plot_data_tag[i]):
    #             dpg.configure_item(self.plot_data_tag[i], rows=1, cols=self.y_data.sample_count)
    #
    #     self.change_range()

    def adjust_to_sample_count_change(self):
        self.x_axis_scaler = self.sample_count
        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            dpg.set_axis_limits(self.y_axis, self.min_y, self.max_y)
            dpg.set_axis_limits(self.x_axis, self.min_x / self.x_axis_scaler, self.max_x / self.x_axis_scaler)
            dpg.add_heat_series(x=buffer, rows=self.y_data.breadth, cols=self.y_data.sample_count, parent=self.y_axis,
                                tag=self.plot_data_tag[0], format=self.format, scale_min=self.min_y, scale_max=self.max_y)
            self.change_colormap()
            self.y_data.release_buffer()

    # def change_range(self):
    #     super().change_range()
    #     dpg.set_axis_limits(self.y_axis, 0.0, 1.0)

    def change_format(self):
        self.format = self.format_option()
        if self.format != '':
            self.hold_format = self.format
        dpg.configure_item(self.plot_data_tag[0], format=self.format)

    def execute(self):
        if self.pending_sample_count != self.sample_count:
            self.sample_count_option.set(self.pending_sample_count)
            self.change_sample_count()

        self.lock.acquire(blocking=True)
        if self.input.fresh_input:   # standard plot
            data = self.input()

            t = type(data)
            if self.app.torch_available and t == torch.Tensor:
                data = any_to_array(data)
                t = np.ndarray
            if t == str:
                data = any_to_array(data)
                t = np.ndarray

            if t not in [list, np.ndarray]:
                ii = any_to_array(data)
                ii = (ii + self.offset) / self.range
                self.y_data.update(ii)
            elif t == list:
                if len(data) == 1:
                    ii = any_to_array(data[0])
                    ii = (ii + self.offset) / self.range
                    self.y_data.update(ii)
                else:
                    rows = len(data)
                    if rows != self.rows:
                        self.rows = rows
                        self.sample_count_option.set(self.sample_count)
                    ii = list_to_array(data, validate=True)
                    if ii is None:
                        return
                    ii = ii.reshape((rows, 1))
                    ii = (ii + self.offset) / self.range
                    self.y_data.update(ii)
            elif t == np.ndarray:
                if data.dtype in [np.csingle, np.cdouble, np.clongdouble]:
                    data = data.real

                if data.dtype in [float, np.float32, np.double, int, np.int64, np.bool_]:
                    rows = data.size
                    if rows != self.rows:
                        self.rows = rows
                    ii = (data.reshape((rows, 1)) + self.offset) / self.range
                    self.y_data.update(ii)

        buffer = self.y_data.get_buffer()
        forced_format = False
        if len(buffer.shape) == 1:
            if self.width / self.rows < 40:
                forced_format = True
                if len(self.format) > 0:
                    self.hold_format = self.format
                    self.format_option.set('')
                    self.format = ''
                    self.change_format()
        else:
            if self.sample_count == 1:
                if self.width / self.rows < 40:
                    forced_format = True
                    if len(self.format) > 0:
                        self.hold_format = self.format
                        self.format_option.set('')
                        self.format = ''
                        self.change_format()
            elif self.width / self.sample_count < 40:
                forced_format = True
                if len(self.format) > 0:
                    self.hold_format = self.format
                    self.format_option.set('')
                    self.format = ''
                    self.change_format()

            elif len(buffer.shape) > 1 and (self.height / buffer.shape[0]) < 16:
                forced_format = True
                if len(self.format) > 0:
                    self.hold_format = self.format
                    self.format_option.set('')
                    self.format = ''
                    self.change_format()

        if not forced_format and self.hold_format != self.format:
            self.format = self.hold_format
            self.format_option.set(self.hold_format)
            self.change_format()

        if buffer is not None:
            dpg.set_value(self.plot_data_tag[0], [buffer.ravel(), self.x_data])
            self.y_data.release_buffer()

        self.lock.release()
        self.send_all()


class ProfileNode(BasePlotNode):
    mousing_plot = None

    @staticmethod
    def factory(name, data, args=None):
        node = ProfileNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.style = -1
        self.style_type = 'bar'
        self.sample_count = 16
        self.min_y = 0
        self.max_y = 1.0
        self.style = 0
        self.update_style = 'input is stream of samples'
        default_color_map = 'deep'

        self.hovered = False

        self.add_sample_count_option()
        self.add_min_and_max_x_options()
        self.add_min_and_max_y_options()

        self.heat_map_colour_property = self.add_option('color', widget_type='combo', default_value=default_color_map,
                                                        callback=self.change_colormap)
        self.heat_map_colour_property.widget.combo_items = ['none', 'deep', 'dark', 'pastel', 'paired', 'viridis',
                                                            'plasma', 'hot', 'cool', 'pink', 'jet', 'twilight',
                                                            'red-blue', 'brown-bluegreen', 'pink-yellowgreen',
                                                            'spectral', 'greys']

        self.continuous_output = self.add_option(label='continuous output', widget_type='checkbox', default_value=False)

        self.was_drawing = False
        self.add_frame_task()
        self.last_pos = [0, 0]
        self.hold_format = self.format

        with dpg.theme() as self.line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 200, 0), category=dpg.mvThemeCat_Plots)

    def custom_create(self, from_file):
        self.reallocate_buffer()
        self.change_sample_count()
        # self.change_style_property()
        self.input.set_label('index into profile')

    def save_custom(self, container):
        self.lock.acquire(blocking=True)
        container['data'] = self.y_data.get_buffer().tolist()
        self.y_data.release_buffer()
        self.lock.release()

    def load_custom(self, container):
        self.lock.acquire(blocking=True)
        if 'data' in container:
            data = np.array(container['data'])
            if len(data.shape) == 1:
                self.y_data.update(data)
            elif len(data.shape) == 2 and data.shape[0] == 1:
                self.y_data.update(data[0])
            buffer = self.y_data.get_buffer()
            dpg.set_value(self.plot_data_tag[0], [self.x_data, buffer.ravel()])
            self.y_data.release_buffer()
        self.lock.release()

    def frame_task(self):
        if ProfileNode.mousing_plot == self.plotter or ProfileNode.mousing_plot is None:
            x = 0
            y = 0
            ref_pos = [-1, -1]
            if self.was_drawing:
                if not dpg.is_mouse_button_down(0):
                    self.was_drawing = False
                    self.output.send(self.y_data.get_buffer()[0])
                    self.y_data.release_buffer()
                    PlotNode.mousing_plot = None
                else:
                    editor = self.app.get_current_editor()
                    if editor is not None:
                        node_padding = editor.node_scalers[dpg.mvNodeStyleVar_NodePadding]
                        window_padding = self.app.window_padding
                        plot_padding = 10
                        mouse = dpg.get_mouse_pos(local=True)
                        pos_x = dpg.get_item_pos(self.plotter)[0] + plot_padding + node_padding[0] + window_padding[0]
                        pos_y = dpg.get_item_pos(self.plotter)[1] + plot_padding + node_padding[1] + window_padding[1] + 4  # 4 is from unknown source

                        size = dpg.get_item_rect_size(self.plotter)
                        size[0] -= (2 * plot_padding)
                        size[1] -= (2 * plot_padding)
                        x_scale = self.sample_count / size[0]
                        y_scale = self.range / size[1]

                        off_x = mouse[0] - pos_x
                        off_y = mouse[1] - pos_y
                        unit_x = off_x * x_scale
                        unit_y = off_y * y_scale
                        unit_y = self.max_y - unit_y
                        if unit_x < 0:
                            unit_x = 0
                        elif unit_x >= self.sample_count:
                            unit_x = self.sample_count - 1
                        if unit_y < self.min_y:
                            unit_y = self.min_y
                        elif unit_y > self.max_y:
                            unit_y = self.max_y
                        x = unit_x
                        y = unit_y
                        ref_pos = [x, y]
                        x = int(x)

            if dpg.is_item_hovered(self.plotter):
                if dpg.is_mouse_button_down(0):
                    if self.hovered and not self.was_drawing:
                        PlotNode.mousing_plot = self.plotter
                        self.was_drawing = True
                else:
                    self.hovered = True
                    self.last_pos = [-1, -1]
                    if self.was_drawing:
                        self.was_drawing = False
                        PlotNode.mousing_plot = None

            else:
                self.hovered = False
                if not dpg.is_mouse_button_down(0):
                    self.was_drawing = False
                    PlotNode.mousing_plot = None

            if self.was_drawing:
                if self.last_pos[0] != -1:
                    last_y = self.last_pos[1]
                    last_x = int(round(self.last_pos[0]))
                    change_x = x - last_x
                    change_y = y - last_y
                    if change_x > 0:
                        for i in range(last_x, x):
                            interpolated_y = ((i - last_x) / change_x) * change_y + last_y
                            self.y_data.set_value(i, interpolated_y)
                    else:
                        for i in range(x, last_x):
                            interpolated_y = ((i - x) / change_x) * change_y + last_y
                            self.y_data.set_value(i, interpolated_y)
                if ref_pos[0] != -1:
                    self.last_pos = ref_pos
                    self.y_data.set_value(x, y)
                    self.y_data.set_write_pos(0)
                    self.update_plot()
                    self.was_drawing = True
                    if self.continuous_output():
                        self.output.send(self.y_data.get_buffer()[0])
                        self.y_data.release_buffer()

    # def value_dragged(self):
    #     if not dpg.is_mouse_button_down(0):
    #         return
    #     self.value_changed()

    # def buffer_changed(self, buffer):
    #     try:
    #         if self.sample_count is not None:
    #             self.sample_count_option.set(self.sample_count)
    #
    #         if self.min_x is not None:
    #             self.min_x_option.set(0)
    #         self.min_x = 0
    #         if self.max_x_option is not None:
    #             self.max_x_option.set(self.sample_count)
    #         self.max_x = self.sample_count
    #
    #         dpg.set_axis_limits(self.x_axis, self.min_x / self.x_axis_scaler, self.max_x / self.x_axis_scaler)
    #
    #         for i in range(len(self.plot_data_tag)):
    #             if dpg.does_item_exist(self.plot_data_tag[i]):
    #                 dpg.configure_item(self.plot_data_tag[i], rows=1, cols=self.y_data.sample_count)
    #
    #         self.change_range()
    #     except Exception as e:
    #         pass

    def adjust_to_sample_count_change(self):
        buffer = self.y_data.get_buffer(block=True)
        if buffer is not None:
            if self.style_type == 'line':
                dpg.add_line_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
            elif self.style_type == 'scatter':
                dpg.add_scatter_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
            elif self.style_type == 'stair':
                dpg.add_stair_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
            elif self.style_type == 'stem':
                dpg.add_stem_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
            elif self.style_type == 'bar':
                dpg.add_bar_series(self.x_data, buffer.ravel(), parent=self.y_axis, tag=self.plot_data_tag[0])
            self.change_colormap()
            self.y_data.release_buffer()

    def get_preset_state(self):
        preset = {}
        if self.label == 'profile':
            data = self.y_data.get_buffer()[0]
            preset['data'] = data.tolist()
            self.y_data.release_buffer()
        return preset

    def set_preset_state(self, preset):
        if 'data' in preset:
            data = preset['data']
            data = np.array(data, dtype=float)
            self.y_data.set_write_pos(0)
            self.y_data.update(data)
            self.execute()

    def execute(self):
        if self.pending_sample_count != self.sample_count:
            self.sample_count_option.set(self.pending_sample_count)
            self.change_sample_count()

        self.lock.acquire(blocking=True)
        if self.input.fresh_input:   # standard plot
            data = self.input()

            t = type(data)
            if t == str:
                if data == 'dump':
                    self.output.send(self.y_data.get_buffer()[0])
                    self.y_data.release_buffer()
                    self.lock.release()
                    return
                else:
                    data = any_to_array(data)
                    t = np.ndarray

            if t not in [list, np.ndarray, torch.Tensor]:
                value = 0
                float_data = any_to_float(data)
                if float_data >= 0 and float_data < self.sample_count:
                    pre_index = int(float_data)
                    post_index = int(float_data + 1)
                    interp = float_data - pre_index
                    value = any_to_float(self.y_data.get_value(pre_index) * (1.0 - interp) + self.y_data.get_value(post_index) * interp)
                self.lock.release()
                self.output.send(value)
                return

            if self.app.torch_available and t == torch.Tensor:
                data = any_to_array(data)
                t = np.ndarray

            if t == torch.Tensor:
                data = tensor_to_array(data)
                t = np.ndarray
            elif t == list:
                if len(data) == 1:
                    ii = any_to_array(float(data[0]))
                    self.y_data.update(ii)
                else:
                    data = list_to_array(data, validate=True)
                    if data is None:
                        return
                    t = np.ndarray
            if t == np.ndarray:
                if data.dtype in [np.csingle, np.cdouble, np.clongdouble]:
                    data = data.real
                if data.dtype in [float, np.float32, np.double, int, np.int64, np.uint8, np.bool_]:
                    if len(data.shape) == 1:
                        if data.shape[0] > self.sample_count:
                            self.pending_sample_count = data.shape[0]
                        else:
                            self.y_data.update(data)
                    elif len(data.shape) == 0:
                        self.y_data.update(np.expand_dims(data, axis=0))

        buffer = self.y_data.get_buffer()
        if buffer is not None:
            dpg.set_value(self.plot_data_tag[0], [self.x_data, buffer.ravel()])
            self.y_data.release_buffer()

        self.lock.release()

