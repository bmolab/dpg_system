import ctypes
import os
import multiprocessing
import numpy as np
from sympy.stats.rv import probability

from dpg_system.node import Node
from dpg_system.conversion_utils import *
import time
import os

import threading
from queue import Queue

from dpg_system.Cairo_Text_Layout import *
# print('pre color')
# from colormaps import _viridis_data
# print('post color')
import random
from datetime import datetime
_viridis_data = [[0.267004, 0.004874, 0.329415],
                 [0.268510, 0.009605, 0.335427],
                 [0.269944, 0.014625, 0.341379],
                 [0.271305, 0.019942, 0.347269],
                 [0.272594, 0.025563, 0.353093],
                 [0.273809, 0.031497, 0.358853],
                 [0.274952, 0.037752, 0.364543],
                 [0.276022, 0.044167, 0.370164],
                 [0.277018, 0.050344, 0.375715],
                 [0.277941, 0.056324, 0.381191],
                 [0.278791, 0.062145, 0.386592],
                 [0.279566, 0.067836, 0.391917],
                 [0.280267, 0.073417, 0.397163],
                 [0.280894, 0.078907, 0.402329],
                 [0.281446, 0.084320, 0.407414],
                 [0.281924, 0.089666, 0.412415],
                 [0.282327, 0.094955, 0.417331],
                 [0.282656, 0.100196, 0.422160],
                 [0.282910, 0.105393, 0.426902],
                 [0.283091, 0.110553, 0.431554],
                 [0.283197, 0.115680, 0.436115],
                 [0.283229, 0.120777, 0.440584],
                 [0.283187, 0.125848, 0.444960],
                 [0.283072, 0.130895, 0.449241],
                 [0.282884, 0.135920, 0.453427],
                 [0.282623, 0.140926, 0.457517],
                 [0.282290, 0.145912, 0.461510],
                 [0.281887, 0.150881, 0.465405],
                 [0.281412, 0.155834, 0.469201],
                 [0.280868, 0.160771, 0.472899],
                 [0.280255, 0.165693, 0.476498],
                 [0.279574, 0.170599, 0.479997],
                 [0.278826, 0.175490, 0.483397],
                 [0.278012, 0.180367, 0.486697],
                 [0.277134, 0.185228, 0.489898],
                 [0.276194, 0.190074, 0.493001],
                 [0.275191, 0.194905, 0.496005],
                 [0.274128, 0.199721, 0.498911],
                 [0.273006, 0.204520, 0.501721],
                 [0.271828, 0.209303, 0.504434],
                 [0.270595, 0.214069, 0.507052],
                 [0.269308, 0.218818, 0.509577],
                 [0.267968, 0.223549, 0.512008],
                 [0.266580, 0.228262, 0.514349],
                 [0.265145, 0.232956, 0.516599],
                 [0.263663, 0.237631, 0.518762],
                 [0.262138, 0.242286, 0.520837],
                 [0.260571, 0.246922, 0.522828],
                 [0.258965, 0.251537, 0.524736],
                 [0.257322, 0.256130, 0.526563],
                 [0.255645, 0.260703, 0.528312],
                 [0.253935, 0.265254, 0.529983],
                 [0.252194, 0.269783, 0.531579],
                 [0.250425, 0.274290, 0.533103],
                 [0.248629, 0.278775, 0.534556],
                 [0.246811, 0.283237, 0.535941],
                 [0.244972, 0.287675, 0.537260],
                 [0.243113, 0.292092, 0.538516],
                 [0.241237, 0.296485, 0.539709],
                 [0.239346, 0.300855, 0.540844],
                 [0.237441, 0.305202, 0.541921],
                 [0.235526, 0.309527, 0.542944],
                 [0.233603, 0.313828, 0.543914],
                 [0.231674, 0.318106, 0.544834],
                 [0.229739, 0.322361, 0.545706],
                 [0.227802, 0.326594, 0.546532],
                 [0.225863, 0.330805, 0.547314],
                 [0.223925, 0.334994, 0.548053],
                 [0.221989, 0.339161, 0.548752],
                 [0.220057, 0.343307, 0.549413],
                 [0.218130, 0.347432, 0.550038],
                 [0.216210, 0.351535, 0.550627],
                 [0.214298, 0.355619, 0.551184],
                 [0.212395, 0.359683, 0.551710],
                 [0.210503, 0.363727, 0.552206],
                 [0.208623, 0.367752, 0.552675],
                 [0.206756, 0.371758, 0.553117],
                 [0.204903, 0.375746, 0.553533],
                 [0.203063, 0.379716, 0.553925],
                 [0.201239, 0.383670, 0.554294],
                 [0.199430, 0.387607, 0.554642],
                 [0.197636, 0.391528, 0.554969],
                 [0.195860, 0.395433, 0.555276],
                 [0.194100, 0.399323, 0.555565],
                 [0.192357, 0.403199, 0.555836],
                 [0.190631, 0.407061, 0.556089],
                 [0.188923, 0.410910, 0.556326],
                 [0.187231, 0.414746, 0.556547],
                 [0.185556, 0.418570, 0.556753],
                 [0.183898, 0.422383, 0.556944],
                 [0.182256, 0.426184, 0.557120],
                 [0.180629, 0.429975, 0.557282],
                 [0.179019, 0.433756, 0.557430],
                 [0.177423, 0.437527, 0.557565],
                 [0.175841, 0.441290, 0.557685],
                 [0.174274, 0.445044, 0.557792],
                 [0.172719, 0.448791, 0.557885],
                 [0.171176, 0.452530, 0.557965],
                 [0.169646, 0.456262, 0.558030],
                 [0.168126, 0.459988, 0.558082],
                 [0.166617, 0.463708, 0.558119],
                 [0.165117, 0.467423, 0.558141],
                 [0.163625, 0.471133, 0.558148],
                 [0.162142, 0.474838, 0.558140],
                 [0.160665, 0.478540, 0.558115],
                 [0.159194, 0.482237, 0.558073],
                 [0.157729, 0.485932, 0.558013],
                 [0.156270, 0.489624, 0.557936],
                 [0.154815, 0.493313, 0.557840],
                 [0.153364, 0.497000, 0.557724],
                 [0.151918, 0.500685, 0.557587],
                 [0.150476, 0.504369, 0.557430],
                 [0.149039, 0.508051, 0.557250],
                 [0.147607, 0.511733, 0.557049],
                 [0.146180, 0.515413, 0.556823],
                 [0.144759, 0.519093, 0.556572],
                 [0.143343, 0.522773, 0.556295],
                 [0.141935, 0.526453, 0.555991],
                 [0.140536, 0.530132, 0.555659],
                 [0.139147, 0.533812, 0.555298],
                 [0.137770, 0.537492, 0.554906],
                 [0.136408, 0.541173, 0.554483],
                 [0.135066, 0.544853, 0.554029],
                 [0.133743, 0.548535, 0.553541],
                 [0.132444, 0.552216, 0.553018],
                 [0.131172, 0.555899, 0.552459],
                 [0.129933, 0.559582, 0.551864],
                 [0.128729, 0.563265, 0.551229],
                 [0.127568, 0.566949, 0.550556],
                 [0.126453, 0.570633, 0.549841],
                 [0.125394, 0.574318, 0.549086],
                 [0.124395, 0.578002, 0.548287],
                 [0.123463, 0.581687, 0.547445],
                 [0.122606, 0.585371, 0.546557],
                 [0.121831, 0.589055, 0.545623],
                 [0.121148, 0.592739, 0.544641],
                 [0.120565, 0.596422, 0.543611],
                 [0.120092, 0.600104, 0.542530],
                 [0.119738, 0.603785, 0.541400],
                 [0.119512, 0.607464, 0.540218],
                 [0.119423, 0.611141, 0.538982],
                 [0.119483, 0.614817, 0.537692],
                 [0.119699, 0.618490, 0.536347],
                 [0.120081, 0.622161, 0.534946],
                 [0.120638, 0.625828, 0.533488],
                 [0.121380, 0.629492, 0.531973],
                 [0.122312, 0.633153, 0.530398],
                 [0.123444, 0.636809, 0.528763],
                 [0.124780, 0.640461, 0.527068],
                 [0.126326, 0.644107, 0.525311],
                 [0.128087, 0.647749, 0.523491],
                 [0.130067, 0.651384, 0.521608],
                 [0.132268, 0.655014, 0.519661],
                 [0.134692, 0.658636, 0.517649],
                 [0.137339, 0.662252, 0.515571],
                 [0.140210, 0.665859, 0.513427],
                 [0.143303, 0.669459, 0.511215],
                 [0.146616, 0.673050, 0.508936],
                 [0.150148, 0.676631, 0.506589],
                 [0.153894, 0.680203, 0.504172],
                 [0.157851, 0.683765, 0.501686],
                 [0.162016, 0.687316, 0.499129],
                 [0.166383, 0.690856, 0.496502],
                 [0.170948, 0.694384, 0.493803],
                 [0.175707, 0.697900, 0.491033],
                 [0.180653, 0.701402, 0.488189],
                 [0.185783, 0.704891, 0.485273],
                 [0.191090, 0.708366, 0.482284],
                 [0.196571, 0.711827, 0.479221],
                 [0.202219, 0.715272, 0.476084],
                 [0.208030, 0.718701, 0.472873],
                 [0.214000, 0.722114, 0.469588],
                 [0.220124, 0.725509, 0.466226],
                 [0.226397, 0.728888, 0.462789],
                 [0.232815, 0.732247, 0.459277],
                 [0.239374, 0.735588, 0.455688],
                 [0.246070, 0.738910, 0.452024],
                 [0.252899, 0.742211, 0.448284],
                 [0.259857, 0.745492, 0.444467],
                 [0.266941, 0.748751, 0.440573],
                 [0.274149, 0.751988, 0.436601],
                 [0.281477, 0.755203, 0.432552],
                 [0.288921, 0.758394, 0.428426],
                 [0.296479, 0.761561, 0.424223],
                 [0.304148, 0.764704, 0.419943],
                 [0.311925, 0.767822, 0.415586],
                 [0.319809, 0.770914, 0.411152],
                 [0.327796, 0.773980, 0.406640],
                 [0.335885, 0.777018, 0.402049],
                 [0.344074, 0.780029, 0.397381],
                 [0.352360, 0.783011, 0.392636],
                 [0.360741, 0.785964, 0.387814],
                 [0.369214, 0.788888, 0.382914],
                 [0.377779, 0.791781, 0.377939],
                 [0.386433, 0.794644, 0.372886],
                 [0.395174, 0.797475, 0.367757],
                 [0.404001, 0.800275, 0.362552],
                 [0.412913, 0.803041, 0.357269],
                 [0.421908, 0.805774, 0.351910],
                 [0.430983, 0.808473, 0.346476],
                 [0.440137, 0.811138, 0.340967],
                 [0.449368, 0.813768, 0.335384],
                 [0.458674, 0.816363, 0.329727],
                 [0.468053, 0.818921, 0.323998],
                 [0.477504, 0.821444, 0.318195],
                 [0.487026, 0.823929, 0.312321],
                 [0.496615, 0.826376, 0.306377],
                 [0.506271, 0.828786, 0.300362],
                 [0.515992, 0.831158, 0.294279],
                 [0.525776, 0.833491, 0.288127],
                 [0.535621, 0.835785, 0.281908],
                 [0.545524, 0.838039, 0.275626],
                 [0.555484, 0.840254, 0.269281],
                 [0.565498, 0.842430, 0.262877],
                 [0.575563, 0.844566, 0.256415],
                 [0.585678, 0.846661, 0.249897],
                 [0.595839, 0.848717, 0.243329],
                 [0.606045, 0.850733, 0.236712],
                 [0.616293, 0.852709, 0.230052],
                 [0.626579, 0.854645, 0.223353],
                 [0.636902, 0.856542, 0.216620],
                 [0.647257, 0.858400, 0.209861],
                 [0.657642, 0.860219, 0.203082],
                 [0.668054, 0.861999, 0.196293],
                 [0.678489, 0.863742, 0.189503],
                 [0.688944, 0.865448, 0.182725],
                 [0.699415, 0.867117, 0.175971],
                 [0.709898, 0.868751, 0.169257],
                 [0.720391, 0.870350, 0.162603],
                 [0.730889, 0.871916, 0.156029],
                 [0.741388, 0.873449, 0.149561],
                 [0.751884, 0.874951, 0.143228],
                 [0.762373, 0.876424, 0.137064],
                 [0.772852, 0.877868, 0.131109],
                 [0.783315, 0.879285, 0.125405],
                 [0.793760, 0.880678, 0.120005],
                 [0.804182, 0.882046, 0.114965],
                 [0.814576, 0.883393, 0.110347],
                 [0.824940, 0.884720, 0.106217],
                 [0.835270, 0.886029, 0.102646],
                 [0.845561, 0.887322, 0.099702],
                 [0.855810, 0.888601, 0.097452],
                 [0.866013, 0.889868, 0.095953],
                 [0.876168, 0.891125, 0.095250],
                 [0.886271, 0.892374, 0.095374],
                 [0.896320, 0.893616, 0.096335],
                 [0.906311, 0.894855, 0.098125],
                 [0.916242, 0.896091, 0.100717],
                 [0.926106, 0.897330, 0.104071],
                 [0.935904, 0.898570, 0.108131],
                 [0.945636, 0.899815, 0.112838],
                 [0.955300, 0.901065, 0.118128],
                 [0.964894, 0.902323, 0.123941],
                 [0.974417, 0.903590, 0.130215],
                 [0.983868, 0.904867, 0.136897],
                 [0.993248, 0.906157, 0.143936]]

def register_layout_nodes():
    Node.app.register_node('llm_layout', TextLayoutNode.factory)
    Node.app.register_node('cairo_layout', CairoTextLayoutNode.factory)


class TextLayoutNode(Node):
    llama_nodes = []

    @staticmethod
    def factory(name, data, args=None):
        node = TextLayoutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.sleep = 0.00
        self.layout = None
        self.input = self.add_input('input', triggers_execution=True)
        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear_layout)
        self.active_line = self.add_input('active_line', widget_type='input_int', default_value=17, callback=self.active_line_changed)
        self.color_mode_input = self.add_input('colour_mode', widget_type='combo', default_value='temperature')
        self.color_mode_input.widget.combo_items = ['temperature', 'entropy', 'probability']
        self.include_prompt = self.add_input('include prompt', widget_type='checkbox', default_value=True)
        self.image_output = self.add_output('layout')
        self.previous_layout_length = 0
        self.cmap = _viridis_data
        self.cmap2 = make_heatmap()
        self.cmap3 = make_coldmap()
        self.layout = LLMLayout([0, 0, 1920, 1080])
        self.layout.set_active_line(17)  # 17
        # self.active_line.set(17)
        self.layout.show_list = False  # False
        self.show_choices = False
        self.chosen_index = 0
        self.speech_lines = 0
        self.max_speech_lines = 100
        self.temp = 1.0
        self.poss_dict = []
        self.new_poss_dict = False
        self.showing_poss_dict = False
        self.streaming_prompt_active = False
        self.streaming_prompt_pos = 0
        self.entropy = 0.1
        self.probability = 0.5

    def execute(self):
        if self.layout is not None:
            data = self.input()
            t = type(data)
            if t == str:
                data = string_to_list(data)
                t = list
            if t == list:
                if type(data[0]) == str:
                    if data[0] == 'save':
                        dir = os.getcwd()
                        date = time.strftime("%d-%m-%Y-%H_%M_%S")
                        self.layout.save_layout_as_text(dir + os.sep + 'llama_output_' + date + '.txt')
                    if data[0] == 'add':
                        self.previous_layout_length = len(self.layout.layout)
                        self.add_text([data[1:]])
                    elif data[0] == 'step_back':
                        count = any_to_int(data[1])
                        self.step_back(count, redraw=False)
                        self.streaming_prompt_active = False
                    # elif data[0] == 'temperature':
                    #     self.temp = any_to_float(data[1])
                    #     self.layout.temp = self.temp
                    # elif data[0] == 'entropy':
                    #     self.entropy = any_to_float(data[1])
                    #     self.layout.entropy = self.entropy
                    # elif data[0] == 'probability':
                    #     self.probability = any_to_float(data[1])
                    elif data[0] == 'prompt':
                        if self.include_prompt():
                            temp_temp = self.temp
                            self.temp = 0.8
                            self.add_text([data[1:]])
                            self.temp = temp_temp
                    elif data[0] == 'streaming_prompt':
                        if not self.streaming_prompt_active:
                            self.streaming_prompt_active = True
                            if self.layout.show_list:
                                self.streaming_prompt_pos = self.previous_layout_length
                            else:
                                self.streaming_prompt_pos = len(self.layout.layout)
                        delete_element_count = len(self.layout.layout) - self.streaming_prompt_pos
                        if delete_element_count > 0:
                            self.layout.step_back(delete_element_count)
                        if len(data) > 1:
                            pairs = data[1]
                            self.add_text(pairs)
                    elif data[0] == 'accept_streamed_prompt':
                        if self.streaming_prompt_active:
                            self.streaming_prompt_active = False
                    elif data[0] == 'backspace_streaming_prompt':
                        if self.streaming_prompt_active and len(self.layout.layout) > self.streaming_prompt_pos:
                            if len(self.layout.layout[-1][1]) > 0:
                                self.layout.layout[-1][1] = self.layout.layout[-1][1][:-1]
                            self.display_layout()
                    elif data[0] == 'choice_list':
                        self.add_text_with_choices(data[1], data[2])
                    elif data[0] == 'choose':
                        self.chosen_index = data[1]
                        self.layout.choose_from_next_word_list(self.chosen_index)
                        self.display_layout()
                    elif data[0] == 'show_probs':
                        self.showing_poss_dict = data[1]
                        old_active_line = self.layout.active_line
                        if self.showing_poss_dict:
                            self.layout.set_active_line(5)
                            self.active_line.set(5)
                        else:
                            self.layout.set_active_line(17)
                            self.active_line.set(17)
                        self.layout.adjust_active_line(old_active_line)
                        self.display_layout()
                    elif data[0] == 'clear' or data[0] == 'reset':
                        self.clear_layout()
                    elif data[0] == 'scroll_up':
                        self.scroll_up()
                    elif data[0] == 'scroll_down':
                        self.scroll_down()

    def scroll_up(self):
        old_active_line = self.layout.active_line
        old_active_line -= 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def scroll_down(self):
        old_active_line = self.layout.active_line
        old_active_line += 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def active_line_changed(self):
        old_active_line = self.layout.active_line
        self.layout.active_line = self.active_line()
        self.layout.adjust_active_line(old_active_line)
        self.display_layout()

    def clear_layout(self):
        self.layout.clear_layout()
        self.display_layout()

    def add_text_with_choices(self, poss_dict, selected ):
        self.poss_dict = poss_dict
        self.chosen_index = selected
        self.new_poss_dict = True

    def add_text(self, new_data):
        # if type(new_data) == list:
        #     if type(new_data[0]) == list:
        #         new_data = new_data[0]
        out_string = ''
        for t in new_data:
            colour_index = None
            if len(t) == 1:
                t = [0, t[0]]
            if len(t) > 2:
                colour_index = t[2]
            out_string += t[1]

            spread_color = [255, 255, 255]

            if colour_index is not None:
                if self.color_mode_input() == 'temperature':
                    temper = colour_index

                    if temper > 1:
                        temper = int((1.0 - ((temper - 1.0) / 8)) * 255)
                        if temper < 0:
                            temper = 0
                        elif temper > 255:
                            temper = 255
                        spread_color = (self.cmap2[temper][0], self.cmap2[temper][1], self.cmap2[temper][2])
                    else:
                        temper = int(((temper - 0.5) * 2) * 255)
                        if temper < 0:
                            temper = 0
                        elif temper > 255:
                            temper = 255
                        spread_color = (self.cmap3[temper][0], self.cmap3[temper][1], self.cmap3[temper][2])
                elif self.color_mode_input() == 'entropy':
                    entropy = colour_index
                    entropy *= 64
                    if entropy < 0:
                        entropy = 0
                    elif entropy > 255:
                        entropy = 255
                    entropy = 255 - int(entropy)
                    print(entropy)

                    spread_color = (self.cmap[entropy][0], self.cmap[entropy][1], self.cmap[entropy][2])
                else:
                    probability = colour_index
                    probability *= 255
                    if probability < 0:
                        probability = 0
                    elif probability > 255:
                        probability = 255
                    probability = int(probability)
                    spread_color = (self.cmap[probability][0], self.cmap[probability][1], self.cmap[probability][2])
            if '\\' in t[1]:
                t[1].replace('\\n', '\n')
                if t[1][0] == '\\' and t[1][1] == 'c':
                    self.clear_layout()


            self.layout.add_word(t[1], spread_color, t[0])
            if self.new_poss_dict:
                self.layout.show_list = True
                self.layout.clear_list()
                spread_color = self.layout.display_next_word_list_all(self.poss_dict, self.chosen_index)
                self.new_poss_dict = False
            else:
                self.layout.show_list = False

            self.display_layout()

    def step_back(self, count=2, redraw=True):
        if len(self.layout.layout) > count:
            self.layout.step_back(step_size=count)
            if redraw:
                self.display_layout()

    def display_layout(self):
        self.layout.draw_layout()
        self.idle_routine()

    def reset_layout(self):
        self.layout.clear_layout()

    def idle_routine(self, dt=0.016):
        if self.layout:
            layout_image = self.layout.dest[..., 0:3].astype(np.float32) / 255
            if self.layout.do_animate_scroll:
                self.layout.animate_scroll(self.layout.scroll_delta)

            if layout_image is not None:
                self.image_output.send(layout_image)

    def make_heatmap(self):
        red = [1.0, 0.0, 0.0]
        orange = [1.0, 0.5, 0.0]
        yellow = [1.0, 1.0, 0.0]
        white = [1.0, 1.0, 1.0]
        cmap = []

        colors = [red, orange, yellow, white]
        positions = [0, 85, 170, 256]

        for idx, color in enumerate(colors):
            start_position = positions[idx]
            if idx < len(colors) - 1:
                position = start_position
                next_position = positions[idx + 1]
                next_color = colors[idx + 1]
                while position < next_position:
                    progress = float(position - start_position) / float(next_position - start_position)
                    this_color = [0.0, 0.0, 0.0]
                    this_color[0] = color[0] * (1.0 - progress) + next_color[0] * progress
                    this_color[1] = color[1] * (1.0 - progress) + next_color[1] * progress
                    this_color[2] = color[2] * (1.0 - progress) + next_color[2] * progress
                    cmap.append(this_color)
                    position += 1
        return cmap

    def make_coldmap(self):
        blue = [0.5, 0.0, 0.8]
        green = [0.5, 0.3, 1.0]
        yellow = [0.5, 0.8, 1.0]
        white = [1.0, 1.0, 1.0]
        cmap = []

        colors = [blue, green, yellow, white]
        positions = [0, 85, 170, 256]

        for idx, color in enumerate(colors):
            start_position = positions[idx]
            if idx < len(colors) - 1:
                position = start_position
                next_position = positions[idx + 1]
                next_color = colors[idx + 1]
                while position < next_position:
                    progress = float(position - start_position) / float(next_position - start_position)
                    this_color = [0.0, 0.0, 0.0]
                    this_color[0] = color[0] * (1.0 - progress) + next_color[0] * progress
                    this_color[1] = color[1] * (1.0 - progress) + next_color[1] * progress
                    this_color[2] = color[2] * (1.0 - progress) + next_color[2] * progress
                    cmap.append(this_color)
                    position += 1
        return cmap


class CairoTextLayoutNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = CairoTextLayoutNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        width = 1920
        height = 1080
        if len(args) > 1:
            width = any_to_int(args[0])
            height = any_to_int(args[1])
        self.sleep = 0.00
        self.layout = None
        self.input = self.add_input('input', triggers_execution=True)
        self.command_input = self.add_input('command', triggers_execution=True)
        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear_layout)
        self.font_input = self.add_input('font path', widget_type='text_input', default_value='', callback=self.font_changed)
        self.font_size_input = self.add_input('font size', widget_type='drag_float', default_value=40, callback=self.font_size_changed)
        self.text_brightness = self.add_input('brightness', widget_type='drag_float', default_value=1.0)
        self.alpha_power = self.add_input('alpha power', widget_type='drag_float', default_value=0.1)
        self.leading_input = self.add_input('leading', widget_type='drag_float', default_value=60, callback=self.leading_changed)
        self.active_line = self.add_input('active_line', widget_type='input_int', default_value=17, callback=self.active_line_changed)
        self.wrap_input = self.add_input('wrap text', widget_type='checkbox', default_value=True, callback=self.wrap_changed)
        self.image_output = self.add_output('layout')
        self.language = None

        self.layout = CairoTextLayout([0, 0, width, height])
        self.layout.set_active_line(1)  # 17

    def leading_changed(self):
        self.layout.leading = self.leading_input()

    def wrap_changed(self):
        self.layout.wrap_text = self.wrap_input()

    def font_changed(self):
        self.layout.get_font(self.font_input())

    def font_size_changed(self):
        self.layout.set_font_size(self.font_size_input())

    def execute(self):
        if self.layout is not None:
            if self.active_input is self.input:
                data = self.input()
                self.clear_layout()
                self.add_text(data)
            elif self.active_input is self.command_input:
                data = self.command_input()
                t = type(data)
                if t == str:
                    data = string_to_list(data)
                    t = list
                if t == list:
                    if type(data[0]) == str:
                        if data[0] == 'save':
                            dir = os.getcwd()
                            date = time.strftime("%d-%m-%Y-%H_%M_%S")
                            self.layout.save_layout_as_text(dir + os.sep + 'llama_output_' + date + '.txt')
                        if data[0] == 'add':
                            self.add_text(data[1:])
                        elif data[0] == 'add_char':
                            if len(data) > 1:
                                added_char = data[1]
                                if added_char == '':
                                    added_char = ' '
                            else:
                                added_char = ' '
                            self.add_text(added_char, add_space=False)
                        elif data[0] == 'clear' or data[0] == 'reset':
                            self.clear_layout()
                        elif data[0] == 'scroll_up':
                            self.scroll_up()
                        elif data[0] == 'scroll_down':
                            self.scroll_down()
                        elif data[0] == 'delete_line':
                            self.delete_line()

    def scroll_up(self):
        old_active_line = self.layout.active_line
        old_active_line -= 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def scroll_down(self):
        old_active_line = self.layout.active_line
        old_active_line += 1
        self.active_line.set(old_active_line)
        self.active_line_changed()

    def active_line_changed(self):
        old_active_line = self.layout.active_line
        self.layout.active_line = self.active_line()
        self.layout.adjust_active_line(old_active_line)
        self.display_layout()

    def clear_layout(self):
        self.layout.clear_layout()
        self.display_layout()

    def delete_line(self):
        self.layout.step_back_to_last_return()
        self.display_layout()

# ISSUE HERE OF OVERNESTED LIST
    def add_text(self, new_data, add_space=True):
        tp = type(new_data)
        if new_data == ' ':
            new_data = [' ']
            tp = type(new_data)
        if tp is str:
            new_data = string_to_list(new_data)
        else:
            new_data = any_to_list(new_data)
        for t in new_data:
            if type(t) == str:
                t = [t, 1.0]
            elif type(t) is not list:
                t = [any_to_string(t), 1.0]
            if len(t) > 1:
                tt = t.copy()
                tt[1] = pow(tt[1], self.alpha_power()) * self.text_brightness()
                if tt[1] != 0.0:
                    if '\\' in tt[0]:
                        tt[0].replace('\\n', '\n')
                        if tt[0] == '\\c':
                            self.clear_layout()
                            tt[0] = ''
                        elif tt[0] == '\\d':
                            print('deleting')
                            self.delete_line()
                            tt[0] = ''
                    self.layout.add_string([tt], add_space)

        self.display_layout()

    def step_back(self):
        if len(self.layout.layout) > 2:
            self.layout.step_back(step_size=2)
            self.display_layout()

    def display_layout(self):
        self.layout.draw_layout()
        self.idle_routine()

    def reset_layout(self):
        self.layout.clear_layout()

    def idle_routine(self, dt=0.016):
        if self.layout:
            layout_image = self.layout.dest[..., 0:3].astype(np.float32) / 255
            # if self.layout.do_animate_scroll:
            #     self.layout.animate_scroll(self.layout.scroll_delta)

            if layout_image is not None:
                self.image_output.send(layout_image)
