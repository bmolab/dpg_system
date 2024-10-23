
import numpy as np

import sys
from dpg_system.Cairo_Utils import *
# import config
from dpg_system.colormaps import _viridis_data, _magma_data, _plasma_data, _inferno_data, make_heatmap, make_coldmap
from tkinter.filedialog import askopenfilename, asksaveasfile
from pathlib import Path
import collections
from numba import jit
import platform
# import nltk
print('?')
free_run = True
generator = None
stop_on_punctuation = False
stop_on_return = False
stop_on_double_return = False
stop_after_speaker = False
osc_transmit = True

paragraph_indent_scaler = 0.0

# layout element is [current_position, word, color]

NOPRINT_TRANS_TABLE = {
    i: None for i in range(0, sys.maxunicode + 1) if not chr(i).isprintable()
}

def make_printable(s):
    """Replace non-printable characters in a string."""

    # the translate method on str removes characters
    # that map to None from the string
    return s.translate(NOPRINT_TRANS_TABLE)

class LLMLayout:
    def __init__(self, frame, font_file_path=''):
        global paragraph_indent_scaler
        self.frame = frame
        self.layout = []
        self.input_in_progress = []
        self.font_size = 40
        self.leading = int(self.font_size * 1.5)
        self.list_leading = self.font_size
        self.list = []
        self.paragraph_indent = self.font_size * paragraph_indent_scaler
        self.show_list = False
        self.choice_menu_position = [0, 0]
        self.choice_item_position = [0, 0]
        self.incomplete = False
        self.face = []
        self.chosen_id = -1
        self.forced_newline = False
#        frame = [frame[0] * 2, frame[1] * 2, frame[2] * 2, frame[3] * 2]
        self.cr, self.dest = self.prepare_drawing(frame)
        self.sorted_poss = None
        self.sorted_poss_ = None
        self.cmap = _viridis_data
        self.cmap2 = make_heatmap()
        self.cmap3 = make_coldmap()
        self.text_color_mode = 'white'
        self.choice_width = 0
        self.selected_word = ''
        self.dashboard_height = 0
        self.get_font(font_file_path)
        self.which_font = 0
        self.util_font = 0
        # self.position_target = [0, 0]
        self.scroll_delta = self.leading / 3.0
        self.do_animate_scroll = False
        self.active_line = 8
        self.cursor_position = [0, self.active_line * self.leading]

        self.most_prob = 0
        self.last_element_to_display = -1
        self.force_newline_next = False
        self.set_active_line(self.active_line)
        self.in_progress_start = 0
        self.in_progress_position = self.cursor_position.copy()
        self.in_progress_previous_position = self.cursor_position.copy()
        self.temp = 1.0
        self.draw_layout()

    def prepare_drawing(self, frame):
        cr, dest_image = set_up_canvas(frame)
        return cr, dest_image

    def set_active_line(self, line):
        pos = self.cursor_position
        self.active_line = line
        self.cursor_position = [pos[0], self.active_line * self.leading]

    def get_font(self, path):
        if path != '':
            self.face.append(create_cairo_font_face_for_file(path, 0))
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 0))
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/n021003l.pfb", 0))
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/b018012l.pfb", 0))  # Bookman Light
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/c059013l.pfb", 0))  # New Century Schoolbook
### for MacOS: 
        else:
            platform_ = platform.system()
            if platform_ == 'Darwin':
                self.face.append(create_cairo_font_face_for_file("/Users/drokeby/Library/Fonts/TradeGothicLTStd.otf", 0))  # utility
            elif platform_ == 'Linux':
                self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/c059013l.pfb",
                                                                 0))  # New Century Schoolbook
        # self.face.append(create_cairo_font_face_for_file("/Library/Fonts/RODE Noto Sans CJK SC R.otf", 0))  # utility

        self.cr.set_font_face(self.face[0])
        self.cr.set_font_size(self.font_size)

    def clear_layout(self):
        del self.layout
        self.layout = []
        self.cursor_position = [0, self.active_line * self.leading]

    def clear_input_in_progress(self):
        del self.input_in_progress
        self.input_in_progress = []
        self.cursor_position = [0, (self.active_line + 1) * self.leading]

    def clear_list(self):
        self.list = []

    def save_layout_text_as(self):
        default_path = str(Path.home())
        file = asksaveasfile(mode='w', defaultextension=".txt", title='Save File As', initialdir=str(default_path))
        if file:
            for element in self.layout:
                file.write(element[1])
            file.flush()
            file.close()

    def save_layout_as_text(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            for element in self.layout:
                f.write(element[1])
            f.flush()
            f.close()

    @jit
    def add_word_with_probability(self, word, prob, token_id):
        color_index = int(prob * 255.0)
        prob_color = (self.cmap[color_index][0], self.cmap[color_index][1], self.cmap[color_index][2])
        self.add_word(word, prob_color, token_id)

    def add_word_with_spread(self, word, spread, token_id):
        color_index = int(spread * 255.0)
        spread_color = (self.cmap[color_index][0], self.cmap[color_index][1], self.cmap[color_index][2])
        self.add_word(word, spread_color, token_id)

    def add_word_with_temp(self, word, spread, token_id):
        color_index = int(spread * 127.0)
        spread_color = (self.cmap[color_index][0], self.cmap[color_index][1], self.cmap[color_index][2])
        self.add_word(word, spread_color, token_id)

    def add_word(self, word, color, token_id, speech_in_progress=False):
        new_element = [self.cursor_position.copy(), word, color, token_id]
        self.add_element_to_layout(new_element)
        if not speech_in_progress:
            self.in_progress_position = self.cursor_position.copy()
            self.in_progress_start = len(self.layout)

    def mark_speech_in_progress_start(self):
        self.in_progress_position = self.cursor_position.copy()
        self.in_progress_start = len(self.layout)
        if self.in_progress_start > 0:
            self.in_progress_previous_position = self.layout[self.in_progress_start - 1][0].copy()
        else:
            self.in_progress_previous_position = self.in_progress_position.copy()

    # def add_input_in_progress(self, word):
    #     self.remove_in_progress()
    #     self.in_progress_start = len(self.layout)
    #     input_list = nltk.word_tokenize()
    #     for word in input_list:
    #         new_element = [self.cursor_position.copy(), word, (255, 255, 255), -1]
    #         self.add_element_to_layout(new_element)
    #         if word != '':
    #             if word[0] == '\n':
    #                 self.layout_new_line()
    #                 self.cursor_position[0] = self.paragraph_indent

    def trailing_returns(self):
        for i in range(len(self.layout)):
            j = 1 + 1
            if self.layout[1][-j] != '\n':
                return i
        return len(self.layout)

    def remove_in_progress(self):
        new_layout = self.layout[0:self.in_progress_start].copy()
        pos = new_layout[-1][0]
        diffY = pos[1] - self.in_progress_previous_position[1]
        if diffY != 0:
            for element in new_layout:
                element[0][1] -= diffY
        self.cursor_position[0] = 0

        del self.layout
        self.layout = new_layout.copy()
        self.cursor_position = self.in_progress_position.copy()
        self.forced_newline = False
        self.force_newline_next = False

    def add_element_to_layout(self, element):
        extents = self.cr.text_extents(element[1])
        advance = extents.x_advance
        internal_returns = 0
        wrapped = False
        if len(element[1]) > 0:
            # n.b. llama3 tokens might include \n in places other than [0]
            if '\n' in element[1]:
                subs = element[1].split('\n')
                if len(subs) > 0:
                    internal_returns = len(subs) - 1
                extents = self.cr.text_extents(subs[-1])
                advance = extents.x_advance

            if len(self.layout) > 0 and element[1][0].isalpha():  # this element is a continuation of a previous word
                previous_element = self.layout[-1]
                previous_word = previous_element[1]
                previous_extents = self.cr.text_extents(previous_word)
                previous_end = previous_element[0].copy()
                previous_end[0] += previous_extents.x_advance

                if '\n' in previous_word:
                    previous_subs = previous_word.split('\n')
                    if len(previous_subs) > 0:
                        previous_internal_returns = len(previous_subs) - 1
                        previous_end[1] += (self.leading * previous_internal_returns)
                    if previous_subs[-1] == '':
                        previous_end[0] = self.paragraph_indent
                    else:
                        sub_extent = self.cr.text_extents(previous_subs[-1])
                        previous_end[0] = self.paragraph_indent + sub_extent.x_advance
                else:
                    if previous_end[0] + advance > self.frame[0] + self.frame[2]:
                        self.layout[-1][0][0] = 0
                        self.layout[-1][0][1] += self.leading
                        element[0] = previous_end.copy()
                        wrapped = True
            elif element[0][0] + advance > self.frame[0] + self.frame[2]:
                element[0] = [0, element[0][1] + self.leading]
                wrapped = True
            last_fragment_start_x = element[0][0]
            if internal_returns > 0:
                last_fragment_start_x = 0
            self.layout.append(element)
            # n.b. if a wrapp caused a return, then choice list will be one too many lines down after scroll to active line
            self.set_choice_menu_position()
            if wrapped:
                self.choice_menu_position[1] -= self.leading
            self.cursor_position = [last_fragment_start_x + extents.x_advance, element[0][1] + internal_returns * self.leading]

    def set_choice_menu_position(self):
        self.choice_menu_position = [self.layout[-1][0][0], self.layout[-1][0][1] + self.list_leading]

    def add_word_with_probability_to_list(self, word, prob, pos, token_id):
        color_index = int(prob * 255.0)
        prob_color = (self.cmap[color_index][0], self.cmap[color_index][1], self.cmap[color_index][2])
        self.add_word_to_list(word, prob_color, pos, token_id)

    def add_word_with_spread_to_list(self, word, spread, pos, token_id):
        color_index = int(spread * 255.0)
        spread_color = (self.cmap[color_index][0], self.cmap[color_index][1], self.cmap[color_index][2])
        self.add_word_to_list(word, spread_color, pos, token_id)

    def add_word_to_list(self, word, color, pos, token_id):
        new_element = [pos.copy(), word, color, token_id]  #  pos.copy() relative to reference
        self.add_element_to_list(new_element)

    def add_element_to_list(self, element):
        self.list.append(element)

    # def layout_new_line(self):
    #     self.scroll_position_to_active_line()
    #     self.cursor_position[0] = self.paragraph_indent

    # def undo_new_line(self, animate=False):
    #     self.scroll_position_to_active_line()
    #
    #     last_element = self.layout[-1]                                  # get last element of layout
    #     self.cursor_position = last_element[0].copy()
    #     extents = self.cr.text_extents(last_element[1])                 # calc extents of last element
    #     self.cursor_position[0] += extents.x_advance                           # plus x_advance

    def scroll_position_to_active_line(self):
        delta = (self.active_line * self.leading) - self.cursor_position[1]
        if delta != 0:
            for element in self.layout:               # shift all elements down one line
                element[0][1] += delta
        self.cursor_position[1] = self.active_line * self.leading
        self.choice_menu_position[1] += delta

    def adjust_active_line(self, old_active_line):
        delta = (self.active_line - old_active_line) * self.leading  # was -1
        for index, element in enumerate(self.layout):  # shift all elements down one line
            element[0][1] += delta

    def internal_returns(self, word):
        subs = word.split('\n')
        return len(subs) - 1
    
    def move_cursor_to_end(self):
        if len(self.layout) > 0:
            element = self.layout[-1]
            self.move_cursor_to_end_of_element(element)
        else:
            self.cursor_position = [0, 0]

    def move_cursor_to_end_of_element(self, element):
        word = element[1]
        position = element[0].copy()
        internal_returns = 0
        advance = 0
        if len(word) > 0:
            subs = word.split('\n')
            internal_returns = len(subs) - 1
            advance = self.cr.text_extents(subs[-1]).x_advance
            if internal_returns > 0:
                position[0] = self.paragraph_indent
        self.cursor_position = [position[0] + advance, position[1] + internal_returns * self.leading]
        
    def step_back(self, step_size):
        if len(self.layout) > step_size:
            del self.layout[-step_size:]                                # delete section     
            self.move_cursor_to_end()

    def get_choice_by_index(self, index):
        item_list = list(self.sorted_poss)
        if index >= len(self.sorted_poss):
            index = len(self.sorted_poss) - 1
        elif index < 0:
            index = 0
        word = self.sorted_poss[index][0]
        token_id = self.sorted_poss[index][2]  # WE MUST SOLVE THIS.... CHOICE DICT NEEDS TO ALSO HOLD TOKEN ID,,, MAKE A SEPARATE DICT THAT TRANSLATES TOKEN ID's ITNO WORDS
        return word, token_id

#    @jit
    def draw_element_at_position(self, element):
        # note: if element includes return characters, we should handle this here
        pos = element[0].copy()
        self.cr.move_to(pos[0], pos[1])
        self.cr.set_source_rgb(element[2][2], element[2][1], element[2][0])
        string = element[1]
        subs = string.split('\n')
        for index, sub in enumerate(subs):
            text = make_printable(sub)
            self.cr.show_text(text)
            if index < len(subs) - 1:
                pos[1] += self.leading
                pos[0] = self.paragraph_indent
                self.cr.move_to(pos[0], pos[1])

 #   @jit
    def draw_layout(self):
        self.scroll_position_to_active_line()
        self.erase_rect()
        self.cr.set_font_face(self.face[self.which_font])
        self.cr.set_font_size(self.font_size)

        for index, element in enumerate(self.layout):
            if element[0][1] > self.dashboard_height:
                if len(element[1]) > 0:
                    self.draw_element_at_position(element)
        if self.show_list:
            for element in self.list:
                self.draw_element_at_position(element)
        if self.show_list:
            self.draw_arrow()

        target = self.cr.get_target()
        target.flush()


    def erase_rect(self):
        self.cr.rectangle(self.frame[0], self.frame[1], self.frame[2], self.frame[3])
        self.cr.set_source_rgb(0.0, 0.0, 0.0)
        self.cr.set_operator(cairo.OPERATOR_OVER)
        self.cr.fill()
        self.cr.set_source_rgb(1.0, 1.0, 1.0)

    def set_choice_item_position_to_chosen_id(self):
        self.choice_item_position = list(self.list[self.chosen_id][0])
        choice_extents = self.cr.text_extents(self.list[self.chosen_id][1])
        self.choice_width = choice_extents.width + choice_extents.x_bearing

    def choose_from_next_word_list(self, new_choice_index):
        # ***** IF WORD CHANGES FROM A WORD WITH A CR TO ONE WITHOUT, THE NEWLINE IS NOT UNDONE
        # SHOULD THIS BE HANDLED HERE?
        # CHOOSING A CHOICE WITH A DIFFERENT LENGTH NEVER FORCES A NEW LINE BECAUSE WE CALCULATE MAX LENGTH OF OPTIONS
        # CHOOSING A CHOICE THAT CHANGES THE NUMBER OF RETURNS IN THE CHOICE MIGHT HAVE SOME PROBLEMATIC EFFECTS
        # when the next word suggested includes a return, when is that return registered in the layout?

        self.chosen_id = new_choice_index
        if len(self.layout) > 0:
            if new_choice_index < len(self.sorted_poss):
                self.layout[-1][1] = self.sorted_poss[new_choice_index][0]
                self.layout[-1][3] = self.sorted_poss[new_choice_index][2]
            self.move_cursor_to_end()
            self.set_choice_item_position_to_chosen_id()

    def display_next_word_list_all(self, poss_dict, chosen_index):
        spread_index = 0
        self.chosen_id = chosen_index
        scaled_prob = []
        lengther = 0
        spread_color = self.cmap2[int(spread_index)]

        if len(poss_dict) > 0:
            self.sorted_poss = poss_dict
            self.most_prob = next(reversed(self.sorted_poss))[1] / self.temp

            self.chosen_id = chosen_index
            total_pos = 0
            for idx, poss in enumerate(self.sorted_poss):
                prob = poss[1]
                val = math.exp(prob / self.temp - self.most_prob)
                # val = math.sqrt(val)
                total_pos += val
                scaled_prob.append(val)

            # for i in range(len(scaled_prob)):
            #     scaled_prob[i] /= total_pos

            max_width = 0
            lengther = len(self.sorted_poss)
            current_total = 0
            spread_captured = False
            if lengther > 0:
                for idx, poss in enumerate(reversed(self.sorted_poss)):
                    word = poss[0]
                    try:
                        prob = scaled_prob[lengther - idx - 1]
                        if len(word) > 0:
                            extents = self.cr.text_extents(word)
                            if extents.x_advance > max_width:
                                max_width = extents.x_advance
                            current_total += prob
                            if current_total > total_pos * 0.75 and not spread_captured:
                                spread_index = idx
                                spread_captured = True
                    except Exception as e:
                        print(word)

            # N.B. self.cursor_position is position AFTER token
            token_start_pos = self.layout[-1][0].copy()
            if len(self.layout) > 0:
                if token_start_pos[0] + max_width > self.frame[0] + self.frame[2]:
                    self.layout[-1][0][0] = 0
                    self.layout[-1][0][1] += self.leading
                    insert_word = self.layout[-1][1]
                    extents = self.cr.text_extents(insert_word)
                    self.cursor_position = [extents.x_advance, self.layout[-1][0][1]]
                    self.set_choice_menu_position()
                    self.choice_menu_position[1] -= self.leading


            for idx, poss in enumerate(self.sorted_poss):
                prob = scaled_prob[idx]
                word = poss[0]
                token_id = poss[2]
                if '\n' in word:
                    subs = word.split('\n')
                    word = '<cr>'.join(subs)
                    self.choice_width = self.cr.text_extents(word)

                choice_menu_item_pos = [self.choice_menu_position[0], self.choice_menu_position[1] + (len(self.sorted_poss) - idx - 1) * self.list_leading]
                self.add_word_with_probability_to_list(word, prob, choice_menu_item_pos, token_id)
            self.set_choice_item_position_to_chosen_id()

        if self.text_color_mode == 'spread':
            if lengther <= 0:
                lengther = .1
            spread_index = ((float(spread_index) / lengther)) * 450
            if spread_index > 255:
                spread_index = 255
            spread_color = self.cmap2[int(spread_index)]
        elif self.text_color_mode == 'probability':
            p = self.sorted_poss[self.chosen_id][1] + 30
            if p < 0:
                p = 0
            prob_index = int(p * 3)
            if prob_index > 255:
                prob_index = 255
            spread_color = self.cmap3[prob_index]
        elif self.text_color_mode == 'white':
            temper = self.temp
            if temper > 1:
                temper = int((1.0 - ((temper - 1.0) / 2)) * 255)
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

        return spread_color

    def locate_mouse_in_layout(self, x, y):
        selection = -1
        choice_selection = -1
        for idx, element in enumerate(self.layout):
            if element[0][0] < x:
                if y + self.leading > element[0][1] > y:
                    selection = idx
            if element[0][1] >= y + self.leading:
                break
        if selection == -1:
            for idx, element in enumerate(self.list):
                if element[0][0] < x:
                    if y + self.list_leading > element[0][1] > y:
                        if element[0][1] > y:
                            choice_selection = idx

        if selection != -1:
            self.selected_word = self.layout[selection][1]
        return selection, choice_selection

    def draw_arrow(self):
        self.cr.set_source_rgb(1.0, 1.0, 1.0)
        self.cr.set_line_width(2)
        center_v = self.choice_item_position[1] - self.list_leading / 2 + self.font_size / 6
        self.cr.move_to(self.choice_item_position[0] + self.choice_width + self.font_size * .8625, center_v - self.font_size / 6)
        self.cr.line_to(self.choice_item_position[0] + self.choice_width + self.font_size * .3625, center_v)
        self.cr.line_to(self.choice_item_position[0] + self.choice_width + self.font_size * .8625, center_v + self.font_size / 6)
        self.cr.stroke()
        self.cr.move_to(self.choice_item_position[0] - self.font_size * .5625, center_v - self.font_size / 6)
        self.cr.line_to(self.choice_item_position[0] - self.font_size * .0625, center_v)
        self.cr.line_to(self.choice_item_position[0] - self.font_size * .5625, center_v + self.font_size / 6)
        self.cr.stroke()

    def draw_cursor(self):
        self.cr.set_source_rgb(0.0, 1.0, 1.0)
        self.cr.move_to(self.cursor_position[0], self.cursor_position[1])
 #       self.cr.show_text(config.generator.edit_text)
 #       extents = self.cr.text_extents(config.generator.edit_text)
 #       pos = self.cursor_position[0] + extents.x_advance
 #       self.cr.set_source_rgb(0.0, 1.0, 1.0)
  #      self.cr.set_line_width(2)
        center_v = self.cursor_position[1] - self.font_size / 3
        self.cr.move_to(self.cursor_position[0], center_v - self.font_size / 2)
        self.cr.line_to(self.cursor_position[0], center_v + self.font_size / 2)
        self.cr.stroke()


class CairoTextLayout:
    def __init__(self, frame, font_file_path=''):
        global paragraph_indent_scaler
        self.frame = frame
        self.layout = []
        self.font_size = 40
        self.leading = int(self.font_size * 1.5)
        self.list_leading = self.font_size
        self.list = []
        self.paragraph_indent = self.font_size * paragraph_indent_scaler
        # self.show_list = False
        # self.choice_menu_position = [0, 0]
        # self.choice_item_position = [0, 0]
        self.incomplete = False
        self.face = []
        # self.chosen_id = -1
        self.forced_newline = False
        #        frame = [frame[0] * 2, frame[1] * 2, frame[2] * 2, frame[3] * 2]
        self.cr, self.dest = self.prepare_drawing(frame)
        # self.sorted_poss = None
        # self.sorted_poss_ = None
        # self.cmap = _viridis_data
        # self.cmap2 = make_heatmap()
        # self.cmap3 = make_coldmap()
        # self.text_color_mode = 'white'
        self.choice_width = 0
        self.selected_word = ''
        self.dashboard_height = 0
        self.get_font(font_file_path)
        self.which_font = 0
        self.util_font = 0
        # self.position_target = [0, 0]
        # self.scroll_delta = self.leading / 3.0
        # self.do_animate_scroll = False
        self.active_line = 8
        self.wrap_text = True
        self.cursor_position = [0, self.active_line * self.leading]

        # self.most_prob = 0
        self.last_element_to_display = -1
        self.force_newline_next = False
        self.set_active_line(self.active_line)
        self.in_progress_start = 0
        self.in_progress_position = self.cursor_position.copy()
        self.in_progress_previous_position = self.cursor_position.copy()
        # self.temp = 1.0
        self.draw_layout()

    def prepare_drawing(self, frame):
        cr, dest_image = set_up_canvas(frame)
        return cr, dest_image

    def set_active_line(self, line):
        pos = self.cursor_position
        self.active_line = line
        self.cursor_position = [pos[0], self.active_line * self.leading]

    def set_font_size(self, size):
        self.font_size = size
        self.cr.set_font_size(self.font_size)
        self.leading = int(self.font_size * 1.5)
        self.list_leading = self.font_size
        self.paragraph_indent = self.font_size * paragraph_indent_scaler

    def get_font(self, path):
        if path != '':
            self.face.append(create_cairo_font_face_for_file(path, 0))
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 0))
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/n021003l.pfb", 0))
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/b018012l.pfb", 0))  # Bookman Light
        # self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/c059013l.pfb", 0))  # New Century Schoolbook
        ### for MacOS:
        else:
            platform_ = platform.system()
            if platform_ == 'Darwin':
                self.face.append(
                    create_cairo_font_face_for_file("/Users/drokeby/Library/Fonts/TradeGothicLTStd.otf",
                                                    0))  # utility
            elif platform_ == 'Linux':
                self.face.append(create_cairo_font_face_for_file("/usr/share/fonts/type1/gsfonts/c059013l.pfb",
                                                                 0))  # New Century Schoolbook
        # self.face.append(create_cairo_font_face_for_file("/Library/Fonts/RODE Noto Sans CJK SC R.otf", 0))  # utility

        self.cr.set_font_face(self.face[- 1])
        self.which_font = len(self.face) - 1
        self.cr.set_font_size(self.font_size)


    def clear_layout(self):
        del self.layout
        self.layout = []
        self.cursor_position = [0, self.active_line * self.leading]

    def save_layout_text_as(self):
        default_path = str(Path.home())
        file = asksaveasfile(mode='w', defaultextension=".txt", title='Save File As', initialdir=str(default_path))
        if file:
            for element in self.layout:
                file.write(element[1])
            file.flush()
            file.close()

    def save_layout_as_text(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            for element in self.layout:
                f.write(element[1])
            f.flush()
            f.close()

    def add_string(self, elements):
        for index, element in enumerate(elements):
            word = element[0]
            alpha = element[1]

            if index == 0:
                the_word = '  ' + word
            else:
                the_word = ' ' + word
            new_element = [self.cursor_position.copy(), the_word, alpha]
            self.add_element_to_layout(new_element)

    def trailing_returns(self):
        for i in range(len(self.layout)):
            j = 1 + 1
            if self.layout[1][-j] != '\n':
                return i
        return len(self.layout)

    def add_element_to_layout(self, element):
        extents = self.cr.text_extents(element[1])
        advance = extents.x_advance
        internal_returns = 0
        wrapped = False
        if len(element[1]) > 0:
            # n.b. llama3 tokens might include \n in places other than [0]
            if '\n' in element[1]:
                subs = element[1].split('\n')
                if len(subs) > 0:
                    internal_returns = len(subs) - 1
                extents = self.cr.text_extents(subs[-1])
                advance = extents.x_advance

            if len(self.layout) > 0 and element[1][0].isalpha():  # this element is a continuation of a previous word
                previous_element = self.layout[-1]
                previous_word = previous_element[1]
                previous_extents = self.cr.text_extents(previous_word)
                previous_end = previous_element[0].copy()
                previous_end[0] += previous_extents.x_advance

                if '\n' in previous_word:
                    previous_subs = previous_word.split('\n')
                    if len(previous_subs) > 0:
                        previous_internal_returns = len(previous_subs) - 1
                        previous_end[1] += (self.leading * previous_internal_returns)
                    if previous_subs[-1] == '':
                        previous_end[0] = self.paragraph_indent
                    else:
                        sub_extent = self.cr.text_extents(previous_subs[-1])
                        previous_end[0] = self.paragraph_indent + sub_extent.x_advance
                else:
                    if self.wrap_text:
                        if previous_end[0] + advance > self.frame[0] + self.frame[2]:
                            self.layout[-1][0][0] = 0
                            self.layout[-1][0][1] += self.leading
                            element[0] = previous_end.copy()
                            wrapped = True
            elif element[0][0] + advance > self.frame[0] + self.frame[2]:
                if self.wrap_text:
                    element[0] = [0, element[0][1] + self.leading]
                    wrapped = True
            last_fragment_start_x = element[0][0]
            if internal_returns > 0 and self.wrap_text:
                last_fragment_start_x = 0
            self.layout.append(element)
            # n.b. if a wrapp caused a return, then choice list will be one too many lines down after scroll to active line
            # self.set_choice_menu_position()
            # if wrapped:
            #     self.choice_menu_position[1] -= self.leading
            self.cursor_position = [last_fragment_start_x + extents.x_advance,
                                    element[0][1] + internal_returns * self.leading]

    def scroll_position_to_active_line(self):
        delta = (self.active_line * self.leading) - self.cursor_position[1]
        if delta != 0:
            for element in self.layout:  # shift all elements down one line
                element[0][1] += delta
        self.cursor_position[1] = self.active_line * self.leading
        # self.choice_menu_position[1] += delta

    def adjust_active_line(self, old_active_line):
        delta = (self.active_line - old_active_line) * self.leading  # was -1
        for index, element in enumerate(self.layout):  # shift all elements down one line
            element[0][1] += delta

    def internal_returns(self, word):
        subs = word.split('\n')
        return len(subs) - 1

    def move_cursor_to_end(self):
        if len(self.layout) > 0:
            element = self.layout[-1]
            self.move_cursor_to_end_of_element(element)
        else:
            self.cursor_position = [0, 0]

    def move_cursor_to_end_of_element(self, element):
        word = element[1]
        position = element[0].copy()
        internal_returns = 0
        advance = 0
        if len(word) > 0:
            subs = word.split('\n')
            internal_returns = len(subs) - 1
            advance = self.cr.text_extents(subs[-1]).x_advance
            if internal_returns > 0:
                position[0] = self.paragraph_indent
        self.cursor_position = [position[0] + advance, position[1] + internal_returns * self.leading]

    def step_back(self, step_size):
        if len(self.layout) > step_size:
            del self.layout[-step_size:]  # delete section
            self.move_cursor_to_end()

    #    @jit
    def draw_element_at_position(self, element):
        # note: if element includes return characters, we should handle this here
        pos = element[0].copy()
        self.cr.move_to(pos[0], pos[1])
        if len(element) > 2:
            self.cr.set_source_rgb(element[2], element[2], element[2])
        else:
            self.cr.set_source_rgb(1.0, 1.0, 1.0)
        string = element[1]
        subs = string.split('\n')
        for index, sub in enumerate(subs):
            text = make_printable(sub)
            self.cr.show_text(text)
            if index < len(subs) - 1:
                pos[1] += self.leading
                pos[0] = self.paragraph_indent
                self.cr.move_to(pos[0], pos[1])

    #   @jit
    def draw_layout(self):
        self.scroll_position_to_active_line()
        self.erase_rect()
        self.cr.set_font_face(self.face[self.which_font])
        self.cr.set_font_size(self.font_size)

        for index, element in enumerate(self.layout):
            if element[0][1] > self.dashboard_height:
                if len(element[1]) > 0:
                    self.draw_element_at_position(element)
        # if self.show_list:
        #     for element in self.list:
        #         self.draw_element_at_position(element)
        # if self.show_list:
        #     self.draw_arrow()

        target = self.cr.get_target()
        target.flush()

    def erase_rect(self):
        self.cr.rectangle(self.frame[0], self.frame[1], self.frame[2], self.frame[3])
        self.cr.set_source_rgb(0.0, 0.0, 0.0)
        self.cr.set_operator(cairo.OPERATOR_OVER)
        self.cr.fill()
        self.cr.set_source_rgb(1.0, 1.0, 1.0)


