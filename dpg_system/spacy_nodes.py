import spacy
import en_core_web_lg

from spacy import displacy
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from scipy import spatial
import time

indent = 0


def register_spacy_nodes():
    Node.app.register_node('rephrase', RephraseNode.factory)
    Node.app.register_node('lemma', LemmaNode.factory)
    Node.app.register_node('spacy_vector', PhraseVectorNode.factory)
    Node.app.register_node('spacy_similarity', PhraseSimilarityNode.factory)
    Node.app.register_node('spacy_confusion', SpacyConfusionMatrixNode.factory)


class PhraseMatch():
    def __init__(self, token=None, phrase=None, score=0.0, bare=False):
        self.token = token
        self.phrase = phrase
        self.score = score
        self.bare = bare

    def set(self, token, phrase, score, bare):
        self.token = token
        self.phrase = phrase
        self.score = score
        self.bare = bare

    def reset(self):
        self.token = None
        self.phrase = None
        self.score = 0.0
        self.bare = False



def print_chunk_list(a_chunk, top_level=0):
    for c in a_chunk:
        if type(c) == list:
            print_chunk_list(c, top_level - 1)
        else:
            print(c.text, end=' ')
    if top_level > 0:
        print('\n')


class SpacyNode(Node):
    nlp = None

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        if self.__class__.nlp is None:
            self.__class__.nlp = spacy.load('en_core_web_lg')


class SpacyConfusionMatrixNode(SpacyNode):
    @staticmethod
    def factory(name, data, args=None):
        node = SpacyConfusionMatrixNode(name, data, args)
        return node
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input = self.add_input("input", triggers_execution=True)
        self.input2 = self.add_input("input2", triggers_execution=True)
        self.output = self.add_output("output")
        self.confusion_matrix = np.zeros((1, 1))
        self.doc1 = None
        self.doc2 = None
        self.data2 = None
        self.vectors_2 = []

    def execute(self):
        if self.input2.fresh_input:
            self.vectors_2 = []
            self.data2 = self.input2()
            for word in self.data2:
                self.vectors_2.append(self.nlp(word).vector)

        if self.data2 is not None and len(self.data2) > 0:
            self.vectors = []
            data1 = self.input()
            self.confusion_matrix = np.ndarray((len(self.data2), len(data1)))
            for index, word in enumerate(data1):
                vector_ = self.nlp(word).vector
                for index2, word2 in enumerate(self.data2):
                    sim = 1-spatial.distance.cosine(vector_, self.vectors_2[index2])
                    self.confusion_matrix[index2, index] = sim
            self.output.send(self.confusion_matrix)


class PhraseVectorNode(SpacyNode):
    @staticmethod
    def factory(name, data, args=None):
        node = PhraseVectorNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sentence = None
        self.doc = None
        self.input = self.add_input('phrase in', triggers_execution=True)
        self.output = self.add_output('phrase vector out')

    def execute(self):
        if self.input.fresh_input:
            sentence = self.input()
            self.sentence = any_to_string(sentence)
            self.doc = self.nlp(self.sentence)
            vector = self.doc.vector
            self.output.send(vector)


class PhraseSimilarityNode(SpacyNode):
    @staticmethod
    def factory(name, data, args=None):
        node = PhraseSimilarityNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.sentence = None
        self.doc = None
        self.sentence2 = None
        self.doc2 = None
        self.input = self.add_input('phrase in', triggers_execution=True)
        self.input2 = self.add_input('phrase 2 in', triggers_execution=True)
        self.output = self.add_output('phrase similarity out')

    def execute(self):
        if self.input.fresh_input:
            sentence = self.input()
            self.sentence = any_to_string(sentence)
            self.doc = self.nlp(self.sentence)
        if self.input2.fresh_input:
            sentence = self.input2()
            self.sentence2 = any_to_string(sentence)
            self.doc2 = self.nlp(self.sentence2)
        if self.doc is not None and self.doc2 is not None:
            sim = self.doc2.similarity(self.doc)
            self.output.send(sim)


class LemmaNode(SpacyNode):
    @staticmethod
    def factory(name, data, args=None):
        node = LemmaNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.doc = None
        self.input = self.add_input('text in', triggers_execution=True)
        self.output = self.add_output('lemmas out')

    def execute(self):
        if self.input.fresh_input:
            self.sentence = self.input()
            self.sentence = any_to_string(self.sentence)
            self.doc = self.nlp(self.sentence)
            lemma_list = []
            for word in self.doc:
                lemma_list.append(word.lemma_)
            self.output.send(lemma_list)


class RephraseNode(SpacyNode):
    @staticmethod
    def factory(name, data, args=None):
        node = RephraseNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.chunks = []
        self.root_index = -1
        self.sentence = ''
        self.indent = 0
        self.replace_sim_threshold = 0.5
        self.doc = None
        self.new_doc = None
        self.last_input_time = time.time()
        self.clear_input_pause = 40.0
        self.complexity_threshold = 6.0
        self.clip_score_threshold = 640.0
        self.output_as_list = False
        self.input = self.add_input('text in', triggers_execution=True)
        self.clip_score_input = self.add_input('clip score')
        self.replace_sim_threshold_property = self.add_property('replace similarity', widget_type='drag_float', default_value=self.replace_sim_threshold)
        self.clear_input_pause_property = self.add_property('clear input pause', widget_type='drag_float', default_value=self.clear_input_pause)
        self.complexity_threshold_property = self.add_property('complexity replace threshold', widget_type='drag_float', default_value=self.complexity_threshold)
        self.clip_score_threshold_property = self.add_property('clip score threshold', widget_type='drag_float', default_value=self.clip_score_threshold)
        self.output_as_list_property = self.add_property('output as list', widget_type='checkbox', default_value=self.output_as_list)
        self.output = self.add_output('results')
        self.previous_sentence = ''
        self.recursion = 0
        self.focus_noun = None
        self.pending_focus_noun = ''
        self.clip_score = 0

        if self.__class__.nlp is not None:
            temp_sentence = 'an apple'
            temp_doc = self.nlp(temp_sentence)
            self.token_an = temp_doc[0]
            temp_sentence = 'a pear'
            temp_doc = self.nlp(temp_sentence)
            self.token_a = temp_doc[0]
            temp_sentence = 'this pear'
            temp_doc = self.nlp(temp_sentence)
            self.token_this = temp_doc[0]
            temp_sentence = 'that pear'
            temp_doc = self.nlp(temp_sentence)
            self.token_that = temp_doc[0]
            temp_sentence = 'those pears'
            temp_doc = self.nlp(temp_sentence)
            self.token_those = temp_doc[0]
            temp_sentence = 'these pears'
            temp_doc = self.nlp(temp_sentence)
            self.token_these = temp_doc[0]
            temp_doc = self.nlp('The woman who was pregnant')
            self.token_who = temp_doc[2]
            temp_doc = self.nlp('they were wet')
            self.token_were = temp_doc[1]
            temp_doc = self.nlp('he was wet')
            self.token_was = temp_doc[1]
            temp_doc = self.nlp('they are wet')
            self.token_are = temp_doc[1]
            temp_doc = self.nlp('he is wet')
            self.token_is = temp_doc[1]
        self.message_handlers['full_tree'] = self.show_full_tree
        self.message_handlers['bare_tree'] = self.show_bare_tree

    def subtree(self, root_word):
        sub = []
        for index, token in enumerate(self.doc):
            if token.text == root_word:
                for t in token.subtree:
                    sub.append(t)
        return sub

    def token_subtree(self, root_token):
        sub = []
        for t in root_token.subtree:
            sub.append(t)
        return sub

    def trigger_subtree(self, root_word):
        sub = self.subtree(root_word)
        if len(sub) > 0:
            string_list = self.token_list_to_string_list(sub)
            self.output.send(string_list)

    def trigger_bare_tree(self, root_word):
        sub = []
        for index, token in enumerate(self.doc):
            if token.text == root_word:
                if token.pos_ == 'ADP':
                    sub = self.prep_phrase(token)
                elif token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                    sub = self.noun_phrase(token)
                elif token.pos_ in ['VERB', 'AUX']:
                    sub = self.verb_phrase(token)
        if len(sub) > 0:
            string_list = self.token_list_to_string_list(sub)
            self.output.send(string_list)

    def prep_phrase(self, token):
        sub = []
        if token.pos_ == 'ADP':
            for t in token.subtree:
                if t is not token and t.pos_ == 'ADP':
                    break
                sub.append(t)
        return sub

    def noun_phrase(self, token):
        sub = []
        if token.pos_ in ['NOUN', 'PROPN', 'PRON']:
            for t in token.subtree:
                if t is not token and t.pos_ in ['ADP', 'VERB']:
                    break
                sub.append(t)
        return sub

    def verb_phrase(self, token):
        sub = []
        found_verb = False
        if token.pos_ == 'VERB':
            for t in token.subtree:
                if t == token:
                    found_verb = True
                if found_verb and t.pos_ == 'ADP':
                    break
                sub.append(t)
        return sub

    def show_full_tree(self, message, data):
        if self.doc is not None:
            if self.doc.has_annotation("DEP"):
                if len(self.sentence) > 0:
                    self.trigger_subtree(data[0])

    def show_bare_tree(self, message, data):
        if self.doc is not None:
            if self.doc.has_annotation("DEP"):
                if len(self.sentence) > 0:
                    self.trigger_bare_tree(data[0])

    def execute(self):
        if self.clip_score_input.fresh_input:
            self.clip_score = self.clip_score_input()
        if self.input.fresh_input:
            input = self.input()
            # handled, do_output = self.check_for_messages(input)
            # if not handled:
            sentence = any_to_string(input)
            self.parse(sentence, self.clip_score)

    def phrase_list_to_string(self, in_list):
        string = ''
        t_ = type(in_list)
        if t_ == list:
            if len(in_list) == 1:
                t = type(in_list[0])
                if t == list:
                    s = self.phrase_list_to_string(in_list[0])
                    if s[0] == ' ':
                        s = s[1:]
                    if len(string) > 0 and string[-1] not in [' ', '[']:
                        string += ' '
                    if s[0] == '[' and s[-1] == ']':
                        string += s
                    else:
                        string += ('[' + s + ']')
                else:
                    string += in_list[0].text
            else:
                if len(string) > 0 and string[-1] not in[' ', '[']:
                    string += ' '
                string += '['
                for item in in_list:
                    t = type(item)
                    if t == list:
                        s = self.phrase_list_to_string(item)
                        if len(s) > 0:
                            if s[0] == ' ':
                                s = s[1:]
                            if len(string) > 0 and string[-1] not in[' ', '[']:
                                string += ' '
                            if s[0] == '[' and s[-1] == ']':
                                string += s
                            else:
                                string += ('[' + s + ']')
                    else:
                        if len(string) > 0 and string[-1] not in[' ', '[']:
                            string += ' '
                        string += item.text
                string += ']'
        else:
            if len(string) > 0 and string[-1] not in[' ', '[']:
                string += ' '
            string += in_list.text
        return string

    def gather_token_list_from_doc(self):
        sentence_list = []
        for index, token in enumerate(self.doc):
            sentence_list.append(token)
        return sentence_list

    def token_list_to_string(self, sentence_list):
        sentence_string_list = self.token_list_to_string_list(sentence_list)
        sentence = ' '.join(sentence_string_list)
        return sentence

    def token_list_to_string_list(self, sentence_list):
        sentence_string_list = []
        for token in sentence_list:
            sentence_string_list.append(token.text)
        return sentence_string_list

    def try_replace_sconj(self, sentence, new_token, conjunction=None):
#        best = PhraseMatch()
        new_sconj = list(self.new_doc)
        for old_index, token in enumerate(self.doc):
            if token.text == new_token.text:  # same sconj
                if token.pos_ != 'ROOT':
                    if token.dep_ == 'prep':
                        current_sconj = self.token_subtree(token)
                    else:
                        root_token = token.head
                        current_sconj = self.token_subtree(root_token)
                else:
                    current_sconj = self.token_subtree(token)
                sentence_list = self.gather_token_list_from_doc()
                start = sentence_list[:current_sconj[0].i]
                end = sentence_list[current_sconj[-1].i + 1:]
                if conjunction is not None:
                    new_sentence_list = start + current_sconj + [conjunction] + new_sconj + end
                else:
                    new_sentence_list = start + new_sconj + end
                new_sentence = self.token_list_to_string(new_sentence_list)
                return new_sentence
        return self.sentence + ' ' + sentence

    def try_replace_prep_phrase(self, sentence, new_token, conjunction=None):
        best = PhraseMatch()
        new_pp = self.token_subtree(new_token)
        for old_index, token in enumerate(self.doc):
            if token.pos_ == 'ADP':
                current_pp = self.prep_phrase(token)
                if len(current_pp) > 0:
                    n_bare = self.new_doc[new_token.i:new_token.i + 1]
                    c_bare = self.doc[current_pp[0].i:current_pp[0].i + 1]
                    sim = c_bare.similarity(n_bare)
                    # n = self.new_doc[new_pp[0].i:new_pp[-1].i + 1]
                    # c = self.doc[current_pp[0].i:current_pp[-1].i + 1]
                    # sim = c.similarity(n)
                    if sim > best.score:
                        best.set(token, current_pp, sim, False)
                    # print(current_pp, sim)
        if best.score > self.replace_sim_threshold:
            best_pp = best.phrase
            sentence_list = self.gather_token_list_from_doc()
            start = sentence_list[:best_pp[0].i]
            end = sentence_list[best_pp[-1].i + 1:]
            if conjunction is not None:
                new_sentence_list = start + best_pp + [conjunction] + new_pp + end
            else:
                new_sentence_list = start + new_pp + end
            new_sentence = self.token_list_to_string(new_sentence_list)
            return new_sentence
        new_sentence = self.sentence + ', ' + sentence
        return new_sentence

    def try_replace_verb_phrase(self, sentence, new_token, conjunction=None):
        best = PhraseMatch()

        # N.B. they are red and green, with focus_noun replacing 'they'
        #  if verb token is AUX (be) and nsubj token matches a token in the sentence
        # then we want 'that are red and green after subject'
        bare_new_vp = self.new_doc[new_token.i:new_token.i + 1]

        new_subject = None
        old_subject = None
        is_aux = False
        if new_token.pos_ == 'AUX':
            is_aux = True
        if new_token.tag_ in ['VBG', 'VBN']:
            is_aux = True
        if new_token.pos_ == 'VERB' and new_token.lemma_ == 'have':
            is_aux = True

        if is_aux:
            assignment = False
            aux = None
            if new_token.tag_ in ['VBG', 'VBN']:
                for kid in new_token.children:
                    if kid.pos_ == 'AUX':
                        if kid.lemma_ in ['be', 'have']:
                            assignment = True
                            aux = kid
                            break
            elif new_token.lemma_ in ['be', 'have']:
                assignment = True
                aux = new_token
            if assignment:
                for t in self.new_doc:
                    if t.dep_ in ['nsubj', 'nsubjpass']:
                        new_subject = t
                        break
                if new_subject is not None:
                    for t in self.doc:
                        if t.text == new_subject.text:
                            old_subject = t
                            break
                if old_subject is not None:
                    # old subject should now be qualified by phrase with 'that' replacing new_subject
                    sentence_list = self.new_doc[new_subject.i + 1:]
                    if sentence_list[0].lemma_ in ['be', 'have']:
                        test_adj = sentence_list[1]
                        convert_to_adj = False
                        if sentence_list[1].dep_ in ['amod', 'acomp']:
                            convert_to_adj = True
                        if sentence_list[1].pos_ == 'VERB' and sentence_list[1].tag_ in ['VBG', 'VBN']:
                            convert_to_adj = True
                        if len(sentence_list) == 2 and convert_to_adj:
                            modifier = sentence_list[1]
                            old_sentence_list = list(self.doc)
                            # NOTE THAT THERE MAY BE COMPOUND
                            start = old_sentence_list[:old_subject.i]
                            end = old_sentence_list[old_subject.i:]

                            for kid in old_subject.children:
                                if kid.dep_ == 'compound':
                                    start = old_sentence_list[:kid.i]
                                    end = old_sentence_list[kid.i:]
                            new_sentence_list = start + [modifier] + end
                            new_sentence = self.token_list_to_string(new_sentence_list)
                            self.pending_focus_noun = new_subject.text
                            return new_sentence

                    sentence_list = list(sentence_list)

                    if new_subject.pos_ == 'PROPN':
                        sentence_list = [self.token_who] + sentence_list
                    else:
                        sentence_list = [self.token_that] + sentence_list
                    if sentence_list[1].lemma_ == 'be':
                        if self.noun_token_is_plural(new_subject):
                            if sentence_list[1].tag_ == 'VBD':
                                sentence_list[1] = self.token_were
                            else:
                                sentence_list[1] = self.token_are
                        else:
                            if sentence_list[1].tag_ == 'VBD':
                                sentence_list[1] = self.token_was
                            else:
                                sentence_list[1] = self.token_is
                    new_phrase = self.token_list_to_string(sentence_list)
                    self.new_doc = self.nlp(new_phrase)
                    new_phrase_list = list(self.new_doc)
                    old_sentence_list = list(self.doc)
                    start = old_sentence_list[:old_subject.i + 1]
                    end = old_sentence_list[old_subject.i + 1:]
                    new_sentence_list = start + new_phrase_list + end
                    new_sentence = self.token_list_to_string(new_sentence_list)
                    self.pending_focus_noun = new_subject.text
                    return new_sentence

        new_vp = self.token_subtree(new_token)
        new_end = new_vp[new_token.i + 1:]

        for old_index, token in enumerate(self.doc):
            if token.pos_ == new_token.pos_:
                current_vp = self.verb_phrase(token)
                if len(current_vp) > 0:
                    n = self.new_doc[new_vp[0].i:new_vp[-1].i + 1]
                    c = self.doc[current_vp[0].i:current_vp[-1].i + 1]
                    sim = c.similarity(n)
                    if sim > best.score:
                        best.set(token, current_vp, sim, False)
                    # print(current_vp, sim)

                    # if new_vp has no subject, we do not look at the subject of the vp
                    # most likely subject remains the same and object / adverd / prep phrase is changing
                    # so we want match up the new_vp and the current_vp and merge them
                    n = bare_new_vp
                    c_bare = self.doc[token.i:token.i + 1]
                    sim = c_bare.similarity(n)
                    if sim > best.score:
                        best.set(token, current_vp, sim, True)
                    # print(c, sim)

        if best.score > self.replace_sim_threshold:
            proposed_vp_to_replace = list(best.phrase)
            sentence_list = self.gather_token_list_from_doc()
            start = sentence_list[:proposed_vp_to_replace[0].i]
            end = sentence_list[proposed_vp_to_replace[-1].i + 1:]
            if best.bare and conjunction is None:
                new_vp = self.merge_vps(proposed_vp_to_replace, new_vp, new_token)
            if conjunction is not None:
                new_sentence_list = start + current_vp + new_vp + end
            else:
                new_sentence_list = start + new_vp + end
            new_sentence = self.token_list_to_string(new_sentence_list)
            return new_sentence

            #  only a single word
        test_noun_string = 'the ' + sentence

        test_sentence = self.conditional_parse(test_noun_string, strip_det=True)
        if test_sentence != '':
            return test_sentence
        # WHAT IF THE PARSER HAS DECIDED THAT A VERB FORM ADJECTIVE IS THE ROOT
        # i.e. damned animal  'damned' - morph Tense=Past | VerbForm=Fin 'animal' - 'dobj'
        #
        new_sentence = self.sentence + ', ' + sentence
        return new_sentence

    def try_replace_noun_phrase_module(self, incoming_phrase, new_token, conjunction=None, strip_det=False, match_pos_exactly=True):
        # incoming_phrase is the new incoming phrase
        best = PhraseMatch()
        bare_new_np = self.new_doc[new_token.i:new_token.i + 1]
        new_sentence = ''
        new_np = self.token_subtree(new_token)
        new_end = new_np[new_token.i + 1:]
        #### NEW END is bad if there is a conjunction in the new_np

        if conjunction is not None and new_np[0] == conjunction:
            new_np = new_np[1:]

        if strip_det and new_np[0].pos_ == 'DET':
            new_np = new_np[1:]

        new_is_proper_noun = (new_token.pos_ == 'PROPN')
        if new_is_proper_noun and new_token.is_lower:
            new_is_proper_noun = False

        for old_index, token in enumerate(self.doc):
            if token.pos_ == new_token.pos_ or (not new_is_proper_noun and new_token.pos_ == 'PROPN' and token.pos_ == 'NOUN'):
                if token.dep_ != 'compound':
                    current_np = self.noun_phrase(token)
                    if len(current_np) > 0:
                        n = self.new_doc[new_np[0].i:new_np[-1].i + 1]
                        c = self.doc[current_np[0].i:current_np[-1].i + 1]
                        sim = c.similarity(n)
                        if sim > best.score:
                            best.set(token, current_np, sim, False)
                        # print(c, n, sim)

                        n = bare_new_np
                        c_bare = self.doc[token.i:token.i + 1]
                        sim = c_bare.similarity(n)
                        if sim > best.score:
                            best.set(token, c, sim, True)
                        # print(c, n, sim)

                        if new_np[0].pos_ != 'DET':
                            if not self.noun_token_is_plural(new_token):
                                # see if we add 'a ' to the start, do we get a better score?
                                test_string = 'a ' + self.token_list_to_string(new_np)
                                temp_doc = self.nlp(test_string)
                                sim = c.similarity(temp_doc)
                                if sim > best.score:
                                    best.set(token, current_np, sim, False)
                                # print('added_article', current_np, test_string, sim)
                        # item = self.nlp.vocab.__getitem__(new_token.text)
#                        if new_token.morph.get('Number')[0] != best.token.morph.get('Number')[0]:
                        new_test_string = 'the ' + new_token.lemma_
                        old_test_string = 'the ' + best.token.lemma_
                        new_temp_doc = self.nlp(new_test_string)
                        old_temp_doc = self.nlp(old_test_string)
                        sim = old_temp_doc.similarity(new_temp_doc)
                        if sim > best.score:
                            best.set(token, current_np, sim, False)
                        # print('forced_singular', old_test_string, new_test_string, sim)

        if best.score > self.replace_sim_threshold or (new_is_proper_noun and best.token.pos_ == 'PROPN'):
            proposed_np_to_replace = list(best.phrase)
            sentence_list = self.gather_token_list_from_doc()
            start = sentence_list[:proposed_np_to_replace[0].i]
            end = sentence_list[proposed_np_to_replace[-1].i + 1:]
            if conjunction is None:
                new_np = self.merge_nps(proposed_np_to_replace, new_np, new_token)
                new_np = self.fix_article(new_np, new_token, best.token)

            # what if not bare but missing determiner?
            if len(new_end) > 0:
                # n.b. new end might include competing prep phrase
                end = new_end + end
            if conjunction is not None:
                new_sentence_list = start + proposed_np_to_replace + [conjunction] + new_np + end
            else:
                new_sentence_list = start + new_np + end
            new_sentence = self.token_list_to_string(new_sentence_list)
        return new_sentence

    def noun_token_is_plural(self, noun_token):
        return noun_token.morph.get('Number')[0] == 'Plur'

    def try_replace_noun_phrase(self, incoming_phrase, new_token, conjunction=None, strip_det=False):
        # incoming_phrase is the new incoming phrase
        #### noun phrases including PP? how to handle?
        new_sentence = self.try_replace_noun_phrase_module(incoming_phrase, new_token, conjunction, strip_det, match_pos_exactly=True)
        if new_sentence == '':
            new_sentence = self.try_replace_noun_phrase_module(incoming_phrase, new_token, conjunction, strip_det,
                                                               match_pos_exactly=False)
        return new_sentence

    # adjective conjunctions???
    def try_replace_adjective(self, sentence, new_token, conjunction=None):
        best = PhraseMatch()
        for old_index, token in enumerate(self.doc):
            if token.pos_ == 'ADJ':
                n = self.new_doc[new_token.i:new_token.i + 1]
                c = self.doc[token.i:token.i + 1]
                sim = c.similarity(n)
                if sim > best.score:
                    best.set(token, None, sim, False)
                # print(token.text, sim)
        if best.score > self.replace_sim_threshold:
            best_token = best.token
            sentence_list = self.gather_token_list_from_doc()
            if conjunction is not None:
                for index, t in enumerate(sentence_list):
                    if t.text == best_token.text:
                        sentence_list = sentence_list[:index + 1] + [conjunction] + [new_token] + sentence_list[index + 1:]
                        break
            else:
                sentence_list[best_token.i] = new_token
            new_sentence = self.token_list_to_string(sentence_list)
            return new_sentence
        if conjunction is not None:
            new_sentence = self.sentence + ' ' + sentence
        else:
            new_sentence = self.sentence + ', ' + sentence
        return new_sentence

    def phrase_complexity(self, doc):
        prep_count = 0
        noun_count = 0
        verb_count = 0
        word_count = 0
        for new_index, new_token in enumerate(doc):
            if new_token.pos_ == 'VERB':
                verb_count += 1
            if new_token.pos_ == 'ADP':
                prep_count += 1
            if new_token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                noun_count += 1
            word_count += 1
        complexity = prep_count + noun_count + verb_count + word_count / 4
        return complexity

    def remove_compound(self, root_token):
        for t in self.doc:
            if t.dep_ == 'compound':
                if t.lemma_ == root_token.lemma_:
                    pre_compound = self.doc[:t.i - 1]
                    post_compound = self.doc[t.i + 1:]
                    new_sentence_list = list(pre_compound) + list(post_compound)
                    new_sentence = self.token_list_to_string(new_sentence_list)
                    return new_sentence
        return ''


    def remove_adjective(self, root_token):
        for t in self.doc:
            if t.pos_ == 'ADJ' or t.dep_ == 'amod':
                if t.lemma_ == root_token.lemma_:
                    adjective_sub_tree = self.token_subtree(t)
                    if adjective_sub_tree[0].pos_ == 'DET':
                        adjective_sub_tree = adjective_sub_tree[1:]
                    if len(t.conjuncts) > 0:
                        if t.dep_ == 'conj':
                            if t.i > 0 and self.doc[t.i - 1].pos_ == 'CCONJ':
                                pre_adjective = self.doc[:t.i - 1]
                                post_adjective = self.doc[t.i + 1:]
                                new_sentence_list = list(pre_adjective) + list(post_adjective)
                                new_sentence = self.token_list_to_string(new_sentence_list)
                                return new_sentence
                        elif t.i < len(self.doc) - 2 and self.doc[t.i + 1].pos_ == 'CCONJ':
                            pre_adjective = self.doc[:t.i]
                            post_adjective = self.doc[t.i + 2:]
                            new_sentence_list = list(pre_adjective) + list(post_adjective)
                            new_sentence = self.token_list_to_string(new_sentence_list)
                            return new_sentence
                    # adjective.dep_ == 'amod'
                        # check for conjunction?
                    if t.dep_ == 'acomp' and t.head.dep_ == 'relcl':
                        rel_clause = self.token_subtree(t.head)
                        pre_adjective = self.doc[:rel_clause[0].i]
                        post_adjective = self.doc[rel_clause[-1].i + 1:]
                        new_sentence_list = list(pre_adjective) + list(post_adjective)
                        new_sentence = self.token_list_to_string(new_sentence_list)
                        return new_sentence
                    # adjective.dep_ == 'acomp' -> adjective.head.dep_ == 'relcl'
                        # remove whole relative clause
                    # adjective.conjunctions -> remove adjective and conjunction
                    if len(self.new_doc) == 2:
                        pre_adjective = self.doc[:adjective_sub_tree[0].i]
                        post_adjective = self.doc[adjective_sub_tree[-1].i + 1:]
                        new_sentence_list = list(pre_adjective) + list(post_adjective)
                        new_sentence = self.token_list_to_string(new_sentence_list)
                        return new_sentence
        return ''

    def remove_adverb(self, root_token):
        for t in self.doc:
            if t.pos_ == 'ADV':
                if t.lemma_ == root_token.lemma_:
                    adverb_sub_tree = self.token_subtree(t)
                    if len(self.new_doc) == 2:
                        pre_adjective = self.doc[:adverb_sub_tree[0].i]
                        post_adjective = self.doc[adverb_sub_tree[-1].i + 1:]
                        new_sentence_list = list(pre_adjective) + list(post_adjective)
                        new_sentence = self.token_list_to_string(new_sentence_list)
                        return new_sentence
        return ''

    def conditional_parse(self, sentence, strip_det=False, clip_score=0.0):
        if self.doc is None:
            return ''
        self.new_doc = self.nlp(sentence)
        if self.new_doc[0].lower_ == 'not':
            root_token = self.new_doc[1:].root
            if root_token.pos_ == 'ADJ':
                new_sentence = self.remove_adjective(root_token)
                if len(new_sentence) > 0:
                    return new_sentence
            if root_token.pos_ == 'ADV':
                new_sentence = self.remove_adverb(root_token)
                if len(new_sentence) > 0:
                    return new_sentence
            if root_token.pos_ in ['NOUN', 'PROPN']:
                new_sentence = self.remove_compound(root_token)
                if len(new_sentence) > 0:
                    return new_sentence
            # if 'not' is follow by a noun phrase
                # capture that noun phrase and remove it

        # IF CLIP SCORE IS TOO HIGH, DO NOT DO THIS...
        if clip_score > self.clip_score_threshold:
            return ''
        if self.new_doc[0].lower_ in ['it', 'they', 'he', 'she'] and self.focus_noun is not None:
            sentence_list = list(self.new_doc)
            sentence_list[0] = self.focus_noun
            sentence = self.token_list_to_string(sentence_list)
            self.new_doc = self.nlp(sentence)
        self.replace_sim_threshold = self.replace_sim_threshold_property()

        root = None
        for new_index, new_token in enumerate(self.new_doc):
            if new_token.dep_ == 'ROOT':
                root = new_token

        complexity = self.phrase_complexity(self.new_doc)

        if self.new_doc[0].pos_ in ['CCONJ', 'SCONJ'] and complexity > self.complexity_threshold_property():
            sentence = self.sentence + ' ' + sentence
            return sentence

        if root.pos_ != 'ADP' and complexity > self.complexity_threshold_property():
            return ''

        self.recursion += 1
        if self.recursion > 1:
            return ''

        conjunction = None
        if self.new_doc[0].pos_ == 'CCONJ':
            conjunction = self.new_doc[0]
            # might look for best match on root??
        elif self.new_doc[0].pos_ == 'SCONJ':
            test_sentence = self.try_replace_sconj(sentence, self.new_doc[0], conjunction)
            if len(test_sentence) > 0:
                return test_sentence

        if root.pos_ == 'SCONJ':
            sentence = self.try_replace_sconj(sentence, root, conjunction)
            sentence = self.sentence + ' ' + sentence
            return sentence
        if root.pos_ in ['ADP']:
            return self.try_replace_prep_phrase(sentence, root, conjunction)
        elif root.pos_ in ['VERB', 'AUX']:
            return self.try_replace_verb_phrase(sentence, root, conjunction)
        elif root.pos_ in ['NOUN', 'PROPN', 'PRON']:
            return self.try_replace_noun_phrase(sentence, root, conjunction, strip_det)
        elif root.pos_ == 'ADJ':
            return self.try_replace_adjective(sentence, root, conjunction)

        return ''

    def parse(self, sentence, clip_score):
        self.chunks = []
        self.root_index = -1
        self.indent = 0
        now = time.time()
        if sentence == 'no' or sentence == 'wrong' or sentence == 'go back':
            sentence = self.previous_sentence

        self.previous_sentence = self.sentence
        self.recursion = 0
        was_rewritten = False

        self.replace_sim_threshold = self.replace_sim_threshold_property()
        self.clear_input_pause = self.clear_input_pause_property()
        if now - self.last_input_time < self.clear_input_pause:
            rewritten_sentence = self.conditional_parse(sentence, clip_score=clip_score)
            if len(rewritten_sentence) > 0:
                sentence = rewritten_sentence
                was_rewritten = True

        if clip_score < self.clip_score_threshold:
            self.last_input_time = now

            self.sentence = sentence
            self.doc = self.nlp(self.sentence)
            if len(self.pending_focus_noun) > 0:
                for t in self.doc:
                    if t.text == self.pending_focus_noun:
                        self.focus_noun = t
                        break
                self.pending_focus_noun = ''

            if self.focus_noun is None or not was_rewritten:
                self.focus_noun = self.choose_focus_noun()

            if self.output_as_list_property():
                token_list = self.gather_token_list_from_doc()
                sentence_list = self.token_list_to_string_list(token_list)
                self.output.send(sentence_list)
            else:
                sentence_string = self.doc.text_with_ws
                self.output.send(sentence_string)

    def choose_focus_noun(self):
        subject = None
        object = None
        prep_object = None

        for t in self.doc:
            if t.dep_ in ['nsubj', 'nsubjpass']:
                subject = t
                break
            elif t.dep_ == 'dobj':
                if object is None:
                    object = t
            elif t.dep_ == 'pobj':
                if prep_object is None:
                    prep_object = t

        if subject is not None:
            return subject
        elif object is not None:
            return object
        return prep_object

    def choose_focus_noun(self):
        subject = None
        object = None
        prep_object = None

        for t in self.doc:
            if t.dep_ in ['nsubj', 'nsubjpass']:
                subject = t
                break
            elif t.dep_ == 'dobj':
                if object is None:
                    object = t
            elif t.dep_ == 'pobj':
                if prep_object is None:
                    prep_object = t

        if subject is not None:
            return subject
        elif object is not None:
            return object
        return prep_object

    def fix_article(self, token_list, new_token, old_token):
        new_is_plural = self.noun_token_is_plural(new_token)
        if new_is_plural and token_list[0].pos_ == 'DET':
            if token_list[0].text in ['a', 'an']:
                token_list = token_list[1:]
            elif token_list[0].lemma_ == 'this':
                token_list[0] = self.token_these
            elif token_list[0].lemma_ == 'that':
                token_list[0] = self.token_those
        elif not new_is_plural:
            if token_list[0].pos_ == 'DET':
                if token_list[0].lemma_ == 'these':
                    token_list[0] = self.token_this
                elif token_list[0].lemma_ == 'those':
                    token_list[0] = self.token_that
            else:
                if self.noun_token_is_plural(old_token):
                    if new_token.pos_ != 'PROPN':
                        token_list.prepend(self.token_a)

        if len(token_list) > 1:
            if token_list[0].text == 'a':
                if token_list[1].text[0] in ['a', 'e', 'i', 'o', 'u']:
                    token_list[0] = self.token_an
            elif token_list[0].text == 'an':
                if token_list[1].text[0] not in ['a', 'e', 'i', 'o', 'u']:
                    token_list[0] = self.token_a
        return token_list

    def merge_nps(self, old_np, new_np, new_token):
        rephrase = []
        new_det = []
        new_adjectives = []
        new_compounds = []
        new_conjunction = []
        if new_np[0].pos_ == 'CCONJ':
            new_conjunction = new_np[0]
        for t in new_np:
            if t.pos_ == 'DET':
                new_det.append(t)
            elif t.dep_ in ['amod', 'advmod', 'nummod', 'poss']:
                new_adjectives.append(t)
            elif t.pos_ == 'ADJ':
                new_adjectives.append(t)
            elif t.dep_ == 'compound':
                new_compounds.append(t)
            elif t.pos_ == 'CCONJ':
                new_adjectives.append(t)
        old_det = []
        old_adjectives = []
        old_compounds = []
        for t in old_np:
            if t.pos_ == 'DET':
                old_det.append(t)
            elif t.dep_ in ['amod', 'advmod', 'nummod', 'poss']:
                old_adjectives.append(t)
            elif t.dep_ == 'compound':
                old_compounds.append(t)

        if len(new_conjunction) == 0:
            if len(new_det) == 0:
                if len(old_det) > 0:
                    rephrase += old_det
            else:
                rephrase += new_det
            if len(new_adjectives) > 0:
                rephrase += new_adjectives
            else:
                rephrase += old_adjectives

            if len(new_compounds) > 0:
                rephrase += new_compounds

            new_np = rephrase.copy()
            new_np.append(new_token)
        return new_np.copy()

    def get_verb_subject(self, vp):
        for t in vp:
            if t.dep_ in ['nsubj', 'nsubjpass']:
                subject_tree = self.token_subtree(t)
                return subject_tree
        return []

    def get_verb_direct_object(self, vp):
        for t in vp:
            if t.dep_ == 'dobj':
                direct_object_tree = self.token_subtree(t)
                return direct_object_tree
        return []

    def merge_vps(self, old_vp, new_vp, new_token):
        rephrase = []
        new_adverb = []
        new_pp = []
        new_subject = self.get_verb_subject(new_vp)
        new_object = self.get_verb_direct_object(new_vp)

        for t in new_vp:
            if t.pos_ == 'ADV':
                new_adverb.append(t)

        old_adverb = []
        old_pp = []
        old_subject = self.get_verb_subject(old_vp)
        old_object = self.get_verb_direct_object(old_vp)

        for t in old_vp:
            if t.pos_ == 'ADV':
                old_adverb.append(t)

        rephrase = []

        if len(new_subject) > 0:
            rephrase += new_subject
        elif len(old_subject):
            rephrase += old_subject

        if len(new_adverb) > 0:
            rephrase += new_adverb
        elif len(old_adverb) > 0:
            rephrase += old_adverb

        rephrase.append(new_token)

        if len(new_object) > 0:
            rephrase += new_object
        elif len(old_object):
            rephrase += old_object

        return rephrase.copy()
