import re
import time
import json
import os
import spacy
from dpg_system.node import Node
from dpg_system.conversion_utils import *
def register_context_tracker_nodes():
    Node.app.register_node('context_tracker', ContextTrackerNode.factory)
# ---------------------------------------------------------------------------
# Vocabulary constants  (JSON-loadable / overridable at runtime)
# ---------------------------------------------------------------------------
TIME_OF_DAY_WORDS = {
    'dawn', 'dusk', 'twilight', 'sunrise', 'sunset', 'noon', 'midnight',
    'nighttime', 'nightfall', 'daybreak', 'morning', 'afternoon', 'evening',
    'night', 'daytime', 'midday',
}
# Phrases that signal time-of-day context (matched with surrounding preposition)
TIME_OF_DAY_PHRASES = [
    r'\b(?:at\s+)?(?:dawn|dusk|twilight|sunrise|sunset|noon|midnight|nightfall|daybreak|midday)\b',
    r'\b(?:in the\s+)(?:morning|afternoon|evening|night|dead of night|golden hour|blue hour)\b',
    r'\b(?:at\s+)(?:nighttime|daytime|night|night-time)\b',
]
ERA_WORDS = {
    'medieval', 'renaissance', 'baroque', 'victorian', 'edwardian',
    'modern', 'futuristic', 'ancient', 'prehistoric', 'classical',
    'neolithic', 'bronze age', 'iron age', 'industrial',
    'contemporary', 'postmodern', 'retro', 'vintage',
}
ERA_PATTERNS = [
    r'\b(?:in the\s+)?(?:medieval|renaissance|baroque|victorian|edwardian|'
    r'modern|futuristic|ancient|prehistoric|classical|industrial|'
    r'contemporary|postmodern)\s*(?:era|period|times|age)?\b',
    r'\b(?:in the\s+)?\d{4}s?\b',                          # "in the 1920s", "1847"
    r'\b(?:(?:\d{1,2})(?:th|st|nd|rd)\s+century)\b',        # "19th century"
    r'\b(?:during the\s+)(?:war|revolution|plague|renaissance|enlightenment)\b',
]
PLACE_PATTERNS = [
    r'\b(?:in|inside|within|into)\s+(?:a|an|the)\s+'
    r'(?:forest|desert|jungle|ocean|sea|mountain|mountains|cave|castle|cathedral|'
    r'church|temple|mosque|synagogue|shrine|garden|field|meadow|swamp|marsh|city|'
    r'village|town|alley|alleyway|street|market|marketplace|library|dungeon|palace|'
    r'tower|ruin|ruins|wasteland|tundra|steppe|savanna|prairie|valley|canyon|gorge|'
    r'bedroom|kitchen|basement|attic|hallway|corridor|throne room|courtyard|'
    r'ballroom|tavern|pub|bar|restaurant|café|hospital|prison|arena|colosseum|'
    r'amphitheatre|amphitheater|graveyard|cemetery|crypt|cellar|warehouse|factory|'
    r'laboratory|lab|studio|workshop|greenhouse|conservatory|observatory|'
    r'space station|spaceship|submarine|underwater|underground)\b',
    r'\b(?:on|at|by|near|along|beside|across)\s+(?:a|an|the)\s+'
    r'(?:cliff|shore|shoreline|riverbank|hilltop|rooftop|bridge|dock|pier|beach|'
    r'lake|pond|river|stream|waterfall|fountain|plaza|square|boulevard|highway|'
    r'railroad|train station|airport|harbor|harbour|port|wharf)\b',
    r'\b(?:outside|outside of)\s+(?:a|an|the)\s+\w+\b',
]
# Standalone place nouns — used for dependency-based extraction when
# no pattern or NER match is found. If one of these nouns appears as
# the object of a preposition, we capture the full prep phrase.
PLACE_NOUNS = {
    # Natural environments
    'forest', 'desert', 'jungle', 'ocean', 'sea', 'mountain', 'mountains',
    'cave', 'valley', 'canyon', 'gorge', 'field', 'meadow', 'swamp', 'marsh',
    'prairie', 'tundra', 'steppe', 'savanna', 'island', 'glacier', 'volcano',
    # Built environments
    'castle', 'cathedral', 'church', 'temple', 'mosque', 'shrine', 'palace',
    'tower', 'dungeon', 'fortress', 'citadel', 'ruin', 'ruins', 'courtyard',
    'city', 'village', 'town', 'alley', 'alleyway', 'street', 'market',
    'marketplace', 'plaza', 'square', 'boulevard', 'avenue',
    'tavern', 'pub', 'bar', 'restaurant', 'café', 'cafe', 'inn',
    'library', 'museum', 'gallery', 'theater', 'theatre', 'arena',
    'colosseum', 'amphitheatre', 'amphitheater', 'stadium',
    'hospital', 'prison', 'asylum', 'monastery', 'convent', 'abbey',
    'warehouse', 'factory', 'laboratory', 'lab', 'studio', 'workshop',
    'greenhouse', 'conservatory', 'observatory',
    # Interior spaces
    'bedroom', 'kitchen', 'basement', 'attic', 'hallway', 'corridor',
    'ballroom', 'throne room', 'cellar', 'crypt', 'chamber', 'parlor',
    'parlour', 'salon', 'study', 'nursery', 'chapel',
    # Landscape features
    'cliff', 'shore', 'shoreline', 'riverbank', 'hilltop', 'rooftop',
    'bridge', 'dock', 'pier', 'beach', 'lake', 'pond', 'river', 'stream',
    'waterfall', 'fountain', 'harbor', 'harbour', 'port', 'wharf',
    # Other
    'graveyard', 'cemetery', 'garden', 'orchard', 'vineyard', 'farm',
    'space station', 'spaceship', 'submarine',
}
STYLE_VOCAB = {
    # Art movements
    'impressionist', 'expressionist', 'cubist', 'surreal', 'surrealist',
    'abstract', 'minimalist', 'baroque', 'rococo', 'art nouveau', 'art deco',
    'gothic', 'romantic', 'neoclassical', 'pop art', 'photorealist',
    'photorealistic', 'hyperrealist', 'hyperrealistic', 'pointillist',
    'fauvist', 'dadaist', 'constructivist', 'suprematist', 'futurist',
    # Moods / aesthetics
    'noir', 'cinematic', 'dreamlike', 'ethereal', 'gritty', 'pastoral',
    'dystopian', 'utopian', 'psychedelic', 'whimsical', 'grotesque',
    'melancholic', 'serene', 'chaotic', 'monochrome', 'vibrant',
    'haunting', 'eerie', 'ominous', 'majestic', 'sublime', 'idyllic',
    'desolate', 'lush', 'sparse', 'ornate', 'austere', 'decadent',
    # Media styles
    'watercolor', 'watercolour', 'oil painting', 'charcoal', 'pencil sketch',
    'woodcut', 'lithograph', 'photograph', 'photographic', 'daguerreotype',
    'fresco', 'mosaic', 'stained glass', 'pixel art', 'anime', 'manga',
    'engraving', 'etching', 'tapestry', 'silkscreen', 'screenprint',
    'digital art', 'digital painting', 'concept art',
    # Lighting styles
    'chiaroscuro', 'backlit', 'candlelit', 'neon', 'bioluminescent',
    'high contrast', 'low key', 'high key', 'dramatic lighting',
    'volumetric lighting', 'rim lighting', 'soft lighting',
    'golden light', 'cold light', 'warm light',
}
WEATHER_VOCAB = {
    'rainy', 'foggy', 'misty', 'snowy', 'stormy', 'overcast', 'sunny',
    'cloudy', 'hazy', 'windy', 'drizzling', 'thundering', 'blazing',
    'rain-soaked', 'sun-drenched', 'frost-covered', 'dew-covered',
    'humid', 'arid', 'freezing', 'sweltering', 'muggy', 'breezy',
    'torrential', 'blizzard', 'hurricane', 'typhoon',
}
WEATHER_PATTERNS = [
    r'\b(?:in|during)\s+(?:a|an|the)\s+(?:rainstorm|snowstorm|thunderstorm|'
    r'blizzard|hurricane|typhoon|downpour|drizzle|fog|mist|haze)\b',
    r'\b(?:under|beneath)\s+(?:a|an|the)\s+(?:blazing|scorching|burning|'
    r'bright|cloudy|overcast|grey|gray|dark|starry|moonlit|starlit)\s+sky\b',
]
# Pre-compiled patterns for performance
_TIME_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in TIME_OF_DAY_PHRASES]
_ERA_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in ERA_PATTERNS]
_PLACE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in PLACE_PATTERNS]
_WEATHER_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in WEATHER_PATTERNS]
# ---------------------------------------------------------------------------
# Utility: walk up dependency tree to get governing prepositional phrase
# ---------------------------------------------------------------------------
def get_prep_phrase_text(token):
    """Given an entity root token, walk up to find the governing preposition
    and return the full prepositional phrase text."""
    head = token.head
    # If the head is a preposition, include it
    if head.pos_ == 'ADP' and head.i < token.i:
        # Gather from prep to end of entity span
        start = head.i
        end = token.i
        # Extend to include any right children of the token (adjectives, etc.)
        for child in token.children:
            if child.i > end:
                end = child.i
        return token.doc[start:end + 1].text
    return token.text
def get_full_prep_phrase_text(prep_token):
    """Given a preposition token, return the full prepositional phrase
    including all modifiers (adjectives, compounds, determiners)."""
    subtree = list(prep_token.subtree)
    if subtree:
        return ' '.join(t.text for t in subtree)
    return prep_token.text
def reconstruct_hyphenated(doc):
    """Rebuild hyphenated compounds from spaCy tokens.
    e.g. ['rain', '-', 'soaked'] -> {'rain-soaked'}
    Returns a set of reconstructed hyphenated words."""
    compounds = set()
    tokens = list(doc)
    i = 0
    while i < len(tokens) - 2:
        if tokens[i + 1].text == '-':
            compound = tokens[i].text.lower() + '-' + tokens[i + 2].text.lower()
            compounds.add(compound)
            i += 3
        else:
            i += 1
    return compounds
# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
class ContextTrackerNode(Node):
    nlp = None
    @staticmethod
    def factory(name, data, args=None):
        node = ContextTrackerNode(name, data, args)
        return node
    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # Load spaCy model for NER + dependency parse.
        # Try to reuse SpacyNode's model if already loaded (avoids 560MB duplication).
        if self.__class__.nlp is None:
            try:
                from dpg_system.spacy_nodes import SpacyNode
                if SpacyNode.nlp is not None:
                    self.__class__.nlp = SpacyNode.nlp
                    print('ContextTrackerNode: reusing SpacyNode model')
            except (ImportError, AttributeError):
                pass
            if self.__class__.nlp is None:
                try:
                    self.__class__.nlp = spacy.load('en_core_web_sm')
                    print('ContextTrackerNode: loaded en_core_web_sm')
                except OSError:
                    try:
                        self.__class__.nlp = spacy.load('en_core_web_lg')
                        print('ContextTrackerNode: loaded en_core_web_lg')
                    except OSError:
                        print('ContextTrackerNode: No spaCy model found! '
                              'Install with: python -m spacy download en_core_web_sm')
        # --- Context slot state ---
        self.slot_names = ['time_of_day', 'era', 'place', 'weather', 'style', 'artist']
        self.slots = {name: '' for name in self.slot_names}
        self.slot_ages = {name: 0 for name in self.slot_names}  # chunks since last update
        self.chunk_count = 0
        # --- Loadable vocabularies ---
        self.time_vocab = set(TIME_OF_DAY_WORDS)
        self.era_vocab = set(ERA_WORDS)
        self.style_vocab = set(STYLE_VOCAB)
        self.weather_vocab = set(WEATHER_VOCAB)
        # --- Inputs ---
        self.text_input = self.add_input('text in', triggers_execution=True)
        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear_all)
        self.set_context_input = self.add_input('set context', triggers_execution=True)
        # Vocabulary loading inputs
        self.load_time_vocab = self.add_input('time vocab', callback=self.receive_time_vocab)
        self.load_place_vocab = self.add_input('place vocab', callback=self.receive_place_vocab)
        self.load_style_vocab = self.add_input('style vocab', callback=self.receive_style_vocab)
        self.load_weather_vocab = self.add_input('weather vocab', callback=self.receive_weather_vocab)
        # Artist input — connect from FuzzyMatchNode's 'replacement out'
        self.artist_input = self.add_input('artist in', triggers_execution=True)
        # --- Properties (per-slot weights) ---
        self.time_weight = self.add_property('time weight', widget_type='drag_float', default_value=0.5)
        self.era_weight = self.add_property('era weight', widget_type='drag_float', default_value=0.5)
        self.place_weight = self.add_property('place weight', widget_type='drag_float', default_value=0.5)
        self.weather_weight = self.add_property('weather weight', widget_type='drag_float', default_value=0.3)
        self.style_weight = self.add_property('style weight', widget_type='drag_float', default_value=0.5)
        self.artist_weight = self.add_property('artist weight', widget_type='drag_float', default_value=0.5)
        self.strength = self.add_property('strength', widget_type='drag_float', default_value=1.0)
        self.decay_chunks = self.add_option('decay chunks', widget_type='drag_int', default_value=0)
        # Display current state
        self.status_label = self.add_label('')
        # --- Outputs ---
        self.context_output = self.add_output('context out')
        self.dict_output = self.add_output('context dict')
        self.detected_output = self.add_output('detected')
        # Weight lookup by slot name
        self.weight_properties = {
            'time_of_day': self.time_weight,
            'era': self.era_weight,
            'place': self.place_weight,
            'weather': self.weather_weight,
            'style': self.style_weight,
            'artist': self.artist_weight,
        }
    # --- Vocabulary receivers ---
    def receive_time_vocab(self):
        data = self.load_time_vocab()
        self._merge_vocab(data, self.time_vocab)
    def receive_place_vocab(self):
        # Place vocab would extend the pattern list, not a simple set
        # For now, accept a list of place words to add to a secondary check
        pass
    def receive_style_vocab(self):
        data = self.load_style_vocab()
        self._merge_vocab(data, self.style_vocab)
    def receive_weather_vocab(self):
        data = self.load_weather_vocab()
        self._merge_vocab(data, self.weather_vocab)
    def _merge_vocab(self, data, target_set):
        if type(data) == list:
            for item in data:
                if type(item) == str:
                    target_set.add(item.lower())
        elif type(data) == str:
            # Try loading as JSON file path
            if os.path.exists(data):
                with open(data, 'r') as f:
                    vocab_list = json.load(f)
                    if type(vocab_list) == list:
                        for item in vocab_list:
                            target_set.add(str(item).lower())
    # --- Clear ---
    def clear_all(self, value=None):
        for name in self.slot_names:
            self.slots[name] = ''
            self.slot_ages[name] = 0
        self.chunk_count = 0
        self.update_status()
        self.emit_output()
    # --- Main execution ---
    def execute(self):
        if self.active_input == self.set_context_input:
            data = self.set_context_input()
            if type(data) == list and len(data) >= 2:
                slot_name = str(data[0])
                value = str(data[1])
                if slot_name in self.slots:
                    self.slots[slot_name] = value
                    self.slot_ages[slot_name] = 0
                    self.detected_output.send([slot_name, value])
                    self.update_status()
                    self.emit_output()
            return
        if self.active_input == self.artist_input:
            data = any_to_string(self.artist_input())
            if data.strip() != '':
                self.slots['artist'] = data.strip()
                self.slot_ages['artist'] = 0
                self.detected_output.send(['artist', self.slots['artist']])
                self.update_status()
                self.emit_output()
            return
        if self.text_input.fresh_input:
            text = any_to_string(self.text_input())
            if text.strip() == '':
                return
            self.chunk_count += 1
            # Age all slots
            for name in self.slot_names:
                if self.slots[name] != '':
                    self.slot_ages[name] += 1
            # Optional decay
            decay = self.decay_chunks()
            if decay > 0:
                for name in self.slot_names:
                    if self.slot_ages[name] > decay:
                        self.slots[name] = ''
                        self.slot_ages[name] = 0
            # --- Run extraction ---
            detections = self.extract_context(text)
            # Apply detections
            changed = False
            for slot_name, value in detections.items():
                if value and value != self.slots[slot_name]:
                    self.slots[slot_name] = value
                    self.slot_ages[slot_name] = 0
                    self.detected_output.send([slot_name, value])
                    changed = True
            if changed:
                self.update_status()
            self.emit_output()
    # --- Extraction engine ---
    def extract_context(self, text):
        """Extract context slots from a text chunk. Returns dict of detected values."""
        detections = {}
        if self.__class__.nlp is None:
            return detections
        doc = self.__class__.nlp(text)
        # Reconstruct hyphenated compounds for weather/style matching
        hyphenated = reconstruct_hyphenated(doc)
        # --- 1. spaCy NER ---
        for ent in doc.ents:
            if ent.label_ in ('TIME',):
                phrase = get_prep_phrase_text(ent.root)
                detections['time_of_day'] = phrase
            elif ent.label_ == 'DATE':
                # Decide if this is era or time-of-day
                ent_lower = ent.text.lower()
                if any(w in ent_lower for w in self.era_vocab):
                    detections['era'] = ent.text
                elif re.match(r'\d{4}s?$', ent_lower) or 'century' in ent_lower:
                    detections['era'] = ent.text
                elif any(w in ent_lower for w in self.time_vocab):
                    detections['time_of_day'] = ent.text
            elif ent.label_ in ('GPE', 'LOC', 'FAC'):
                # Get the full prepositional phrase if available
                phrase = get_prep_phrase_text(ent.root)
                detections['place'] = phrase
        # --- 2. Pattern matching ---
        text_lower = text.lower()
        # Time of day patterns
        if 'time_of_day' not in detections:
            for pattern in _TIME_PATTERNS_COMPILED:
                match = pattern.search(text)
                if match:
                    detections['time_of_day'] = match.group(0).strip()
                    break
        # Era patterns
        if 'era' not in detections:
            for pattern in _ERA_PATTERNS_COMPILED:
                match = pattern.search(text)
                if match:
                    detections['era'] = match.group(0).strip()
                    break
        # Place patterns
        if 'place' not in detections:
            for pattern in _PLACE_PATTERNS_COMPILED:
                match = pattern.search(text)
                if match:
                    detections['place'] = match.group(0).strip()
                    break
        # Weather patterns
        for pattern in _WEATHER_PATTERNS_COMPILED:
            match = pattern.search(text)
            if match:
                detections['weather'] = match.group(0).strip()
                break
        # --- 3. Keyword matching ---
        # Weather keywords — check individual tokens AND hyphenated compounds
        if 'weather' not in detections:
            for token in doc:
                if token.text.lower() in self.weather_vocab:
                    detections['weather'] = token.text.lower()
                    break
            # Also check reconstructed hyphenated compounds
            if 'weather' not in detections:
                for compound in hyphenated:
                    if compound in self.weather_vocab:
                        detections['weather'] = compound
                        break
        # Style keywords (check single words, bigrams, and hyphenated)
        if 'style' not in detections:
            # Check single tokens
            for token in doc:
                if token.text.lower() in self.style_vocab:
                    detections['style'] = token.text.lower()
                    break
            # Check bigrams for multi-word styles ("art nouveau", "pop art", etc.)
            if 'style' not in detections:
                words = text_lower.split()
                for i in range(len(words) - 1):
                    bigram = words[i] + ' ' + words[i + 1]
                    if bigram in self.style_vocab:
                        detections['style'] = bigram
                        break
            # Check hyphenated compounds
            if 'style' not in detections:
                for compound in hyphenated:
                    if compound in self.style_vocab:
                        detections['style'] = compound
                        break
        # --- 4. Era from keywords (if not caught by NER or patterns) ---
        if 'era' not in detections:
            for token in doc:
                if token.text.lower() in self.era_vocab:
                    # Try to get the surrounding phrase
                    phrase = get_prep_phrase_text(token)
                    detections['era'] = phrase
                    break
        # --- 5. Dependency-based place extraction ---
        # If no place was found via NER or patterns, look for place nouns
        # that are objects of prepositions and capture the full prep phrase.
        if 'place' not in detections:
            for token in doc:
                token_lower = token.text.lower()
                if token_lower in PLACE_NOUNS:
                    # Check if this noun (or its head) is in a prep phrase
                    if token.dep_ in ('pobj', 'dobj', 'attr', 'nsubj', 'compound'):
                        # Walk up to find the governing preposition
                        head = token.head
                        if token.dep_ == 'compound':
                            # The actual head noun is the parent; check its prep
                            head = head.head
                        if head.pos_ == 'ADP':
                            detections['place'] = get_full_prep_phrase_text(head)
                        elif head.head.pos_ == 'ADP':
                            detections['place'] = get_full_prep_phrase_text(head.head)
                        else:
                            # No preposition, but the noun itself is a place
                            detections['place'] = token.text
                    break
        return detections
    # --- Output ---
    def emit_output(self):
        """Build and send the weighted prompt list."""
        strength = self.strength()
        prompt_list = []
        for slot_name in self.slot_names:
            value = self.slots[slot_name]
            if value != '':
                weight_prop = self.weight_properties[slot_name]
                weight = weight_prop() * strength
                prompt_list.append([value, weight])
        self.context_output.send(prompt_list)
        # Also send the raw dict for debugging
        self.dict_output.send(dict(self.slots))
    # --- Status display ---
    def update_status(self):
        parts = []
        for name in self.slot_names:
            if self.slots[name] != '':
                short_name = name.replace('time_of_day', 'time').replace('weather', 'wx')
                parts.append(f'{short_name}: {self.slots[name]}')
        status = ' | '.join(parts) if parts else '(no context)'
        self.status_label.set(status)