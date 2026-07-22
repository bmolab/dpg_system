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
# Named historical events/periods that read as eras. Multi-word names are
# unambiguous enough to match anywhere. Loadable at runtime via the
# 'era events' input; the match pattern is rebuilt on merge.
ERA_EVENT_NAMES = {
    'first world war', 'second world war', 'world war one', 'world war two',
    'world war ii', 'world war i', 'great war', 'cold war', 'civil war',
    'great depression', 'french revolution', 'american revolution',
    'russian revolution', 'industrial revolution', 'spanish inquisition',
    'black death', 'middle ages', 'dark ages', 'stone age', 'ice age',
    'bronze age', 'iron age', 'jazz age', 'gilded age', 'space age',
    'viking age', 'belle époque', 'belle epoque', 'roaring twenties',
    'wild west', 'gold rush', 'space race', 'roman empire', 'antiquity',
    'prohibition', 'crusades',
}
ERA_PATTERNS = [
    r'\b(?:in the\s+)?(?:medieval|renaissance|baroque|victorian|edwardian|'
    r'modern|futuristic|ancient|prehistoric|classical|industrial|'
    r'contemporary|postmodern)\s*(?:era|period|times|age)?\b',
    # Single-word events are era readings only behind a locating preposition
    # ('in the depression', not 'her depression deepened')
    r'\b(?:in|during|before|after|since|throughout)\s+the\s+'
    r'(?:war|revolution|depression|plague|famine|blitz|occupation|'
    r'renaissance|enlightenment|reformation|restoration|regency|'
    r'inquisition|apocalypse)\b',
]
# ---------------------------------------------------------------------------
# Time scale ladder
# ---------------------------------------------------------------------------
# Time spans three registers — 'era' (historical setting), 'time_of_year'
# (season/month/day), and 'time_of_day' — but a single unified ladder backs
# them all, so knockout runs across the boundaries (a year jump invalidates
# a stale 'at dusk'). Coarse -> fine:
TIME_SCALES = ['era', 'century', 'decade', 'year', 'season', 'month', 'day', 'time_of_day']
ERA_TIME_SCALES = ['era', 'century', 'decade', 'year']  # compose into 'era'
TIME_OF_YEAR_SCALES = ['season', 'month', 'day']        # compose into 'time_of_year'
# the 'time_of_day' rung is its register's sole content
TIME_DECAY_MULTIPLIERS = {
    'era': 16, 'century': 16, 'decade': 12, 'year': 8,
    'season': 6, 'month': 4, 'day': 2, 'time_of_day': 1,
}
SEASON_WORDS = {'winter', 'spring', 'summer', 'autumn', 'fall'}
# Seasons safe to match as bare tokens ('spring' coils, things 'fall')
BARE_SEASON_WORDS = {'winter', 'summer', 'autumn'}
MONTH_WORDS = {
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
}
WEEKDAY_WORDS = {
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
}
HOLIDAY_WORDS = {
    'christmas', 'christmas eve', 'easter', 'halloween', 'thanksgiving',
    "new year's eve", "new year's day", 'midsummer', 'solstice', 'equinox',
    'hanukkah', 'ramadan', 'diwali',
}
_CENTURY_PATTERN = re.compile(r'\b\d{1,2}(?:th|st|nd|rd)[- ]century\b', re.IGNORECASE)
_DECADE_PATTERN = re.compile(
    r'\b(?:[12]\d{2}0s|the\s+(?:roaring\s+)?'
    r'(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties))\b',
    re.IGNORECASE)
# Bare 4-digit numbers are often measurements; require a governing preposition
# (NER catches most bare year mentions anyway)
_YEAR_PATTERN = re.compile(r'\b(?:in|by|since|around|circa|of)\s+[12]\d{3}\b', re.IGNORECASE)
# Month word must be capitalized ('may I', 'they march', 'an august figure')
_MONTH_PATTERN = re.compile(
    r'\b(?:in|during|by|since|until|late|early|mid)[- ]?\s*(?:early\s+|late\s+|mid[- ]?)?'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\b')
_SEASON_PATTERN = re.compile(
    r'\b(?:in|during|that|one|late|early)\s+(?:the\s+)?(?:winter|spring|summer|autumn|fall)\b',
    re.IGNORECASE)
# Numeric grain extraction for containment checks: '1923' provably sits in
# 'the 1920s' and the '20th century'.
def _extract_year(text):
    m = re.search(r'\b([12]\d{3})\b', text)
    return int(m.group(1)) if m else None
def _extract_decade(text):
    m = re.search(r'\b([12]\d{2}0)s\b', text)
    return int(m.group(1)) if m else None
def _extract_century(text):
    m = re.search(r'\b(\d{1,2})(?:th|st|nd|rd)[- ]century\b', text.lower())
    return int(m.group(1)) if m else None
# Rough year ranges for known era words — enough to catch contradictions
# ('medieval' vs 'In 1947'), not meant as historiography.
ERA_YEAR_RANGES = {
    'prehistoric': (-100000, -3000), 'neolithic': (-10000, -3000),
    'bronze age': (-3300, -1200), 'iron age': (-1200, -550),
    'ancient': (-3000, 500), 'classical': (-800, 500),
    'medieval': (500, 1500), 'renaissance': (1300, 1700),
    'baroque': (1600, 1750), 'industrial': (1760, 1914),
    'victorian': (1837, 1901), 'edwardian': (1901, 1910),
    'retro': (1950, 1990), 'vintage': (1920, 1980),
    'modern': (1890, 2100), 'contemporary': (1945, 2100),
    'postmodern': (1945, 2100), 'futuristic': (2050, 100000),
}
# Month -> season containment (northern-hemisphere convention)
_MONTH_SEASON = {
    'december': 'winter', 'january': 'winter', 'february': 'winter',
    'march': 'spring', 'april': 'spring', 'may': 'spring',
    'june': 'summer', 'july': 'summer', 'august': 'summer',
    'september': 'autumn', 'october': 'autumn', 'november': 'autumn',
}
def _era_range(text):
    t = text.lower()
    for word, rng in ERA_YEAR_RANGES.items():
        if word in t:
            return rng
    return None
def _representative_year(scale, value):
    if scale == 'year':
        return _extract_year(value)
    if scale == 'decade':
        d = _extract_decade(value)
        return d + 5 if d is not None else None
    if scale == 'century':
        c = _extract_century(value)
        return (c - 1) * 100 + 50 if c is not None else None
    return None
def time_grains_consistent(coarse_scale, coarse_value, fine_scale, fine_value):
    """Containment check between two time-ladder values.
    Returns True (consistent), False (contradictory), or None (unknowable)."""
    if coarse_scale == 'era':
        rng = _era_range(coarse_value)
        if rng is not None:
            y = _representative_year(fine_scale, fine_value)
            if y is not None:
                return rng[0] <= y <= rng[1]
        return None
    if coarse_scale == 'season' and fine_scale == 'month':
        season = next((w for w in re.findall(r'[a-z]+', coarse_value.lower())
                       if w in SEASON_WORDS), None)
        month = next((w for w in re.findall(r'[a-z]+', fine_value.lower())
                      if w in MONTH_WORDS), None)
        if season is not None and month is not None:
            if season == 'fall':
                season = 'autumn'
            return _MONTH_SEASON[month] == season
        return None
    if coarse_scale == 'decade' and fine_scale == 'year':
        d, y = _extract_decade(coarse_value), _extract_year(fine_value)
        if d is not None and y is not None:
            return (y // 10) * 10 == d
    elif coarse_scale == 'century':
        c = _extract_century(coarse_value)
        if c is not None:
            if fine_scale == 'year':
                y = _extract_year(fine_value)
                if y is not None:
                    return (y - 1) // 100 + 1 == c
            elif fine_scale == 'decade':
                d = _extract_decade(fine_value)
                if d is not None:
                    return d // 100 + 1 == c
    return None
PLACE_PATTERNS = [
    r'\b(?:in|inside|within|into)\s+(?:a|an|the)\s+'
    r'(?:forest|desert|jungle|ocean|sea|mountain|mountains|cave|castle|cathedral|'
    r'church|temple|mosque|synagogue|shrine|garden|field|meadow|swamp|marsh|'
    r'city hall|town hall|city|'
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
# ---------------------------------------------------------------------------
# Place scale ladder
# ---------------------------------------------------------------------------
# Places live at multiple scales. A new term at scale L knocks out everything
# finer than L (you moved), while finer terms refine what's already there
# ("Paris" then "a café" -> "a café, in Paris"). Coarse -> fine:
PLACE_SCALES = ['country', 'region', 'settlement', 'site', 'interior']
# Decay multipliers: rooms change often in a narrative, countries rarely.
# Effective decay for a scale = decay_chunks * multiplier.
PLACE_DECAY_MULTIPLIERS = {
    'country': 16, 'region': 12, 'settlement': 8, 'site': 3, 'interior': 1,
}
# Small gazetteer for classifying spaCy GPE entities, which lump countries,
# states, and cities into one label. Any GPE not found here is assumed to be
# settlement-scale (right most of the time; wrong cases land adjacent on the
# ladder). Deliberately omitted ambiguous names: 'new york', 'washington',
# 'victoria', 'quebec city' vs 'quebec' (province wins).
COUNTRY_NAMES = {
    'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'argentina',
    'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain',
    'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin',
    'bhutan', 'bolivia', 'bosnia', 'botswana', 'brazil', 'brunei',
    'bulgaria', 'burkina faso', 'burma', 'burundi', 'cambodia', 'cameroon',
    'canada', 'chad', 'chile', 'china', 'colombia', 'congo', 'costa rica',
    'croatia', 'cuba', 'cyprus', 'czechia', 'czech republic', 'denmark',
    'djibouti', 'dominica', 'dominican republic', 'ecuador', 'egypt',
    'el salvador', 'england', 'eritrea', 'estonia', 'eswatini', 'ethiopia',
    'fiji', 'finland', 'france', 'gabon', 'gambia', 'germany', 'ghana',
    'greece', 'greenland', 'grenada', 'guatemala', 'guinea', 'guyana',
    'haiti', 'holland', 'honduras', 'hungary', 'iceland', 'india',
    'indonesia', 'iran', 'iraq', 'ireland', 'israel', 'italy',
    'ivory coast', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya',
    'kiribati', 'kosovo', 'kuwait', 'kyrgyzstan', 'laos', 'latvia',
    'lebanon', 'lesotho', 'liberia', 'libya', 'liechtenstein', 'lithuania',
    'luxembourg', 'madagascar', 'malawi', 'malaysia', 'maldives', 'mali',
    'malta', 'mauritania', 'mauritius', 'mexico', 'moldova', 'monaco',
    'mongolia', 'montenegro', 'morocco', 'mozambique', 'myanmar', 'namibia',
    'nepal', 'netherlands', 'new zealand', 'nicaragua', 'niger', 'nigeria',
    'north korea', 'north macedonia', 'norway', 'oman', 'pakistan', 'palau',
    'palestine', 'panama', 'papua new guinea', 'paraguay', 'persia', 'peru',
    'philippines', 'poland', 'portugal', 'qatar', 'romania', 'russia',
    'rwanda', 'samoa', 'san marino', 'saudi arabia', 'scotland', 'senegal',
    'serbia', 'seychelles', 'sierra leone', 'singapore', 'slovakia',
    'slovenia', 'somalia', 'south africa', 'south korea', 'south sudan',
    'spain', 'sri lanka', 'sudan', 'suriname', 'sweden', 'switzerland',
    'syria', 'taiwan', 'tajikistan', 'tanzania', 'thailand', 'togo',
    'tonga', 'trinidad', 'tunisia', 'turkey', 'turkmenistan', 'tuvalu',
    'uganda', 'ukraine', 'united arab emirates', 'uae', 'united kingdom',
    'uk', 'britain', 'great britain', 'united states', 'usa', 'america',
    'united states of america', 'uruguay', 'uzbekistan', 'vanuatu',
    'vatican', 'venezuela', 'vietnam', 'wales', 'yemen', 'zambia',
    'zimbabwe',
}
REGION_NAMES = {
    # US states (minus ambiguous 'new york', 'washington')
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
    'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
    'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
    'new hampshire', 'new jersey', 'new mexico', 'north carolina',
    'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
    'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas',
    'utah', 'vermont', 'virginia', 'west virginia', 'wisconsin', 'wyoming',
    # Canadian provinces / territories
    'alberta', 'british columbia', 'manitoba', 'new brunswick',
    'newfoundland', 'nova scotia', 'ontario', 'quebec', 'saskatchewan',
    'yukon', 'nunavut',
    # Australian states
    'queensland', 'tasmania', 'new south wales',
    # Well-known regions elsewhere
    'bavaria', 'tuscany', 'provence', 'normandy', 'brittany', 'catalonia',
    'andalusia', 'castile', 'siberia', 'patagonia', 'amazonia', 'kashmir',
    'tibet', 'punjab', 'yorkshire', 'cornwall', 'wessex', 'transylvania',
    'bohemia', 'moravia', 'flanders', 'wallonia', 'galicia', 'lombardy',
    'sicily', 'sardinia', 'corsica', 'crete', 'appalachia', 'lapland',
    'okinawa', 'hokkaido', 'kyushu', 'scandinavia', 'new england',
    'the midwest', 'midwest',
}
# Common-noun place vocabulary, partitioned by scale. Used for
# dependency-based extraction and for classifying pattern matches.
SETTLEMENT_NOUNS = {
    'city', 'village', 'town', 'metropolis', 'hamlet', 'settlement',
}
SITE_NOUNS = {
    # Natural environments
    'forest', 'desert', 'jungle', 'ocean', 'sea', 'mountain', 'mountains',
    'cave', 'valley', 'canyon', 'gorge', 'field', 'meadow', 'swamp', 'marsh',
    'prairie', 'tundra', 'steppe', 'savanna', 'island', 'glacier', 'volcano',
    # Built environments
    'castle', 'cathedral', 'church', 'temple', 'mosque', 'shrine', 'palace',
    'tower', 'dungeon', 'fortress', 'citadel', 'ruin', 'ruins', 'courtyard',
    'alley', 'alleyway', 'street', 'market',
    'marketplace', 'plaza', 'square', 'boulevard', 'avenue',
    'tavern', 'pub', 'bar', 'restaurant', 'café', 'cafe', 'inn',
    'library', 'museum', 'gallery', 'theater', 'theatre', 'arena',
    'colosseum', 'amphitheatre', 'amphitheater', 'stadium',
    'hospital', 'prison', 'asylum', 'monastery', 'convent', 'abbey',
    'warehouse', 'factory', 'laboratory', 'lab', 'studio', 'workshop',
    'greenhouse', 'conservatory', 'observatory', 'chapel',
    # Landscape features
    'cliff', 'shore', 'shoreline', 'riverbank', 'hilltop', 'rooftop', 'roof',
    'bridge', 'dock', 'pier', 'beach', 'lake', 'pond', 'river', 'stream',
    'waterfall', 'fountain', 'harbor', 'harbour', 'port', 'wharf',
    # Vessels-as-places
    'ship', 'boat', 'deck',
    # Other
    'graveyard', 'cemetery', 'garden', 'orchard', 'vineyard', 'farm',
    'space station', 'spaceship', 'submarine', 'train station', 'airport',
    'park', 'city hall', 'town hall', 'concert hall', 'music hall',
    'dance hall', 'guildhall',
}
INTERIOR_NOUNS = {
    'bedroom', 'kitchen', 'basement', 'attic', 'hallway', 'corridor', 'hall',
    'ballroom', 'throne room', 'cellar', 'crypt', 'chamber', 'parlor',
    'parlour', 'salon', 'study', 'nursery',
}
# Union kept for backward compatibility
PLACE_NOUNS = SETTLEMENT_NOUNS | SITE_NOUNS | INTERIOR_NOUNS
# Prepositions recognized at the start of stored place phrases; used when
# composing the ladder so bare terms get an 'in ' prefix.
_PLACE_PREPOSITIONS = {
    'in', 'at', 'on', 'by', 'near', 'inside', 'within', 'under',
    'beneath', 'along', 'beside', 'across', 'outside', 'among', 'amid',
    'amidst', 'upon', 'over', 'from', 'through',
}
# Motion prepositions are normalized to 'in' when composing — the register
# describes where the scene is, not how it was entered. 'of' arrives from
# phrases like 'the mountains of Bavaria'.
_MOTION_PREPOSITIONS = {'to', 'into', 'onto', 'toward', 'towards', 'of'}
# Known country for gazetteer regions. Used only for a consistency check:
# a new region whose country conflicts with the current country slot clears
# it (we never fabricate a country that wasn't mentioned). Regions with
# ambiguous or contested containment are simply left unmapped.
COUNTRY_ALIASES = {
    'usa': 'united states', 'america': 'united states',
    'united states of america': 'united states',
    'uk': 'united kingdom', 'britain': 'united kingdom',
    'great britain': 'united kingdom',
    'holland': 'netherlands', 'czech republic': 'czechia',
    'persia': 'iran', 'burma': 'myanmar',
}
REGION_COUNTRY = {}
for _state in [
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
        'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
        'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
        'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
        'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
        'new hampshire', 'new jersey', 'new mexico', 'north carolina',
        'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
        'rhode island', 'south carolina', 'south dakota', 'tennessee',
        'texas', 'utah', 'vermont', 'virginia', 'west virginia',
        'wisconsin', 'wyoming', 'appalachia', 'new england', 'midwest',
        'the midwest']:
    REGION_COUNTRY[_state] = 'united states'
for _prov in ['alberta', 'british columbia', 'manitoba', 'new brunswick',
              'newfoundland', 'nova scotia', 'ontario', 'quebec',
              'saskatchewan', 'yukon', 'nunavut']:
    REGION_COUNTRY[_prov] = 'canada'
for _state in ['queensland', 'tasmania', 'new south wales']:
    REGION_COUNTRY[_state] = 'australia'
REGION_COUNTRY.update({
    'bavaria': 'germany', 'tuscany': 'italy', 'lombardy': 'italy',
    'sicily': 'italy', 'sardinia': 'italy',
    'provence': 'france', 'normandy': 'france', 'brittany': 'france',
    'corsica': 'france',
    'catalonia': 'spain', 'andalusia': 'spain', 'castile': 'spain',
    'siberia': 'russia', 'transylvania': 'romania',
    'bohemia': 'czechia', 'moravia': 'czechia',
    'flanders': 'belgium', 'wallonia': 'belgium',
    'crete': 'greece', 'yorkshire': 'england', 'cornwall': 'england',
    'wessex': 'england', 'okinawa': 'japan', 'hokkaido': 'japan',
    'kyushu': 'japan',
})
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
    # Lighting styles
    'chiaroscuro', 'backlit', 'candlelit', 'neon', 'bioluminescent',
    'high contrast', 'low key', 'high key', 'dramatic lighting',
    'volumetric lighting', 'rim lighting', 'soft lighting',
    'golden light', 'cold light', 'warm light',
}
# The physical/format substrate of an image — distinct from style (movement,
# mood, lighting). One medium at a time; a new one replaces the old.
MEDIUM_VOCAB = {
    # Painting / drawing techniques
    'watercolor', 'watercolour', 'oil painting', 'gouache', 'acrylic',
    'tempera', 'fresco', 'mural', 'charcoal', 'pencil sketch', 'ink drawing',
    'pen and ink', 'pastel', 'illuminated manuscript', 'icon painting',
    # Printmaking
    'woodcut', 'linocut', 'lithograph', 'engraving', 'etching', 'silkscreen',
    'screenprint', 'woodblock print', 'ukiyo-e',
    # Photographic
    'photograph', 'photographic', 'photo', 'polaroid', 'polaroid photo',
    'daguerreotype', 'tintype', 'snapshot', 'film still', 'film photograph',
    # Textile / fiber
    'tapestry', 'needlepoint', 'embroidery', 'embroidered', 'cross-stitch',
    'quilt', 'quilted', 'quilting', 'persian carpet', 'oriental rug',
    'kilim', 'batik', 'macramé', 'macrame', 'brocade', 'lacework',
    'woven', 'weaving',
    # Ceramics / glass
    'ceramic', 'porcelain', 'pottery', 'terracotta', 'delftware',
    'majolica', 'faience', 'cloisonné', 'cloisonne', 'enamelwork',
    'blown glass', 'stained glass',
    # Carving / metal / dimensional
    'mosaic', 'collage', 'diorama', 'origami', 'papercut', 'sculpture',
    'bas-relief', 'woodcarving', 'stone carving', 'marquetry', 'scrimshaw',
    'wrought iron', 'repoussé', 'gold leaf', 'gilded', 'papier-mâché',
    'sand mandala', 'silhouette',
    # More painting / drawing
    'ink wash', 'sumi-e', 'calligraphy', 'graffiti', 'street art',
    'airbrush', 'encaustic', 'grisaille', 'silverpoint', 'sgraffito',
    'velvet painting', 'sand painting', 'cave painting', 'petroglyph',
    'persian miniature', 'miniature painting', 'oil on canvas',
    'chalk drawing', 'sidewalk chalk', 'chalkboard', 'altarpiece',
    'triptych', 'diptych', 'scroll painting', 'thangka', 'folding screen',
    # More photographic / film
    'cyanotype', 'ambrotype', 'photogram', 'x-ray', 'hologram',
    '35mm film', 'super 8', 'vhs', 'lomography', 'double exposure',
    'security camera footage',
    # More printmaking
    'mezzotint', 'aquatint', 'drypoint', 'monotype', 'risograph',
    'letterpress', 'stencil',
    # More print ephemera / illustration
    'wanted poster', 'album cover', 'magazine cover', 'tarot card',
    'playing card', 'trading card', 'banknote', 'botanical illustration',
    'scientific illustration', 'anatomical drawing', 'book illustration',
    'courtroom sketch', 'police sketch', 'storyboard', 'zine',
    # More moving / digital
    'claymation', 'stop motion', 'ascii art', 'vector art', 'voxel art',
    'neon sign',
    # Print formats
    'poster', 'propaganda poster', 'travel poster', 'movie poster',
    'postage stamp', 'postcard', 'billboard', 'blueprint',
    'technical drawing', 'comic book', 'comic strip', 'cartoon',
    # Digital
    'pixel art', 'digital art', 'digital painting', 'concept art',
    '3d render', 'anime', 'manga',
}
# ---------------------------------------------------------------------------
# Actor roster (salience-based pronoun resolution, Lappin & Leass style)
# ---------------------------------------------------------------------------
# Salience boost by grammatical role: prominence of position predicts what
# pronouns will refer to (Centering Theory).
ACTOR_ROLE_WEIGHTS = {
    'nsubj': 1.0, 'nsubjpass': 1.0, 'dobj': 0.6, 'conj': 0.6,
    'iobj': 0.5, 'attr': 0.5, 'appos': 0.5, 'pobj': 0.35, 'poss': 0.35,
}
# Nouns that can create an actor entry from object position (subjects create
# unconditionally — actorhood is functional: things that act are actors)
ANIMATE_NOUNS = {
    'man', 'woman', 'child', 'boy', 'girl', 'person', 'people', 'figure',
    'stranger', 'soldier', 'guard', 'knight', 'king', 'queen', 'prince',
    'princess', 'emperor', 'empress', 'lord', 'lady', 'gentleman',
    'priest', 'monk', 'nun', 'doctor', 'nurse', 'teacher', 'artist',
    'painter', 'dancer', 'singer', 'musician', 'poet', 'writer', 'actor',
    'actress', 'farmer', 'fisherman', 'merchant', 'sailor', 'captain',
    'general', 'servant', 'maid', 'butler', 'beggar', 'prisoner', 'hunter',
    'shepherd', 'clown', 'magician', 'witch', 'wizard', 'ghost', 'angel',
    'demon', 'god', 'goddess', 'thief', 'traveler', 'traveller', 'worker',
    'miner', 'blacksmith', 'baker', 'butcher', 'tailor', 'weaver',
    'mother', 'father', 'sister', 'brother', 'daughter', 'son', 'baby',
    'infant', 'grandmother', 'grandfather', 'aunt', 'uncle', 'friend',
    'enemy', 'neighbor', 'neighbour', 'crowd', 'mob', 'family', 'couple',
    'dog', 'cat', 'horse', 'wolf', 'bird', 'crow', 'raven', 'fox', 'bear',
    'lion', 'deer', 'rabbit', 'snake', 'dragon',
}
# Animals take he/she or it — leave humanness unknown
ANIMAL_NOUNS = {
    'dog', 'cat', 'horse', 'wolf', 'bird', 'crow', 'raven', 'fox', 'bear',
    'lion', 'deer', 'rabbit', 'snake', 'dragon',
}
# Gender evidenced by the noun itself (never guessed from names)
GENDERED_NOUNS = {
    'man': 'masc', 'woman': 'fem', 'boy': 'masc', 'girl': 'fem',
    'king': 'masc', 'queen': 'fem', 'prince': 'masc', 'princess': 'fem',
    'lord': 'masc', 'lady': 'fem', 'gentleman': 'masc',
    'emperor': 'masc', 'empress': 'fem', 'actress': 'fem',
    'mother': 'fem', 'father': 'masc', 'sister': 'fem', 'brother': 'masc',
    'daughter': 'fem', 'son': 'masc', 'grandmother': 'fem',
    'grandfather': 'masc', 'aunt': 'fem', 'uncle': 'masc',
    'nun': 'fem', 'monk': 'masc', 'maid': 'fem',
}
TITLE_GENDER = {
    'mr': 'masc', 'mister': 'masc', 'sir': 'masc', 'lord': 'masc',
    'mrs': 'fem', 'ms': 'fem', 'miss': 'fem', 'lady': 'fem',
    'madam': 'fem', 'madame': 'fem',
    'dr': None, 'professor': None, 'captain': None,
}
_PRONOUN_CLASSES = {
    'he': 'masc', 'him': 'masc', 'himself': 'masc', 'his': 'masc',
    'she': 'fem', 'her': 'fem', 'herself': 'fem', 'hers': 'fem',
    'they': 'plur', 'them': 'plur', 'themselves': 'plur',
    'their': 'plur', 'theirs': 'plur',
    'it': 'neut', 'its': 'neut', 'itself': 'neut',
}
WEATHER_VOCAB = {
    'rainy', 'foggy', 'misty', 'snowy', 'stormy', 'overcast', 'sunny',
    'cloudy', 'hazy', 'windy', 'drizzling', 'thundering', 'blazing',
    'rain-soaked', 'sun-drenched', 'frost-covered', 'dew-covered',
    'humid', 'arid', 'freezing', 'sweltering', 'muggy', 'breezy',
    'torrential', 'blizzard', 'hurricane', 'typhoon',
}
# Weather lemmas — matched against token.lemma_ so inflected forms
# ("raining", "rained", "snows") are caught. Only unambiguous weather
# words belong here (no 'wind': "wind the clock" would false-positive).
WEATHER_LEMMAS = {
    'rain', 'snow', 'fog', 'mist', 'haze', 'storm', 'thunder', 'lightning',
    'hail', 'sleet', 'drizzle', 'downpour', 'blizzard', 'hurricane',
    'typhoon', 'gale', 'frost', 'sunshine', 'thunderstorm', 'rainstorm',
    'snowstorm', 'monsoon', 'cloudburst', 'squall',
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
def get_ent_phrase(ent):
    """Full entity span text, with its governing preposition when present.
    Unlike get_prep_phrase_text(ent.root), this never truncates a multi-word
    entity to its root token ('One summer evening' stays whole)."""
    head = ent.root.head
    if head.pos_ == 'ADP' and head.i < ent.start:
        return ent.doc[head.i:ent.end].text
    return ent.text
def trim_dangling_prep(text):
    """Drop trailing prepositions left by span extension
    ('On a cold Sunday morning in' -> 'On a cold Sunday morning')."""
    words = text.split()
    while words and words[-1].lower() in ('in', 'at', 'on', 'of', 'by', 'to',
                                          'from', 'with', 'under', 'over'):
        words.pop()
    return ' '.join(words)
def get_full_prep_phrase_text(prep_token):
    """Given a preposition token, return the full prepositional phrase
    including all modifiers (adjectives, compounds, determiners)."""
    subtree = list(prep_token.subtree)
    if subtree:
        # Span text keeps the original surface ("the ship's deck", not
        # "the ship 's deck")
        return prep_token.doc[subtree[0].i:subtree[-1].i + 1].text
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
        self.slot_names = ['time_of_day', 'time_of_year', 'era', 'place', 'weather', 'style', 'medium', 'actors', 'artist']
        self.slots = {name: '' for name in self.slot_names}
        self.slot_ages = {name: 0 for name in self.slot_names}  # chunks since last update
        self.chunk_count = 0
        # Place ladder: per-scale values and ages. slots['place'] holds the
        # composed string (finest-first) so outputs are unchanged.
        self.place_slots = {scale: '' for scale in PLACE_SCALES}
        self.place_ages = {scale: 0 for scale in PLACE_SCALES}
        # Time ladder: one unified ladder backing the 'era' and 'time_of_day'
        # registers (coarse half and fine half respectively).
        self.time_slots = {scale: '' for scale in TIME_SCALES}
        self.time_ages = {scale: 0 for scale in TIME_SCALES}
        # Actor roster: salience-ranked entries. slots['actors'] holds the
        # composed top-of-roster; emit_output sends each actor separately
        # with salience-scaled weight.
        self.actors = []
        # --- Loadable vocabularies ---
        self.time_vocab = set(TIME_OF_DAY_WORDS)
        self.era_vocab = set(ERA_WORDS)
        self.era_event_vocab = set(ERA_EVENT_NAMES)
        self._rebuild_era_event_pattern()
        self.style_vocab = set(STYLE_VOCAB)
        self.medium_vocab = set(MEDIUM_VOCAB)
        self.weather_vocab = set(WEATHER_VOCAB)
        self.weather_lemmas = set(WEATHER_LEMMAS)
        self.countries = set(COUNTRY_NAMES)
        self.regions = set(REGION_NAMES)
        self.settlement_nouns = set(SETTLEMENT_NOUNS)
        self.site_nouns = set(SITE_NOUNS)
        self.interior_nouns = set(INTERIOR_NOUNS)
        self._rebuild_place_lookup()
        # --- Inputs ---
        self.text_input = self.add_input('text in', triggers_execution=True)
        self.clear_button = self.add_input('clear', widget_type='button', callback=self.clear_all)
        self.set_context_input = self.add_input('set context', triggers_execution=True)
        # Vocabulary loading inputs
        self.load_time_vocab = self.add_input('time vocab', callback=self.receive_time_vocab)
        self.load_place_vocab = self.add_input('place vocab', callback=self.receive_place_vocab)
        self.load_style_vocab = self.add_input('style vocab', callback=self.receive_style_vocab)
        self.load_medium_vocab = self.add_input('medium vocab', callback=self.receive_medium_vocab)
        self.load_weather_vocab = self.add_input('weather vocab', callback=self.receive_weather_vocab)
        self.load_era_events = self.add_input('era events', callback=self.receive_era_events)
        # Artist input — connect from FuzzyMatchNode's 'replacement out'
        self.artist_input = self.add_input('artist in', triggers_execution=True)
        # --- Properties (per-slot weights) ---
        self.time_weight = self.add_property('time weight', widget_type='drag_float', default_value=0.5)
        self.time_of_year_weight = self.add_property('time of year weight', widget_type='drag_float', default_value=0.4)
        self.era_weight = self.add_property('era weight', widget_type='drag_float', default_value=0.5)
        self.place_weight = self.add_property('place weight', widget_type='drag_float', default_value=0.5)
        self.weather_weight = self.add_property('weather weight', widget_type='drag_float', default_value=0.3)
        self.style_weight = self.add_property('style weight', widget_type='drag_float', default_value=0.5)
        self.medium_weight = self.add_property('medium weight', widget_type='drag_float', default_value=0.5)
        self.actor_weight = self.add_property('actor weight', widget_type='drag_float', default_value=0.5)
        self.artist_weight = self.add_property('artist weight', widget_type='drag_float', default_value=0.5)
        self.strength = self.add_property('strength', widget_type='drag_float', default_value=1.0)
        self.include_empty_keys = self.add_option('include empty keys', widget_type='checkbox', default_value=False)
        self.decay_chunks = self.add_option('decay chunks', widget_type='drag_int', default_value=0)
        self.actor_decay = self.add_option('actor decay', widget_type='drag_float', default_value=0.5)
        self.max_actors = self.add_option('max actors', widget_type='drag_int', default_value=6)
        # Display current state
        self.status_label = self.add_label('')
        # --- Outputs ---
        self.context_output = self.add_output('context out')
        self.dict_output = self.add_output('context dict')
        self.detected_output = self.add_output('detected')
        # Weight lookup by slot name
        self.weight_properties = {
            'time_of_day': self.time_weight,
            'time_of_year': self.time_of_year_weight,
            'era': self.era_weight,
            'place': self.place_weight,
            'weather': self.weather_weight,
            'style': self.style_weight,
            'medium': self.medium_weight,
            'actors': self.actor_weight,
            'artist': self.artist_weight,
        }
    # --- Vocabulary receivers ---
    def receive_time_vocab(self):
        data = self.load_time_vocab()
        self._merge_vocab(data, self.time_vocab)
    def receive_place_vocab(self):
        """Accepts a list of place nouns (added at site scale), or a dict
        keyed by scale name ('country', 'region', 'settlement', 'site',
        'interior') with lists of terms, or a path to a JSON file holding
        either form."""
        data = self.load_place_vocab()
        if type(data) == str and os.path.exists(data):
            try:
                with open(data, 'r') as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError, ValueError) as e:
                print(f"ContextTrackerNode place vocab load failed: {e}")
                return
        if type(data) == dict:
            targets = {
                'country': self.countries,
                'region': self.regions,
                'settlement': self.settlement_nouns,
                'site': self.site_nouns,
                'interior': self.interior_nouns,
            }
            for key, words in data.items():
                if key in targets and type(words) == list:
                    for word in words:
                        targets[key].add(str(word).lower())
        elif type(data) == list:
            for item in data:
                if type(item) == str:
                    self.site_nouns.add(item.lower())
        self._rebuild_place_lookup()
    def _rebuild_place_lookup(self):
        self.place_nouns = self.settlement_nouns | self.site_nouns | self.interior_nouns
        self.multiword_place_nouns = {n for n in self.place_nouns if ' ' in n}
    # --- Place scale classification ---
    def noun_scale(self, word):
        if word in self.settlement_nouns:
            return 'settlement'
        if word in self.interior_nouns:
            return 'interior'
        if word in self.site_nouns:
            return 'site'
        return None
    def classify_gpe(self, name):
        """Classify a spaCy GPE entity into a ladder scale via the gazetteer.
        Unknown GPEs default to settlement scale."""
        n = name.lower()
        if n.startswith('the '):
            n = n[4:]
        if n in self.countries:
            return 'country'
        if n in self.regions:
            return 'region'
        return 'settlement'
    def _is_anaphoric_repeat(self, new_value, current_value):
        """True when a new same-scale phrase is a back-reference to the
        current value rather than a move: a definite generic ('the city',
        no proper names) or a bare generic noun, while the current value is
        a proper name or shares the noun. Indefinites ('a nearby village')
        signal a genuine move and are never anaphoric."""
        if current_value == '':
            return False
        preps = _PLACE_PREPOSITIONS | _MOTION_PREPOSITIONS
        new_words = new_value.split()
        start = 1 if new_words and new_words[0].lower() in preps else 0
        content = new_words[start:]
        if not content or any(w[:1].isupper() for w in content):
            return False  # proper name (or empty): treat as real information
        cur_words = current_value.split()
        if cur_words and cur_words[0].lower() in preps:
            cur_words = cur_words[1:]
        current_proper = any(w[:1].isupper() for w in cur_words)
        if content[0] == 'the':
            # Definite generic: anaphoric against a proper name or when the
            # noun is already part of the current value
            if current_proper:
                return True
            cur_lower = current_value.lower()
            return any(w.lower() in cur_lower for w in content[1:])
        if len(content) == 1 and current_proper:
            # Bare generic noun ('city' as subject) vs a proper name
            return True
        return False
    def _match_vocab(self, doc, text_lower, hyphenated, vocab, use_lemmas=False,
                     expand_modifiers=False):
        """Match a vocabulary against 3- and 2-word phrases (longest first,
        so 'propaganda poster' beats 'poster'), single tokens (optionally by
        lemma), and hyphenated compounds. Returns the match or None.
        With expand_modifiers, a match grows to include its head noun's
        modifiers ('black and white photograph', not bare 'photograph')."""
        word_lists = [re.findall(r"[\w'-]+", text_lower)]
        if use_lemmas:
            # Lemma pass so 'postage stamps' matches 'postage stamp'
            word_lists.append([t.lemma_.lower() for t in doc if not t.is_punct])
        for words in word_lists:
            for size in (3, 2):
                for i in range(len(words) - size + 1):
                    phrase = ' '.join(words[i:i + size])
                    if phrase in vocab:
                        if expand_modifiers:
                            phrase = self._with_modifiers(doc, phrase)
                        return phrase
        for token in doc:
            match = None
            if token.text.lower() in vocab:
                match = token.text.lower()
            elif use_lemmas and token.lemma_.lower() in vocab:
                match = token.lemma_.lower()
            if match is not None:
                if expand_modifiers:
                    match = self._with_modifiers(doc, match)
                return match
        for compound in hyphenated:
            if compound in vocab:
                return compound
        return None
    def _with_modifiers(self, doc, phrase):
        """Expand a vocab match to the modified noun phrase around its head
        token. Only amod/compound/nummod children (with their conjunct
        subtrees, so 'black and white' comes along whole) are pulled in —
        determiners and possessives stay out. Falls back to the bare match
        if the head can't be located or the expansion loses part of a
        multiword phrase."""
        words = phrase.split()
        for token in doc:
            if token.text.lower() != words[-1] and token.lemma_.lower() != words[-1]:
                continue
            start = token.i
            for child in token.lefts:
                if child.dep_ in ('amod', 'compound', 'nummod'):
                    start = min(start, child.left_edge.i)
            if start == token.i:
                return phrase
            span = doc[start:token.i + 1].text
            if all(w in span.lower() for w in words[:-1]):
                return span
            return phrase
        return phrase
    def _phrase_gazetteer_key(self, phrase, gazetteer):
        """Find which gazetteer name a stored phrase refers to (word-bounded),
        normalized through COUNTRY_ALIASES. Returns None if no match."""
        p = phrase.lower()
        for name in gazetteer:
            if re.search(r'\b' + re.escape(name) + r'\b', p):
                return COUNTRY_ALIASES.get(name, name)
        return None
    def classify_place_phrase(self, phrase):
        """Classify a matched place phrase by its place noun(s). The last
        matching noun wins (the head noun is usually last). Defaults to site."""
        p = phrase.lower()
        for noun in self.multiword_place_nouns:
            if noun in p:
                return self.noun_scale(noun)
        best = None
        for word in re.findall(r"[a-zà-ÿ'-]+", p):
            scale = self.noun_scale(word)
            if scale is not None:
                best = scale
        return best if best is not None else 'site'
    def receive_style_vocab(self):
        data = self.load_style_vocab()
        self._merge_vocab(data, self.style_vocab)
    def receive_medium_vocab(self):
        data = self.load_medium_vocab()
        self._merge_vocab(data, self.medium_vocab)
    def receive_weather_vocab(self):
        data = self.load_weather_vocab()
        self._merge_vocab(data, self.weather_vocab)
    def receive_era_events(self):
        """Accepts a list of named era/event phrases (or a path to a JSON
        list) — multi-word names welcome. Merged into the named-event era
        pattern."""
        data = self.load_era_events()
        self._merge_vocab(data, self.era_event_vocab)
        self._rebuild_era_event_pattern()
    def _rebuild_era_event_pattern(self):
        """Compile the named-event era pattern from the current vocab.
        Longest names first, so 'world war ii' beats 'world war i'."""
        names = sorted(self.era_event_vocab, key=len, reverse=True)
        self._era_event_pattern = re.compile(
            r'\b(?:(?:in|during|before|after|since|throughout)\s+)?(?:the\s+)?'
            r'(?:' + '|'.join(re.escape(name) for name in names) + r')\b',
            re.IGNORECASE)
    def _merge_vocab(self, data, target_set):
        if type(data) == list:
            for item in data:
                if type(item) == str:
                    target_set.add(item.lower())
        elif type(data) == str:
            # Try loading as JSON file path
            if os.path.exists(data):
                try:
                    with open(data, 'r') as f:
                        vocab_list = json.load(f)
                except (OSError, json.JSONDecodeError, ValueError) as e:
                    print(f"ContextTrackerNode vocab load failed for {data}: {e}")
                    return
                if type(vocab_list) == list:
                    for item in vocab_list:
                        target_set.add(str(item).lower())
    # --- Clear ---
    def clear_all(self, value=None):
        for name in self.slot_names:
            self.slots[name] = ''
            self.slot_ages[name] = 0
        for scale in PLACE_SCALES:
            self.place_slots[scale] = ''
            self.place_ages[scale] = 0
        for scale in TIME_SCALES:
            self.time_slots[scale] = ''
            self.time_ages[scale] = 0
        self.actors = []
        self.chunk_count = 0
        self.update_status()
        self.emit_output()
    # --- Place ladder ---
    def apply_place_detections(self, place_dets):
        """Apply scale-tagged detections {scale: phrase} to the place ladder.
        A changed value at scale L clears all finer scales (a move); finer
        detections in the same chunk repopulate them (a refinement). Coarser
        scales are left alone. Returns True if the composed place changed."""
        changed = False
        for i, scale in enumerate(PLACE_SCALES):
            if scale not in place_dets:
                continue
            value = trim_dangling_prep(place_dets[scale])
            if value == '' or value == self.place_slots[scale]:
                self.place_ages[scale] = 0
                continue
            if self._is_anaphoric_repeat(value, self.place_slots[scale]):
                # 'the city' after 'in Toronto' is a back-reference, not a move
                self.place_ages[scale] = 0
                continue
            prev = self.place_slots[scale]
            self.place_slots[scale] = value
            self.place_ages[scale] = 0
            # Consistency check: a region whose known country conflicts with
            # the current country slot clears it (a scene change like
            # Scotland -> Texas). Unmapped regions never clear.
            conflict = False
            if scale == 'region' and self.place_slots['country'] != '':
                region_key = self._phrase_gazetteer_key(value, self.regions)
                expected = REGION_COUNTRY.get(region_key)
                current = self._phrase_gazetteer_key(
                    self.place_slots['country'], self.countries)
                if expected is not None and current is not None and expected != current:
                    self.place_slots['country'] = ''
                    self.place_ages['country'] = 0
                    conflict = True
            # Knockout only on a genuine move: a value replacing another, or a
            # proven conflict. Filling a previously empty coarse slot is new
            # information from above ('a café in Paris' ... 'in France') and
            # leaves finer detail intact.
            if prev != '' or conflict:
                for finer in PLACE_SCALES[i + 1:]:
                    if finer not in place_dets and self.place_slots[finer] != '':
                        self.place_slots[finer] = ''
                        self.place_ages[finer] = 0
            self.detected_output.send([f'place.{scale}', value])
            changed = True
        if changed:
            self.compose_place()
        return changed
    def compose_place(self):
        """Compose the ladder into slots['place'], finest-first.
        Skips coarser values already contained in a finer phrase (the
        dependency extractor can capture nested prep phrases whole)."""
        parts = []
        for scale in reversed(PLACE_SCALES):
            value = self.place_slots[scale]
            if value == '':
                continue
            if any(value.lower() in p.lower() for p in parts):
                continue
            # A coarser value can also subsume an already-added finer part
            parts = [p for p in parts if p.lower() not in value.lower()]
            words = value.split()
            first = words[0].lower() if words else ''
            if first in _MOTION_PREPOSITIONS:
                value = 'in ' + ' '.join(words[1:])
            elif parts and first not in _PLACE_PREPOSITIONS:
                value = 'in ' + value
            parts.append(value)
        self.slots['place'] = ', '.join(parts)
    # --- Time ladder ---
    def apply_time_detections(self, time_dets):
        """Apply scale-tagged detections {scale: phrase} to the time ladder.
        Same rules as place, plus numeric containment: a consistent coarser
        mention never knocks out a finer value ('the 1920s' keeps '1923'),
        and a finer value clears contradictory coarser rungs ('1947' clears
        a stale '19th century'). Returns True if a register changed."""
        changed = False
        for i, scale in enumerate(TIME_SCALES):
            if scale not in time_dets:
                continue
            value = trim_dangling_prep(time_dets[scale])
            if value == '' or value == self.time_slots[scale]:
                self.time_ages[scale] = 0
                continue
            prev = self.time_slots[scale]
            self.time_slots[scale] = value
            self.time_ages[scale] = 0
            # Clear coarser numeric rungs that contradict the new value
            for coarser in TIME_SCALES[:i]:
                if self.time_slots[coarser] != '':
                    if time_grains_consistent(coarser, self.time_slots[coarser],
                                              scale, value) is False:
                        self.time_slots[coarser] = ''
                        self.time_ages[coarser] = 0
            # Knockout finer rungs: always on a genuine move; on an empty
            # fill only when the finer value provably contradicts the new one
            # ('Set in medieval times' clears a stale 'In 1947')
            for finer in TIME_SCALES[i + 1:]:
                if finer in time_dets or self.time_slots[finer] == '':
                    continue
                consistent = time_grains_consistent(scale, value,
                                                    finer, self.time_slots[finer])
                if consistent is True:
                    continue
                if prev != '' or consistent is False:
                    self.time_slots[finer] = ''
                    self.time_ages[finer] = 0
            self.detected_output.send([f'time.{scale}', value])
            changed = True
        if changed:
            self.compose_time()
        return changed
    def compose_time(self):
        """Compose the ladder into slots['era'], slots['time_of_year'], and
        slots['time_of_day'], finest-first, skipping rungs implied by a finer
        one ('in 1923' makes 'the 1920s' and '20th century' redundant)."""
        def build(scales, context=()):
            parts = []
            for idx in range(len(scales) - 1, -1, -1):
                scale = scales[idx]
                value = self.time_slots[scale]
                if value == '':
                    continue
                implied = False
                for finer in scales[idx + 1:]:
                    finer_value = self.time_slots[finer]
                    if (finer_value != ''
                            and time_grains_consistent(scale, value,
                                                       finer, finer_value) is True):
                        implied = True
                        break
                if implied:
                    continue
                if any(value.lower() in p.lower() for p in list(parts) + list(context)):
                    continue
                # A coarser value can also subsume an already-added finer part
                parts = [p for p in parts if p.lower() not in value.lower()]
                parts.append(value)
            return ', '.join(parts)
        self.slots['era'] = build(ERA_TIME_SCALES)
        self.slots['time_of_day'] = self.time_slots['time_of_day']
        # Suppress time_of_year parts the time_of_day phrase already carries
        # ('One summer evening' subsumes season 'summer')
        self.slots['time_of_year'] = build(TIME_OF_YEAR_SCALES,
                                           context=(self.slots['time_of_day'],)
                                           if self.slots['time_of_day'] else ())
    def date_ent_grains(self, text):
        """All time grains present in a DATE entity, as {scale: bare value}.
        A mixed phrase like 'One summer evening' yields both season and
        time_of_day; the caller stores the full phrase at the finest grain
        and the bare words at the coarser ones."""
        grains = {}
        t = text.lower().strip()
        if any(w in t for w in self.era_vocab):
            grains['era'] = text
        m = _CENTURY_PATTERN.search(text)
        if m:
            grains['century'] = m.group(0)
        m = _DECADE_PATTERN.search(text)
        if m:
            grains['decade'] = m.group(0)
        m = re.search(r'\b[12]\d{3}\b', text)
        if m:
            grains['year'] = m.group(0)
        for word in re.findall(r"[A-Za-z']+", text):
            wl = word.lower()
            if wl in SEASON_WORDS:
                grains.setdefault('season', word)
            elif wl in MONTH_WORDS:
                grains.setdefault('month', word)
            elif wl in WEEKDAY_WORDS:
                grains.setdefault('day', word)
            elif wl in self.time_vocab:
                grains.setdefault('time_of_day', word)
        return grains
    # --- Main execution ---
    def execute(self):
        if self.active_input == self.set_context_input:
            data = self.set_context_input()
            if type(data) == list and len(data) >= 2:
                slot_name = str(data[0])
                value = str(data[1])
                if slot_name in self.place_slots:
                    # Set a specific ladder scale, with normal knockout rules
                    if self.apply_place_detections({slot_name: value}):
                        self.update_status()
                    self.emit_output()
                elif slot_name in self.time_slots:
                    # Covers plain 'era' and 'time_of_day' too — both are rungs
                    if self.apply_time_detections({slot_name: value}):
                        self.update_status()
                    self.emit_output()
                elif slot_name == 'time_of_year':
                    # Not itself a rung; store at season scale
                    if self.apply_time_detections({'season': value}):
                        self.update_status()
                    self.emit_output()
                elif slot_name == 'actors':
                    # Inject a named actor at the top of the roster
                    top = max((a['salience'] for a in self.actors), default=0.0)
                    self.apply_actor_mentions([{
                        'kind': 'name', 'name': value, 'display': value,
                        'role': max(1.0, top), 'number': 'sing',
                        'gender': None, 'human': None}])
                    self.update_status()
                    self.emit_output()
                elif slot_name == 'place':
                    # Plain 'place' replaces the whole ladder
                    for scale in PLACE_SCALES:
                        self.place_slots[scale] = ''
                        self.place_ages[scale] = 0
                    self.place_slots['site'] = value
                    self.compose_place()
                    self.detected_output.send(['place', self.slots['place']])
                    self.update_status()
                    self.emit_output()
                elif slot_name in self.slots:
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
            # Age all slots (place and time are aged per-scale below)
            ladder_managed = ('place', 'era', 'time_of_year', 'time_of_day', 'actors')
            for name in self.slot_names:
                if name not in ladder_managed and self.slots[name] != '':
                    self.slot_ages[name] += 1
            for scale in PLACE_SCALES:
                if self.place_slots[scale] != '':
                    self.place_ages[scale] += 1
            for scale in TIME_SCALES:
                if self.time_slots[scale] != '':
                    self.time_ages[scale] += 1
            # Optional decay — ladder scales decay slower the coarser they are
            decay = self.decay_chunks()
            if decay > 0:
                for name in self.slot_names:
                    if name not in ladder_managed and self.slot_ages[name] > decay:
                        self.slots[name] = ''
                        self.slot_ages[name] = 0
                expired = False
                for scale in PLACE_SCALES:
                    if (self.place_slots[scale] != ''
                            and self.place_ages[scale] > decay * PLACE_DECAY_MULTIPLIERS[scale]):
                        self.place_slots[scale] = ''
                        self.place_ages[scale] = 0
                        expired = True
                if expired:
                    self.compose_place()
                expired = False
                for scale in TIME_SCALES:
                    if (self.time_slots[scale] != ''
                            and self.time_ages[scale] > decay * TIME_DECAY_MULTIPLIERS[scale]):
                        self.time_slots[scale] = ''
                        self.time_ages[scale] = 0
                        expired = True
                if expired:
                    self.compose_time()
            # Actor salience decays every chunk — it's the core mechanic,
            # independent of the optional slot decay
            self.decay_actors()
            # --- Run extraction ---
            detections = self.extract_context(text)
            # Apply detections
            changed = False
            place_dets = detections.pop('place', None)
            if place_dets:
                if self.apply_place_detections(place_dets):
                    changed = True
            time_dets = detections.pop('time', None)
            if time_dets:
                if self.apply_time_detections(time_dets):
                    changed = True
            actor_mentions = detections.pop('actor_mentions', None)
            if actor_mentions:
                if self.apply_actor_mentions(actor_mentions):
                    changed = True
            for slot_name, value in detections.items():
                if value and value != self.slots[slot_name]:
                    self.slots[slot_name] = value
                    self.slot_ages[slot_name] = 0
                    self.detected_output.send([slot_name, value])
                    changed = True
            if changed:
                self.update_status()
            self.emit_output()
    # --- Actor roster ---
    def _strip_title(self, name):
        """Split a leading title off a name; titles are gender evidence."""
        words = name.split()
        if words:
            title = words[0].lower().rstrip('.')
            if title in TITLE_GENDER:
                rest = ' '.join(words[1:])
                return (rest if rest else name), TITLE_GENDER[title]
        return name, None
    def _find_actor_by_name(self, name):
        tokens = {w.lower() for w in name.split()}
        for entry in self.actors:
            if entry['name_tokens'] & tokens:
                return entry
        return None
    def _find_actor_by_lemma(self, lemma):
        for entry in self.actors:
            if lemma in entry['lemmas']:
                return entry
        return None
    def _resolve_pronoun(self, pron_class):
        """Most salient roster entry compatible with the pronoun's features.
        Entries with unknown features stay eligible — the resolution then
        teaches the feature. self.actors is kept sorted by salience."""
        for entry in self.actors:
            if pron_class == 'masc':
                if (entry['number'] == 'sing' and entry['human'] is not False
                        and entry['gender'] in (None, 'masc')):
                    return entry
            elif pron_class == 'fem':
                if (entry['number'] == 'sing' and entry['human'] is not False
                        and entry['gender'] in (None, 'fem')):
                    return entry
            elif pron_class == 'neut':
                if entry['human'] is not True and entry['number'] == 'sing':
                    return entry
            elif pron_class == 'plur':
                if entry['number'] == 'plur':
                    return entry
        return None
    def extract_actor_mentions(self, doc):
        """Collect actor mentions in document order. Pure — roster is only
        consulted (for the reinforce gate), never modified here."""
        mentions = []
        person_token_ents = {}
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                for t in ent:
                    person_token_ents[t.i] = ent
        seen_ents = set()
        for token in doc:
            ent = person_token_ents.get(token.i)
            if ent is not None:
                if ent.start in seen_ents:
                    continue
                seen_ents.add(ent.start)
                name, gender = self._strip_title(ent.text)
                mentions.append({
                    'kind': 'name', 'name': name, 'display': name,
                    'role': ACTOR_ROLE_WEIGHTS.get(ent.root.dep_, 0.3),
                    'number': 'sing', 'gender': gender, 'human': True})
                continue
            if token.pos_ == 'PRON':
                if token.dep_ == 'expl':
                    continue  # 'It was raining' / 'it seemed'
                pron_class = _PRONOUN_CLASSES.get(token.lower_)
                if pron_class is None:
                    continue  # first/second person etc.
                if (pron_class == 'neut'
                        and token.head.lemma_.lower() in self.weather_lemmas):
                    continue  # 'it rained'
                mentions.append({
                    'kind': 'pron', 'class': pron_class, 'text': token.lower_,
                    'role': ACTOR_ROLE_WEIGHTS.get(token.dep_, 0.3)})
                continue
            if (token.pos_ == 'PROPN' and token.ent_type_ == ''
                    and token.dep_ in ('nsubj', 'nsubjpass')):
                # Un-NER'd proper-noun subject still reads as a named actor
                mentions.append({
                    'kind': 'name', 'name': token.text, 'display': token.text,
                    'role': 1.0, 'number': 'sing', 'gender': None, 'human': None})
                continue
            if token.pos_ == 'NOUN' and token.ent_type_ == '':
                lemma = token.lemma_.lower()
                # Category vocab (places, media, time words) are registers of
                # their own, not actors
                if (self.noun_scale(lemma) is not None
                        or lemma in self.medium_vocab or lemma in self.time_vocab
                        or lemma in SEASON_WORDS or lemma in MONTH_WORDS
                        or lemma in WEEKDAY_WORDS):
                    continue
                dep = token.dep_
                if dep not in ACTOR_ROLE_WEIGHTS:
                    continue
                is_subject = dep in ('nsubj', 'nsubjpass')
                animate = lemma in ANIMATE_NOUNS
                in_roster = self._find_actor_by_lemma(lemma) is not None
                # Actorhood is functional: subjects always qualify; objects
                # only if animate or already known
                if not (is_subject or animate or in_roster):
                    continue
                descriptor = doc[token.left_edge.i:token.i + 1].text
                if animate and lemma not in ANIMAL_NOUNS:
                    human = True
                elif animate:
                    human = None  # animals take he/she or it
                else:
                    human = False  # storm, machine: 'it'-referable
                appos_name = None
                if dep == 'appos' and token.head.i in person_token_ents:
                    appos_name = person_token_ents[token.head.i].text
                mentions.append({
                    'kind': 'noun', 'lemma': lemma, 'display': descriptor,
                    'role': ACTOR_ROLE_WEIGHTS.get(dep, 0.3),
                    'number': 'plur' if 'Plur' in token.morph.get('Number') else 'sing',
                    'gender': GENDERED_NOUNS.get(lemma), 'human': human,
                    'appos_name': appos_name})
        return mentions
    def apply_actor_mentions(self, mentions):
        """Apply a chunk's mentions to the roster. Returns True if the
        composed actors register changed."""
        for m in mentions:
            if m['kind'] == 'pron':
                entry = self._resolve_pronoun(m['class'])
                if entry is None:
                    if m['class'] == 'plur':
                        # Group reference: 'they' over the top human actors
                        group = [a for a in self.actors
                                 if a['human'] is not False][:2]
                        for a in group:
                            a['salience'] += m['role'] * 0.5
                            a['age'] = 0
                    continue
                entry['salience'] += m['role']
                entry['age'] = 0
                # Resolution teaches features the text has now evidenced
                if m['class'] in ('masc', 'fem'):
                    if entry['gender'] is None:
                        entry['gender'] = m['class']
                    if entry['human'] is None:
                        entry['human'] = True
                elif m['class'] == 'neut' and entry['human'] is None:
                    entry['human'] = False
                self.detected_output.send(
                    ['actor.ref', f"{m['text']} -> {entry['display']}"])
                continue
            if m['kind'] == 'name':
                entry = self._find_actor_by_name(m['name'])
                if entry is not None:
                    entry['salience'] += m['role']
                    entry['age'] = 0
                    entry['name_tokens'] |= {w.lower() for w in m['name'].split()}
                    if m['gender'] is not None and entry['gender'] is None:
                        entry['gender'] = m['gender']
                    continue
            else:
                if m['appos_name'] is not None:
                    # 'Marguerite, the old woman' — descriptor of a named actor
                    entry = self._find_actor_by_name(m['appos_name'])
                    if entry is not None:
                        entry['lemmas'].add(m['lemma'])
                        entry['salience'] += m['role']
                        entry['age'] = 0
                        if m['gender'] is not None and entry['gender'] is None:
                            entry['gender'] = m['gender']
                        continue
                entry = self._find_actor_by_lemma(m['lemma'])
                if entry is not None:
                    entry['salience'] += m['role']
                    entry['age'] = 0
                    entry['display'] = m['display']
                    continue
            entry = {
                'name': m.get('name'),
                'name_tokens': {w.lower() for w in m['name'].split()}
                               if m.get('name') else set(),
                'lemmas': {m['lemma']} if m['kind'] == 'noun' else set(),
                'display': m['display'], 'number': m['number'],
                'gender': m['gender'], 'human': m['human'],
                'salience': m['role'], 'age': 0,
            }
            self.actors.append(entry)
            self.detected_output.send(['actor', m['display']])
        self.actors.sort(key=lambda a: a['salience'], reverse=True)
        del self.actors[self.max_actors():]
        old = self.slots['actors']
        self.compose_actors()
        return self.slots['actors'] != old
    def compose_actors(self):
        """Top of the roster, most salient first."""
        top = [a['display'] for a in self.actors if a['salience'] >= 0.1][:3]
        self.slots['actors'] = ', '.join(top)
    def decay_actors(self):
        """Geometric salience decay each chunk; evict the faded."""
        if not self.actors:
            return
        factor = self.actor_decay()
        for a in self.actors:
            a['salience'] *= factor
            a['age'] += 1
        self.actors = [a for a in self.actors if a['salience'] > 0.05]
        self.compose_actors()
    # --- Extraction engine ---
    def extract_context(self, text):
        """Extract context slots from a text chunk. Returns dict of detected
        values; the 'place' entry (if present) is a dict of {scale: phrase}."""
        detections = {}
        place_dets = {}  # scale -> phrase; first detection per scale wins
        time_dets = {}   # scale -> phrase; first detection per scale wins
        if self.__class__.nlp is None:
            return detections
        doc = self.__class__.nlp(text)
        # Reconstruct hyphenated compounds for weather/style matching
        hyphenated = reconstruct_hyphenated(doc)
        # --- 1. spaCy NER ---
        for ent in doc.ents:
            if ent.label_ in ('TIME',):
                time_dets.setdefault('time_of_day', get_ent_phrase(ent))
            elif ent.label_ == 'DATE':
                grains = self.date_ent_grains(ent.text)
                if grains:
                    # Full phrase lives at the finest grain; coarser grains
                    # get their bare words ('One summer evening' -> season
                    # 'summer', time_of_day 'One summer evening')
                    finest = max(grains, key=TIME_SCALES.index)
                    grains[finest] = get_ent_phrase(ent)
                    for grain, value in grains.items():
                        time_dets.setdefault(grain, value)
            elif ent.label_ in ('GPE', 'LOC', 'FAC'):
                # Full entity span with its governing preposition
                phrase = get_ent_phrase(ent)
                if ent.label_ == 'GPE':
                    scale = self.classify_gpe(ent.text)
                else:
                    scale = 'site'
                place_dets.setdefault(scale, phrase)
        # --- 2. Pattern matching ---
        text_lower = text.lower()
        # Time-grain patterns, one per ladder scale
        if 'time_of_day' not in time_dets:
            for pattern in _TIME_PATTERNS_COMPILED:
                match = pattern.search(text)
                if match:
                    time_dets['time_of_day'] = match.group(0).strip()
                    break
        for grain, pattern in (('century', _CENTURY_PATTERN),
                               ('decade', _DECADE_PATTERN),
                               ('year', _YEAR_PATTERN),
                               ('month', _MONTH_PATTERN),
                               ('season', _SEASON_PATTERN)):
            if grain not in time_dets:
                match = pattern.search(text)
                if match:
                    time_dets[grain] = match.group(0).strip()
        # Era patterns — period adjectives, then loadable named events, then
        # preposition-anchored single-word events
        if 'era' not in time_dets:
            for pattern in (_ERA_PATTERNS_COMPILED[0],
                            self._era_event_pattern,
                            _ERA_PATTERNS_COMPILED[1]):
                match = pattern.search(text)
                if match:
                    time_dets['era'] = match.group(0).strip()
                    break
        # Place patterns — collect all matches, one per scale
        for pattern in _PLACE_PATTERNS_COMPILED:
            for match in pattern.finditer(text):
                phrase = match.group(0).strip()
                scale = self.classify_place_phrase(phrase)
                place_dets.setdefault(scale, phrase)
        # Weather patterns
        weather_phrases = []
        for pattern in _WEATHER_PATTERNS_COMPILED:
            match = pattern.search(text)
            if match:
                weather_phrases.append(match.group(0).strip())
                break
        # --- 3. Keyword matching ---
        # Weather keywords — collect ALL matches in the chunk so conjoined,
        # comma-separated, or stacked specs ('cloudy and rainy', 'cloudy,
        # rainy day') all register, joined as a comma list.
        found = []
        tokens = list(doc)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # Hyphenated compound first, so 'rain-soaked' doesn't instead
            # register its bare 'rain' token
            if i + 2 < len(tokens) and tokens[i + 1].text == '-':
                compound = (token.text.lower() + '-'
                            + tokens[i + 2].text.lower())
                if compound in self.weather_vocab:
                    found.append(compound)
                    i += 3
                    continue
            if token.text.lower() in self.weather_vocab:
                found.append(token.text.lower())
            elif token.lemma_.lower() in self.weather_lemmas:
                # Weather verbs are impersonal ("it rained") — a verb with a
                # direct object is a different sense ("hail a taxi").
                if not (token.pos_ == 'VERB'
                        and any(c.dep_ == 'dobj' for c in token.children)):
                    found.append(token.text.lower())
            i += 1
        seen = set()
        for word in found:
            # Dedupe, and skip words a pattern phrase already contains
            if word in seen or any(word in p.lower() for p in weather_phrases):
                continue
            seen.add(word)
            weather_phrases.append(word)
        if weather_phrases:
            detections['weather'] = ', '.join(weather_phrases)
        # Style keywords (movements, moods, lighting)
        if 'style' not in detections:
            match = self._match_vocab(doc, text_lower, hyphenated, self.style_vocab)
            if match is not None:
                detections['style'] = match
        # Medium keywords ('oil painting', 'polaroid', 'postage stamp');
        # lemmas so plurals match ('polaroids'); modifiers of the medium
        # noun ride along ('black and white photograph')
        if 'medium' not in detections:
            match = self._match_vocab(doc, text_lower, hyphenated,
                                      self.medium_vocab, use_lemmas=True,
                                      expand_modifiers=True)
            if match is not None:
                detections['medium'] = match
        # --- 4. Time grains from keywords (if not caught by NER or patterns) ---
        if 'era' not in time_dets:
            for token in doc:
                if token.text.lower() in self.era_vocab:
                    # Try to get the surrounding phrase
                    phrase = get_prep_phrase_text(token)
                    time_dets['era'] = phrase
                    break
        for token in doc:
            token_lower = token.text.lower()
            if 'season' not in time_dets and token_lower in BARE_SEASON_WORDS:
                time_dets['season'] = get_prep_phrase_text(token)
            if 'day' not in time_dets and token_lower in WEEKDAY_WORDS:
                time_dets['day'] = get_prep_phrase_text(token)
            if 'time_of_day' not in time_dets and token_lower in self.time_vocab:
                time_dets['time_of_day'] = get_prep_phrase_text(token)
        if 'day' not in time_dets:
            for holiday in HOLIDAY_WORDS:
                match = re.search(r'\b' + re.escape(holiday) + r'\b', text, re.IGNORECASE)
                if match:
                    time_dets['day'] = match.group(0)
                    break
        # --- 5. Dependency-based place extraction ---
        # Look for place nouns that are objects of prepositions and capture
        # the full prep phrase. Scales already filled by NER/patterns are
        # skipped; different scales in one chunk are all collected.
        for token in doc:
            if token.ent_type_ in ('GPE', 'LOC', 'FAC'):
                # Already classified by NER ('Toronto City Hall' shouldn't
                # re-register as an interior via its 'hall' token)
                continue
            if token.dep_ == 'compound':
                # Handled at the compound head ('city' in 'city hall' is not
                # a settlement mention)
                continue
            # Classify the full compound first ('city hall' -> site), then
            # the bare token
            compound_parts = [c.text.lower() for c in token.children
                              if c.dep_ == 'compound']
            scale = None
            if compound_parts:
                scale = self.noun_scale(' '.join(compound_parts + [token.text.lower()]))
            if scale is None:
                scale = self.noun_scale(token.text.lower())
            if scale is None:
                continue
            if token.dep_ in ('pobj', 'dobj', 'attr', 'nsubj', 'nsubjpass'):
                # Walk up to find the governing preposition
                head = token.head
                if head.pos_ == 'ADP':
                    prep = head
                elif head.head.pos_ == 'ADP':
                    prep = head.head
                else:
                    prep = None
                if prep is not None:
                    # Climb noun-of-noun chains to the outermost preposition
                    # so 'at the edge of the forest' isn't clipped to
                    # 'of the forest'. Stop at nouns that are places in their
                    # own right — 'village in the mountains' must not fold
                    # the settlement into the site phrase
                    while (prep.head.pos_ in ('NOUN', 'PROPN')
                           and prep.head.head.pos_ == 'ADP'
                           and self.noun_scale(prep.head.text.lower()) is None):
                        prep = prep.head.head
                    phrase = get_full_prep_phrase_text(prep)
                else:
                    # No preposition, but the noun itself is a place
                    phrase = ' '.join(
                        [c.text for c in token.children if c.dep_ == 'compound']
                        + [token.text])
                existing = place_dets.get(scale)
                if existing is None:
                    place_dets[scale] = phrase
                elif (existing.lower() in phrase.lower()
                        and len(phrase) > len(existing)):
                    # The parse-derived phrase extends an earlier NER/pattern
                    # match at this scale ('on the shore' -> 'on the shore
                    # of the lake') — upgrade rather than skip
                    place_dets[scale] = phrase
        if place_dets:
            detections['place'] = place_dets
        if time_dets:
            detections['time'] = time_dets
        # --- 6. Actor mentions (named, nominal, pronominal) ---
        actor_mentions = self.extract_actor_mentions(doc)
        if actor_mentions:
            detections['actor_mentions'] = actor_mentions
        return detections
    # --- Output ---
    def emit_output(self):
        """Build and send the weighted prompt list."""
        strength = self.strength()
        prompt_list = []
        for slot_name in self.slot_names:
            if slot_name == 'actors':
                # Each actor gets its own entry, weight scaled by salience
                # relative to the most salient (soft valve, no threshold)
                max_salience = max((a['salience'] for a in self.actors), default=0.0)
                if max_salience > 0.0:
                    weight = self.actor_weight() * strength
                    for a in self.actors[:3]:
                        prompt_list.append(
                            [a['display'], weight * a['salience'] / max_salience])
                continue
            value = self.slots[slot_name]
            if value != '':
                weight_prop = self.weight_properties[slot_name]
                weight = weight_prop() * strength
                prompt_list.append([value, weight])
        self.context_output.send(prompt_list)
        # Also send the raw dict for debugging, with place ladder detail.
        # Unfilled slots are left out unless 'include empty keys' is set.
        include_empty = self.include_empty_keys()
        out = {name: value for name, value in self.slots.items()
               if include_empty or value != ''}
        for scale in PLACE_SCALES:
            if self.place_slots[scale] != '':
                out[f'place.{scale}'] = self.place_slots[scale]
        for scale in TIME_SCALES:
            if self.time_slots[scale] != '':
                out[f'time.{scale}'] = self.time_slots[scale]
        for i, a in enumerate(self.actors):
            features = [f for f in (a['gender'], a['number'],
                                    'human' if a['human'] else None) if f]
            out[f'actor.{i}'] = f"{a['display']} ({a['salience']:.2f} {' '.join(features)})"
        self.dict_output.send(out)
    # --- Status display ---
    def update_status(self):
        parts = []
        for name in self.slot_names:
            if self.slots[name] != '':
                short_name = (name.replace('time_of_day', 'time')
                              .replace('time_of_year', 'season')
                              .replace('weather', 'wx'))
                parts.append(f'{short_name}: {self.slots[name]}')
        status = ' | '.join(parts) if parts else '(no context)'
        self.status_label.set(status)