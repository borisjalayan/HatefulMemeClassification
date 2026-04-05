"""OCR text extraction from meme images.

Strategy:
  1. JSONL lookup — if the uploaded image matches a known dataset image
     (by filename), return the ground-truth text from train/dev/test.jsonl.
  2. docTR meme extraction — morphological mask isolates white-on-dark
     meme text, then docTR reads the cleaned image.

All outputs are post-processed through a correction dictionary
that fixes capitalisation and common OCR misreads.
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)

# ─── Singletons ──────────────────────────────────────────────────────────────
_doctr_model = None
_jsonl_text_index: dict[str, str] | None = None  # filename → text

# ─── Image preprocessing settings ────────────────────────────────────────────
_MIN_WORD_LENGTH = 2

# ─── Default meme mask parameters (tuned) ────────────────────────────────────
_WHITE_THRESH = 180
_BORDER_THRESH = 60
_DILATE_ITER = 4
_KERNEL_SIZE = 3


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECTION DICTIONARY — 700+ words
#
# Maps lowercase → correct form.  Covers:
#   • Proper nouns (religious, ethnic, political figures & groups)
#   • Country / region names
#   • Acronyms & abbreviations
#   • Common OCR misreads
# ═══════════════════════════════════════════════════════════════════════════════
CORRECTIONS: dict[str, str] = {
    # ── Religious figures & terms ─────────────────────────────────────────
    "jesus": "Jesus", "christ": "Christ", "god": "God", "allah": "Allah",
    "muhammad": "Muhammad", "mohammed": "Mohammed", "moses": "Moses",
    "buddha": "Buddha", "satan": "Satan", "lucifer": "Lucifer",
    "bible": "Bible", "quran": "Quran", "koran": "Koran", "torah": "Torah",
    "genesis": "Genesis", "exodus": "Exodus", "gospel": "Gospel",
    "psalm": "Psalm", "psalms": "Psalms", "revelation": "Revelation",
    "christian": "Christian", "christians": "Christians",
    "christianity": "Christianity", "christmas": "Christmas",
    "catholic": "Catholic", "catholics": "Catholics", "catholicism": "Catholicism",
    "protestant": "Protestant", "protestants": "Protestants",
    "baptist": "Baptist", "baptists": "Baptists",
    "methodist": "Methodist", "lutheran": "Lutheran",
    "orthodox": "Orthodox", "evangelical": "Evangelical",
    "muslim": "Muslim", "muslims": "Muslims",
    "islam": "Islam", "islamic": "Islamic", "islamist": "Islamist",
    "islamophobia": "Islamophobia", "islamophobic": "Islamophobic",
    "islamaphobia": "Islamaphobia", "islamaphobic": "Islamaphobic",
    "jew": "Jew", "jews": "Jews", "jewish": "Jewish",
    "judaism": "Judaism", "antisemitic": "antisemitic",
    "antisemitism": "antisemitism",
    "hindu": "Hindu", "hindus": "Hindus", "hinduism": "Hinduism",
    "sikh": "Sikh", "sikhs": "Sikhs", "sikhism": "Sikhism",
    "atheist": "Atheist", "atheists": "Atheists", "atheism": "Atheism",
    "agnostic": "Agnostic",
    "sharia": "Sharia", "halal": "Halal", "haram": "Haram",
    "jihad": "Jihad", "jihadist": "Jihadist", "jihadists": "Jihadists",
    "allahu": "Allahu", "akbar": "Akbar",
    "ramadan": "Ramadan", "eid": "Eid", "hajj": "Hajj",
    "kosher": "Kosher", "rabbi": "Rabbi", "imam": "Imam",
    "pope": "Pope", "vatican": "Vatican",
    "church": "Church", "mosque": "Mosque", "synagogue": "Synagogue",
    "temple": "Temple",
    "heaven": "Heaven", "hell": "Hell", "purgatory": "Purgatory",
    "angel": "Angel", "angels": "Angels",
    "apostle": "Apostle", "apostles": "Apostles",
    "pharisee": "Pharisee", "pharisees": "Pharisees",
    "noah": "Noah", "abraham": "Abraham", "isaac": "Isaac",
    "jacob": "Jacob", "david": "David", "solomon": "Solomon",
    "mary": "Mary", "joseph": "Joseph", "paul": "Paul", "peter": "Peter",
    "john": "John", "matthew": "Matthew", "mark": "Mark", "luke": "Luke",
    "adam": "Adam", "eve": "Eve",

    # ── Political figures ─────────────────────────────────────────────────
    "trump": "Trump", "trump's": "Trump's",
    "donald": "Donald", "donaldtrump": "DonaldTrump",
    "obama": "Obama", "obama's": "Obama's", "obamacare": "Obamacare",
    "barack": "Barack",
    "biden": "Biden", "biden's": "Biden's",
    "hillary": "Hillary", "hillary's": "Hillary's",
    "clinton": "Clinton", "clinton's": "Clinton's",
    "bernie": "Bernie", "sanders": "Sanders",
    "pelosi": "Pelosi", "pence": "Pence", "kamala": "Kamala",
    "harris": "Harris", "mcconnell": "McConnell",
    "aoc": "AOC", "ocasio-cortez": "Ocasio-Cortez",
    "putin": "Putin", "putin's": "Putin's",
    "merkel": "Merkel", "macron": "Macron", "trudeau": "Trudeau",
    "netanyahu": "Netanyahu", "erdogan": "Erdogan",
    "kim": "Kim", "jong": "Jong",
    "gandhi": "Gandhi", "mandela": "Mandela",
    "mlk": "MLK", "luther": "Luther",
    "kennedy": "Kennedy", "jfk": "JFK",
    "lincoln": "Lincoln", "washington": "Washington",
    "reagan": "Reagan", "bush": "Bush",
    "hitler": "Hitler", "hitler's": "Hitler's",
    "nazi": "Nazi", "nazis": "Nazis", "nazism": "Nazism",
    "mussolini": "Mussolini", "stalin": "Stalin",
    "isis": "ISIS", "al-qaeda": "Al-Qaeda", "taliban": "Taliban",
    "hamas": "Hamas", "hezbollah": "Hezbollah",
    "kkk": "KKK", "klan": "Klan",

    # ── Political parties & movements ─────────────────────────────────────
    "democrat": "Democrat", "democrats": "Democrats",
    "democratic": "Democratic",
    "republican": "Republican", "republicans": "Republicans",
    "liberal": "Liberal", "liberals": "Liberals",
    "liberalism": "Liberalism",
    "conservative": "Conservative", "conservatives": "Conservatives",
    "conservatism": "Conservatism",
    "libertarian": "Libertarian", "libertarians": "Libertarians",
    "socialist": "Socialist", "socialists": "Socialists",
    "socialism": "Socialism",
    "communist": "Communist", "communists": "Communists",
    "communism": "Communism",
    "marxist": "Marxist", "marxism": "Marxism",
    "fascist": "Fascist", "fascism": "Fascism",
    "antifa": "Antifa",
    "maga": "MAGA",
    "blm": "BLM",
    "nra": "NRA",
    "gop": "GOP",
    "un": "UN",
    "nato": "NATO",
    "eu": "EU",
    "fbi": "FBI", "cia": "CIA", "nsa": "NSA",
    "dhs": "DHS", "ice": "ICE",
    "aclu": "ACLU",

    # ── Ethnicities, races, nationalities ─────────────────────────────────
    "african": "African", "africans": "Africans",
    "african-american": "African-American", "african-americans": "African-Americans",
    "asian": "Asian", "asians": "Asians",
    "caucasian": "Caucasian", "caucasians": "Caucasians",
    "hispanic": "Hispanic", "hispanics": "Hispanics",
    "latino": "Latino", "latina": "Latina", "latinos": "Latinos",
    "arab": "Arab", "arabs": "Arabs", "arabic": "Arabic",
    "persian": "Persian", "persians": "Persians",
    "indian": "Indian", "indians": "Indians",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean": "Korean", "koreans": "Koreans",
    "vietnamese": "Vietnamese",
    "filipino": "Filipino", "filipinos": "Filipinos",
    "mexican": "Mexican", "mexicans": "Mexicans",
    "cuban": "Cuban", "cubans": "Cubans",
    "puerto rican": "Puerto Rican",
    "brazilian": "Brazilian",
    "colombian": "Colombian",
    "haitian": "Haitian", "haitians": "Haitians",
    "jamaican": "Jamaican",
    "european": "European", "europeans": "Europeans",
    "british": "British", "english": "English",
    "french": "French", "german": "German", "germans": "Germans",
    "italian": "Italian", "italians": "Italians",
    "spanish": "Spanish", "portuguese": "Portuguese",
    "russian": "Russian", "russians": "Russians",
    "polish": "Polish", "irish": "Irish", "scottish": "Scottish",
    "swedish": "Swedish", "norwegian": "Norwegian",
    "dutch": "Dutch", "belgian": "Belgian",
    "greek": "Greek", "turkish": "Turkish",
    "israeli": "Israeli", "israelis": "Israelis",
    "palestinian": "Palestinian", "palestinians": "Palestinians",
    "syrian": "Syrian", "syrians": "Syrians",
    "iraqi": "Iraqi", "iranian": "Iranian",
    "afghan": "Afghan", "afghans": "Afghans",
    "pakistani": "Pakistani", "pakistanis": "Pakistanis",
    "somali": "Somali", "somalis": "Somalis",
    "nigerian": "Nigerian",
    "egyptian": "Egyptian",
    "native american": "Native American",
    "indigenous": "Indigenous",
    "aboriginal": "Aboriginal",
    "romani": "Romani",
    "gypsy": "Gypsy", "gypsies": "Gypsies",

    # ── Countries & regions ───────────────────────────────────────────────
    "america": "America", "america's": "America's",
    "american": "American", "americans": "Americans",
    "usa": "USA", "us": "US",
    "united states": "United States",
    "canada": "Canada", "canadian": "Canadian",
    "mexico": "Mexico",
    "brazil": "Brazil",
    "argentina": "Argentina",
    "europe": "Europe",
    "england": "England", "britain": "Britain",
    "uk": "UK",
    "france": "France",
    "germany": "Germany",
    "italy": "Italy",
    "spain": "Spain",
    "russia": "Russia",
    "china": "China",
    "japan": "Japan",
    "korea": "Korea",
    "north korea": "North Korea",
    "south korea": "South Korea",
    "india": "India",
    "pakistan": "Pakistan",
    "afghanistan": "Afghanistan",
    "iran": "Iran",
    "iraq": "Iraq",
    "syria": "Syria",
    "israel": "Israel", "israel's": "Israel's",
    "palestine": "Palestine",
    "saudi": "Saudi",
    "saudi arabia": "Saudi Arabia",
    "egypt": "Egypt",
    "turkey": "Turkey",
    "africa": "Africa",
    "nigeria": "Nigeria",
    "somalia": "Somalia",
    "yemen": "Yemen",
    "libya": "Libya",
    "sudan": "Sudan",
    "australia": "Australia",
    "new zealand": "New Zealand",
    "middle east": "Middle East",
    "southeast asia": "Southeast Asia",
    "latin america": "Latin America",
    "texas": "Texas", "california": "California",
    "florida": "Florida", "new york": "New York",
    "alabama": "Alabama", "mississippi": "Mississippi",
    "michigan": "Michigan", "ohio": "Ohio",
    "georgia": "Georgia", "virginia": "Virginia",
    "chicago": "Chicago", "detroit": "Detroit",
    "portland": "Portland", "seattle": "Seattle",
    "jerusalem": "Jerusalem", "mecca": "Mecca",
    "medina": "Medina", "bethlehem": "Bethlehem",
    "auschwitz": "Auschwitz", "hiroshima": "Hiroshima",
    "nagasaki": "Nagasaki", "pearl harbor": "Pearl Harbor",

    # ── LGBTQ+ terms ─────────────────────────────────────────────────────
    "lgbt": "LGBT", "lgbtq": "LGBTQ", "lgbtq+": "LGBTQ+",
    "gay": "gay", "gays": "gays",
    "lesbian": "lesbian", "lesbians": "lesbians",
    "bisexual": "bisexual",
    "transgender": "transgender", "transgendered": "transgendered",
    "transgenderism": "transgenderism",
    "trans": "trans", "transsexual": "transsexual",
    "transphobic": "transphobic", "transphobia": "transphobia",
    "tranny": "tranny",
    "queer": "queer",
    "nonbinary": "nonbinary", "non-binary": "non-binary",
    "homophobia": "homophobia", "homophobic": "homophobic",
    "pride": "Pride",

    # ── Historical events & terms ─────────────────────────────────────────
    "holocaust": "Holocaust",
    "9/11": "9/11", "september 11": "September 11",
    "world war": "World War",
    "civil war": "Civil War",
    "cold war": "Cold War",
    "crusade": "Crusade", "crusades": "Crusades",
    "confederate": "Confederate", "confederacy": "Confederacy",
    "slavery": "slavery", "slave": "slave", "slaves": "slaves",
    "genocide": "genocide",
    "apartheid": "Apartheid",
    "jim crow": "Jim Crow",
    "civil rights": "Civil Rights",
    "emancipation": "Emancipation",
    "reconstruction": "Reconstruction",

    # ── Social media / internet culture ───────────────────────────────────
    "facebook": "Facebook", "twitter": "Twitter",
    "instagram": "Instagram", "tiktok": "TikTok",
    "reddit": "Reddit", "youtube": "YouTube",
    "google": "Google", "snapchat": "Snapchat",
    "whatsapp": "WhatsApp", "telegram": "Telegram",
    "tumblr": "Tumblr", "pinterest": "Pinterest",
    "cnn": "CNN", "fox": "Fox", "msnbc": "MSNBC",
    "bbc": "BBC", "nbc": "NBC", "abc": "ABC", "cbs": "CBS",
    "breitbart": "Breitbart", "infowars": "InfoWars",
    "snopes": "Snopes",
    "meme": "meme", "memes": "memes",
    "karen": "Karen", "karens": "Karens",
    "boomer": "Boomer", "boomers": "Boomers",
    "millennial": "Millennial", "millennials": "Millennials",
    "gen z": "Gen Z", "gen x": "Gen X",
    "zoomer": "Zoomer", "zoomers": "Zoomers",
    "simp": "simp", "incel": "incel", "incels": "incels",
    "chad": "Chad",
    "pepe": "Pepe",
    "sjw": "SJW", "sjws": "SJWs",
    "pc": "PC", "woke": "woke",
    "snowflake": "snowflake", "snowflakes": "snowflakes",
    "triggered": "triggered",
    "based": "based", "cringe": "cringe", "cope": "cope",
    "redpill": "redpill", "bluepill": "bluepill",

    # ── Misc proper nouns & brands ────────────────────────────────────────
    "disney": "Disney", "marvel": "Marvel", "dc": "DC",
    "netflix": "Netflix", "hbo": "HBO", "amazon": "Amazon",
    "walmart": "Walmart", "mcdonald's": "McDonald's",
    "starbucks": "Starbucks", "nike": "Nike",
    "tesla": "Tesla", "apple": "Apple", "microsoft": "Microsoft",
    "nfl": "NFL", "nba": "NBA", "mlb": "MLB", "fifa": "FIFA",
    "espn": "ESPN",
    "nasa": "NASA",
    "covid": "COVID", "covid-19": "COVID-19",
    "coronavirus": "Coronavirus",
    "aids": "AIDS", "hiv": "HIV",
    "ptsd": "PTSD", "adhd": "ADHD",
    "dna": "DNA", "iq": "IQ",
    "mph": "MPH", "lbs": "lbs",
    "am": "AM", "pm": "PM",
    "ok": "OK", "lol": "LOL", "lmao": "LMAO",
    "omg": "OMG", "wtf": "WTF", "smh": "SMH",
    "stfu": "STFU", "af": "AF",
    "diy": "DIY", "fyi": "FYI", "asap": "ASAP",
    "rip": "RIP", "aka": "AKA",
    "vs": "vs", "etc": "etc",

    # ── Days, months ──────────────────────────────────────────────────────
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
    "thursday": "Thursday", "friday": "Friday",
    "saturday": "Saturday", "sunday": "Sunday",
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June",
    "july": "July", "august": "August", "september": "September",
    "october": "October", "november": "November", "december": "December",

    # ── Common OCR misreads ───────────────────────────────────────────────
    "l": "I", "ll": "ll",
    "0": "O",
    "rn": "m",
    "vv": "w",
    "teh": "the", "hte": "the",
    "adn": "and", "nad": "and",
    "taht": "that", "tath": "that",
    "thier": "their", "theit": "their",
    "wiht": "with", "witht": "with",
    "becuase": "because", "becasue": "because",
    "beacuse": "because", "becouse": "because",
    "eveyrone": "everyone", "evryone": "everyone",
    "eveyone": "everyone", "evreyone": "everyone",
    "peolpe": "people", "pepole": "people", "poeple": "people",
    "shoudl": "should", "shuold": "should",
    "woudl": "would", "wuold": "would",
    "cuold": "could", "coudl": "could",
    "doesnt": "doesn't", "dont": "don't", "didnt": "didn't",
    "isnt": "isn't", "wasnt": "wasn't", "werent": "weren't",
    "havent": "haven't", "hasnt": "hasn't",
    "wont": "won't", "wouldnt": "wouldn't",
    "shouldnt": "shouldn't", "couldnt": "couldn't",
    "cant": "can't", "aint": "ain't",
    "youre": "you're", "theyre": "they're",
    "im": "I'm", "ive": "I've",
    "hes": "he's", "shes": "she's",
    "thats": "that's", "whats": "what's",
    "lets": "let's", "whos": "who's",
    "theres": "there's", "heres": "here's",
    "n0t": "not",
    "soicial": "social", "socail": "social",
    "politcal": "political", "politacal": "political",
    "goverment": "government", "govenment": "government",
    "goverrnment": "government",
    "terroism": "terrorism", "terorrism": "terrorism",
    "terroist": "terrorist", "terrosit": "terrorist",
    "religon": "religion", "relgion": "religion",
    "immgrant": "immigrant", "immigant": "immigrant",
    "immgrants": "immigrants", "immigants": "immigrants",
    "ilegal": "illegal", "illegall": "illegal",
    "boarder": "border", "boarders": "borders",
    "refguees": "refugees", "refguee": "refugee",
    "violance": "violence", "voilence": "violence",
    "rasict": "racist", "rascist": "racist",
    "racisim": "racism", "rascism": "racism",
    "prvilege": "privilege", "privelege": "privilege",
    "discrimnation": "discrimination",
    "seperate": "separate", "seprate": "separate",
    "definately": "definitely", "definatly": "definitely",
    "necesary": "necessary", "neccessary": "necessary",
    "recieve": "receive", "recieves": "receives",
    "beleive": "believe", "beleives": "believes",
    "occured": "occurred", "occuring": "occurring",
    "begining": "beginning", "begginning": "beginning",
    "geneocide": "genocide",
    "holocuast": "Holocaust", "holacaust": "Holocaust",
    "christain": "Christian", "christains": "Christians",
    "mulsim": "Muslim", "mulsims": "Muslims",
    "muslem": "Muslim", "muslems": "Muslims",
    "muhammed": "Muhammad", "mohamad": "Mohammad",
    "isreal": "Israel", "isreali": "Israeli",
    "palastine": "Palestine", "palistine": "Palestine",
    "palistinian": "Palestinian", "palastinian": "Palestinian",
    "palistinians": "Palestinians", "palastinians": "Palestinians",
    "afgahnistan": "Afghanistan", "afganistan": "Afghanistan",
    "pakisan": "Pakistan",
    "amerca": "America", "amercia": "America",
    "amercian": "American", "amercians": "Americans",
    "europian": "European", "europians": "Europeans",

    # ── Hateful meme vocabulary (from dataset) ────────────────────────────
    "dishwasher": "dishwasher",
    "goat": "goat", "goats": "goats",
    "bomb": "bomb", "bombs": "bombs", "bomber": "bomber",
    "terrorist": "terrorist", "terrorists": "terrorists",
    "terrorism": "terrorism",
    "refugee": "refugee", "refugees": "refugees",
    "immigrant": "immigrant", "immigrants": "immigrants",
    "immigration": "immigration",
    "deportation": "deportation", "deport": "deport",
    "deported": "deported",
    "border": "border", "borders": "borders",
    "wall": "wall",
    "welfare": "welfare",
    "ghetto": "ghetto",
    "thug": "thug", "thugs": "thugs",
    "gangster": "gangster", "gangsters": "gangsters",
    "criminal": "criminal", "criminals": "criminals",
    "rape": "rape", "rapist": "rapist",
    "pedophile": "pedophile",
    "retard": "retard", "retarded": "retarded",
    "cripple": "cripple", "crippled": "crippled",
    "disabled": "disabled", "disability": "disability",
    "mental": "mental", "mentally": "mentally",
    "autistic": "autistic", "autism": "autism",
    "privilege": "privilege", "privileged": "privileged",
    "oppression": "oppression", "oppressed": "oppressed",
    "supremacy": "supremacy", "supremacist": "supremacist",
    "supremacists": "supremacists",
    "patriarchy": "patriarchy",
    "feminist": "feminist", "feminists": "feminists",
    "feminism": "feminism",
    "misogyny": "misogyny", "misogynist": "misogynist",
    "sexist": "sexist", "sexism": "sexism",
    "bigot": "bigot", "bigots": "bigots", "bigotry": "bigotry",
    "xenophobia": "xenophobia", "xenophobic": "xenophobic",
    "propaganda": "propaganda",
    "conspiracy": "conspiracy",
    "extremist": "extremist", "extremists": "extremists",
    "radicalize": "radicalize", "radicalized": "radicalized",
    "stereotype": "stereotype", "stereotypes": "stereotypes",
    "segregation": "segregation",
    "plantation": "plantation",
    "colonialism": "colonialism", "colonizer": "colonizer",
    "reparations": "reparations",
    "affirmative": "affirmative",
    "diversity": "diversity",
    "inclusion": "inclusion",
    "equity": "equity",
    "equality": "equality",
    "intersectionality": "intersectionality",
    "microaggression": "microaggression",
    "trigger": "trigger", "triggers": "triggers",
    "safe space": "safe space",
    "hate speech": "hate speech",
    "hate crime": "hate crime",
    "free speech": "free speech",
    "amendment": "Amendment",
    "constitution": "Constitution",
    "constitutional": "Constitutional",
    "bill of rights": "Bill of Rights",
    "second amendment": "Second Amendment",
    "first amendment": "First Amendment",
}

# Build a fast lookup (lowercase key → corrected form)
_CORRECTIONS_LOWER: dict[str, str] = {k.lower(): v for k, v in CORRECTIONS.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. JSONL text lookup (instant — no OCR needed for known dataset images)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_jsonl_index(data_dir: str = "data") -> dict[str, str]:
    """Build a filename → text index from all JSONL splits.

    Loads train.jsonl, dev.jsonl, and test.jsonl.  Keys are bare
    filenames (e.g. "01235.png") so we can match uploaded images
    regardless of directory prefix.
    """
    global _jsonl_text_index
    if _jsonl_text_index is not None:
        return _jsonl_text_index

    index: dict[str, str] = {}
    data_path = Path(data_dir)

    for split_file in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        jsonl_path = data_path / split_file
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "").strip()
                img_field = record.get("img", "")
                if text and img_field:
                    # Store by bare filename: "img/01235.png" → "01235.png"
                    filename = Path(img_field).name
                    index[filename] = text

    _jsonl_text_index = index
    logger.info("JSONL text index built: %d entries from %s", len(index), data_dir)
    return index


def _try_jsonl_lookup(image_path: str) -> str:
    """Check if the image matches a known dataset image by filename.

    Args:
        image_path: Path to the uploaded image.

    Returns:
        Ground-truth text if found, empty string otherwise.
    """
    index = _build_jsonl_index()
    filename = Path(image_path).name

    if filename in index:
        text = index[filename]
        logger.info("JSONL lookup hit for %s: %s", filename, text[:120])
        return text

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. docTR meme text extraction (for unknown / new images)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_doctr_model():
    """Load docTR OCR model (singleton)."""
    global _doctr_model
    if _doctr_model is None:
        from doctr.models import ocr_predictor
        logger.info("Loading docTR OCR model (one-time setup)...")
        _doctr_model = ocr_predictor(pretrained=True)
        logger.info("docTR model ready")
    return _doctr_model


def _is_real_text(text: str) -> bool:
    """Heuristic: is this likely real text or OCR noise?"""
    if len(text) < _MIN_WORD_LENGTH:
        return False
    chars = text.replace(" ", "")
    if not chars:
        return False
    letter_ratio = sum(c.isalpha() for c in chars) / len(chars)
    return letter_ratio >= 0.5


def _try_doctr_meme(image_path: str) -> str:
    """Extract meme text using morphological masking + docTR.

    Creates a black-text-on-white-background image by detecting white
    text regions near dark borders (typical meme format), then runs
    docTR on the cleaned image.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text or empty string on failure.
    """
    try:
        from doctr.io import DocumentFile

        model = _get_doctr_model()

        img = cv2.imread(image_path)
        if img is None:
            logger.warning("docTR: could not read image %s", image_path)
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Build meme text mask: white regions near dark borders
        white_mask = gray > _WHITE_THRESH
        black_mask = gray < _BORDER_THRESH

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (_KERNEL_SIZE, _KERNEL_SIZE)
        )
        black_dilated = cv2.dilate(
            black_mask.astype(np.uint8), kernel, iterations=_DILATE_ITER
        )

        meme_text_mask = (white_mask & black_dilated.astype(bool)).astype(np.uint8) * 255
        meme_text_mask = cv2.morphologyEx(meme_text_mask, cv2.MORPH_CLOSE, kernel)
        meme_text_mask = cv2.morphologyEx(meme_text_mask, cv2.MORPH_OPEN, kernel)

        # Create clean OCR input: black text on white background
        masked = np.full_like(img, 255)
        masked[meme_text_mask > 0] = [0, 0, 0]

        # Write to temp file for docTR
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, masked)

        try:
            doc = DocumentFile.from_images(tmp.name)
            result = model(doc)
            ocr_text = result.render().strip()
        finally:
            os.unlink(tmp.name)

        if ocr_text and _is_real_text(ocr_text):
            logger.info("docTR meme extraction: %s", ocr_text[:120])
            return ocr_text

        # If masked approach got nothing, try docTR on the original image
        logger.debug("docTR mask extraction empty, trying original image")
        doc = DocumentFile.from_images(image_path)
        result = model(doc)
        ocr_text = result.render().strip()

        if ocr_text and _is_real_text(ocr_text):
            logger.info("docTR direct extraction: %s", ocr_text[:120])
            return ocr_text

        return ""

    except ImportError:
        logger.debug("python-doctr not installed")
        return ""
    except Exception as e:
        logger.warning("docTR error: %s", e)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing: correction dictionary
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_corrections(text: str) -> str:
    """Apply the correction dictionary to fix capitalisation and OCR misreads.

    Processes word by word, preserving surrounding punctuation.

    Args:
        text: Raw OCR text.

    Returns:
        Corrected text.
    """
    words = text.split()
    corrected: list[str] = []

    for word in words:
        # Strip punctuation to match dictionary, then reattach
        leading = ""
        trailing = ""
        core = word

        # Peel off leading punctuation
        while core and not core[0].isalnum():
            leading += core[0]
            core = core[1:]

        # Peel off trailing punctuation
        while core and not core[-1].isalnum():
            trailing = core[-1] + trailing
            core = core[:-1]

        if not core:
            corrected.append(word)
            continue

        lookup = core.lower()
        if lookup in _CORRECTIONS_LOWER:
            corrected.append(leading + _CORRECTIONS_LOWER[lookup] + trailing)
        else:
            corrected.append(word)

    result = " ".join(corrected)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(image_path: str) -> str:
    """Extract text from a meme image.

    Strategy:
        1. JSONL lookup — if the image filename matches a dataset entry,
           return the ground-truth text instantly (no OCR needed).
        2. docTR meme extraction — morphological mask + docTR OCR.

    All outputs go through the correction dictionary.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted and corrected text, or empty string on failure.
    """
    # 1. Try JSONL lookup first (instant, perfect text for known images)
    text = _try_jsonl_lookup(image_path)
    if text:
        # JSONL text is already clean — still apply corrections for consistency
        corrected = _apply_corrections(text)
        return corrected

    # 2. docTR meme extraction for unknown images
    text = _try_doctr_meme(image_path)

    if not text:
        logger.warning("All OCR methods failed for %s", image_path)
        return ""

    # Apply corrections
    corrected = _apply_corrections(text)
    if corrected != text:
        logger.info("Corrections applied: %r → %r", text[:80], corrected[:80])

    return corrected
