import re
from langdetect import detect
from sentence_splitter import SentenceSplitter

def clean_text(text):
    clean_text = []
    text = text.strip()
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line:
            line = re.sub(r'\s+', ' ', line)
            clean_text.append(line)
    return "\n".join(clean_text)
    
def detect_lang(text):
    """Detect language using langdetect (offline, stable)."""
    max_len = 500
    chunk = text[0 : min(max_len, len(text))]
    lang = detect(chunk)
    if lang.startswith('zh'):
        lang = 'zh'
    return lang

def split_sents(text, lang):
    if lang in LANG.SPLITTER:
        if lang == 'zh':
            sents = _split_zh(text)
        elif lang == 'vi':
            sents = _split_vi(text)
        else:
            # Split từng dòng riêng biệt rồi mới gọi SentenceSplitter
            # → giữ nguyên ranh giới dòng (chapter header, etc.)
            splitter = SentenceSplitter(language=lang)
            sents = []
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Nếu dòng là chapter header → giữ nguyên, không split thêm
                if _is_en_chapter_number(line):
                    sents.append(line)
                else:
                    line_sents = splitter.split(text=line)
                    sents.extend([s.strip() for s in line_sents if s.strip()])
        return sents
    else:
        raise Exception('The language {} is not suppored yet.'.format(LANG.ISO[lang]))


# --- Chapter header detection ---
# English number words used as chapter headers in Journey to the West
_EN_NUMBER_WORDS = {
    'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN',
    'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN',
    'EIGHTEEN', 'NINETEEN', 'TWENTY', 'THIRTY', 'FORTY', 'FIFTY', 'SIXTY',
    'SEVENTY', 'EIGHTY', 'NINETY', 'HUNDRED',
}

def _is_en_chapter_number(line):
    """Check if a line is an English chapter number like 'FIFTY-TWO', 'ONE', etc."""
    line = line.strip()
    if not line:
        return False
    # Split on hyphens and spaces: "FIFTY-TWO" → ["FIFTY", "TWO"]
    parts = re.split(r'[-\s]+', line)
    return all(p in _EN_NUMBER_WORDS for p in parts) and len(parts) <= 4

def _is_vi_chapter_header(line):
    """Check if a line is a Vietnamese chapter header like 'HỒI THỨ NĂM MƯƠI HAI'."""
    line = line.strip().upper()
    return line.startswith('HỒI THỨ') or re.match(r'^HỒI\s+\d+', line) is not None

def _normalize_chapter_headers(text, lang):
    """Add sentence-ending punctuation to chapter headers so SentenceSplitter
    treats them as standalone sentences instead of merging with adjacent text.
    
    Before: '...next chapter.\\nFIFTY-TWO\\nWukong greatly disturbed...'
    After:  '...next chapter.\\nFIFTY-TWO.\\nWukong greatly disturbed...'
    """
    lines = text.split('\n')
    normalized = []
    
    for line in lines:
        stripped = line.strip()
        
        if lang == 'en' and _is_en_chapter_number(stripped):
            # "FIFTY-TWO" → "FIFTY-TWO."
            if not stripped.endswith('.'):
                line = stripped + '.'
        elif lang == 'vi' and _is_vi_chapter_header(stripped):
            # "HỒI THỨ NĂM MƯƠI HAI" → "HỒI THỨ NĂM MƯƠI HAI."
            if not stripped.endswith('.'):
                line = stripped + '.'
        
        normalized.append(line)
    
    return '\n'.join(normalized)
    
    
def _split_zh(text, limit=1000):
        sent_list = []
        text = re.sub('(?P<quotation_mark>([。？！](?![”’"\'）])))', r'\g<quotation_mark>\n', text)
        text = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'）])', r'\g<quotation_mark>\n', text)

        sent_list_ori = text.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)

        return sent_list

def _split_vi(text, limit=1000):
    """Vietnamese sentence splitting with dialogue-aware merging.
    
    Uses underthesea for proper Vietnamese NLP sentence splitting,
    then applies heuristics to merge dialogue tags and noise fragments.
    Falls back to regex splitting if underthesea is not available.
    """
    try:
        from underthesea import sent_tokenize
        raw_sents = sent_tokenize(text)
    except ImportError:
        # Fallback: regex-based splitting (original logic)
        raw_sents = _split_vi_regex(text, limit)
    
    raw_sents = [s.strip() for s in raw_sents if s.strip()]
    merged = _merge_short_fragments(raw_sents, min_len=15)
    
    final = []
    for sent in merged:
        while len(sent) > limit:
            final.append(sent[:limit])
            sent = sent[limit:]
        if sent:
            final.append(sent)
    return final

def _split_vi_regex(text, limit=1000):
    sent_list = []
    text = re.sub('(?P<quotation_mark>([.?!](?![”’"\'）])))', r'\g<quotation_mark>\n', text)
    text = re.sub('(?P<quotation_mark>([.?!]|…{1,2})[”’"\'）])', r'\g<quotation_mark>\n', text)

    sent_list_ori = text.splitlines()
    for sent in sent_list_ori:
        sent = sent.strip()
        if not sent:
            continue
        while len(sent) > limit:
            sent_list.append(sent[:limit])
            sent = sent[limit:]
        sent_list.append(sent)
    return sent_list

_DIALOGUE_TAG_RE = re.compile(
    r'^[\w\s]{2,25}\s*(nói|hỏi|đáp|thưa|cười|bảo|rằng|quát|kêu|gọi|mắng|la|than|khóc|đọc|ngâm|xướng)\s*[:.]?\s*$',
    re.UNICODE
)
_NOISE_RE = re.compile(r'^[\s\.\,\;\:\!\?\-\"\'…“”]*$')

def _merge_short_fragments(sents, min_len=15):
    if not sents: return sents
    merged = []
    carry = ""
    for i, sent in enumerate(sents):
        if carry:
            sent = carry + " " + sent
            carry = ""
        if _DIALOGUE_TAG_RE.match(sent.strip()) and i < len(sents) - 1:
            carry = sent
            continue
        if _NOISE_RE.match(sent.strip()):
            if merged:
                merged[-1] = merged[-1] + " " + sent
            else:
                carry = sent
            continue
        if (len(sent.strip()) < min_len 
            and not sent.strip().endswith(('.', '!', '?', '…'))
            and i < len(sents) - 1):
            carry = sent
            continue
        merged.append(sent)
    if carry:
        if merged: merged[-1] = merged[-1] + " " + carry
        else: merged.append(carry)
    return merged

def yield_overlaps(lines, num_overlaps):
    lines = [_preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for out_line in _layer(lines, overlap):
            # check must be here so all outputs are unique
            out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
            yield out_line2

def _layer(lines, num_overlaps, comb=' '):
    if num_overlaps < 1:
        raise Exception('num_overlaps must be >= 1')
    out = ['PAD', ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii:ii + num_overlaps]))
    return out
    
def _preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line
    
class LANG:
    SPLITTER = {
        'ca': 'Catalan',
        'zh': 'Chinese',
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'en': 'English',
        'fi': 'Finnish',
        'fr': 'French',
        'de': 'German',
        'el': 'Greek',
        'hu': 'Hungarian',
        'is': 'Icelandic',
        'it': 'Italian',
        'lt': 'Lithuanian',
        'lv': 'Latvian',
        'no': 'Norwegian',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ro': 'Romanian',
        'ru': 'Russian',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'es': 'Spanish',
        'sv': 'Swedish',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
    }
    ISO = {
		'aa': 'Afar',
		'ab': 'Abkhaz',
		'af': 'Afrikaans',
		'ak': 'Akan',
		'am': 'Amharic',
		'an': 'Aragonese',
		'ar': 'Arabic',
		'as': 'Assamese',
		'av': 'Avaric',
		'ay': 'Aymara',
		'az': 'Azerbaijani',
		'ba': 'Bashkir',
		'be': 'Belarusian',
		'bg': 'Bulgarian',
		'bh': 'Bihari',
		'bi': 'Bislama',
		'bm': 'Bambara',
		'bn': 'Bengali',
		'bo': 'Tibetan',
		'br': 'Breton',
		'bs': 'Bosnian',
		'ca': 'Catalan',
		'ce': 'Chechen',
		'ch': 'Chamorro',
		'co': 'Corsican',
		'cr': 'Cree',
		'cs': 'Czech',
		'cv': 'Chuvash',
		'cy': 'Welsh',
		'da': 'Danish',
		'de': 'German',
		'dv': 'Divehi',
		'dz': 'Dzongkha',
		'ee': 'Ewe',
		'el': 'Greek',
		'en': 'English',
		'es': 'Spanish',
		'et': 'Estonian',
		'eu': 'Basque',
		'fa': 'Persian',
		'ff': 'Fula',
		'fi': 'Finnish',
		'fj': 'Fijian',
		'fo': 'Faroese',
		'fr': 'French',
		'fy': 'Western Frisian',
		'ga': 'Irish',
		'gd': 'Scottish Gaelic',
		'gl': 'Galician',
		'gn': 'Guaraní',
		'gu': 'Gujarati',
		'gv': 'Manx',
		'ha': 'Hausa',
		'he': 'Hebrew',
		'hi': 'Hindi',
		'ho': 'Hiri Motu',
		'hr': 'Croatian',
		'ht': 'Haitian',
		'hu': 'Hungarian',
		'hy': 'Armenian',
		'hz': 'Herero',
		'id': 'Indonesian',
		'ig': 'Igbo',
		'ii': 'Nuosu',
		'ik': 'Inupiaq',
		'io': 'Ido',
		'is': 'Icelandic',
		'it': 'Italian',
		'iu': 'Inuktitut',
		'ja': 'Japanese',
		'jv': 'Javanese',
		'ka': 'Georgian',
		'kg': 'Kongo',
		'ki': 'Kikuyu',
		'kj': 'Kwanyama',
		'kk': 'Kazakh',
		'kl': 'Kalaallisut',
		'km': 'Khmer',
		'kn': 'Kannada',
		'ko': 'Korean',
		'kr': 'Kanuri',
		'ks': 'Kashmiri',
		'ku': 'Kurdish',
		'kv': 'Komi',
		'kw': 'Cornish',
		'ky': 'Kyrgyz',
		'lb': 'Luxembourgish',
		'lg': 'Ganda',
		'li': 'Limburgish',
		'ln': 'Lingala',
		'lo': 'Lao',
		'lt': 'Lithuanian',
		'lu': 'Luba-Katanga',
		'lv': 'Latvian',
		'mg': 'Malagasy',
		'mh': 'Marshallese',
		'mi': 'Māori',
		'mk': 'Macedonian',
		'ml': 'Malayalam',
		'mn': 'Mongolian',
		'mr': 'Marathi',
		'ms': 'Malay',
		'mt': 'Maltese',
		'my': 'Burmese',
		'na': 'Nauru',
		'nb': 'Norwegian Bokmål',
		'nd': 'North Ndebele',
		'ne': 'Nepali',
		'ng': 'Ndonga',
		'nl': 'Dutch',
		'nn': 'Norwegian Nynorsk',
		'no': 'Norwegian',
		'nr': 'South Ndebele',
		'nv': 'Navajo',
		'ny': 'Chichewa',
		'oc': 'Occitan',
		'oj': 'Ojibwe',
		'om': 'Oromo',
		'or': 'Oriya',
		'os': 'Ossetian',
		'pa': 'Panjabi',
		'pl': 'Polish',
		'ps': 'Pashto',
		'pt': 'Portuguese',
		'qu': 'Quechua',
		'rm': 'Romansh',
		'rn': 'Kirundi',
		'ro': 'Romanian',
		'ru': 'Russian',
		'rw': 'Kinyarwanda',
		'sa': 'Sanskrit',
		'sc': 'Sardinian',
		'sd': 'Sindhi',
		'se': 'Northern Sami',
		'sg': 'Sango',
		'si': 'Sinhala',
		'sk': 'Slovak',
		'sl': 'Slovenian',
		'sm': 'Samoan',
		'sn': 'Shona',
		'so': 'Somali',
		'sq': 'Albanian',
		'sr': 'Serbian',
		'ss': 'Swati',
		'st': 'Southern Sotho',
		'su': 'Sundanese',
		'sv': 'Swedish',
		'sw': 'Swahili',
		'ta': 'Tamil',
		'te': 'Telugu',
		'tg': 'Tajik',
		'th': 'Thai',
		'ti': 'Tigrinya',
		'tk': 'Turkmen',
		'tl': 'Tagalog',
		'tn': 'Tswana',
		'to': 'Tonga',
		'tr': 'Turkish',
		'ts': 'Tsonga',
		'tt': 'Tatar',
		'tw': 'Twi',
		'ty': 'Tahitian',
		'ug': 'Uighur',
		'uk': 'Ukrainian',
		'ur': 'Urdu',
		'uz': 'Uzbek',
		've': 'Venda',
		'vi': 'Vietnamese',
		'wa': 'Walloon',
		'wo': 'Wolof',
		'xh': 'Xhosa',
		'yi': 'Yiddish',
		'yo': 'Yoruba',
		'za': 'Zhuang',
		'zh': 'Chinese',
		'zu': 'Zulu',
    }
