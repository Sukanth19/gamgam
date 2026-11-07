"""Turn wikitext into clean, citeable chunks.

Pipeline per page:
1. Pull the infobox out as "key: value" facts (its own chunk).
2. Split the page by headings (lead + each section, keeping the heading path).
3. Render templates: keep the TEXT of quote/dialogue/spoiler templates (that's
   where a lot of lore lives), drop navigation boxes, galleries, icons, etc.
4. Strip remaining markup (links, bold, refs) to plain text.
5. Split long sections into ~250-word chunks with a small overlap, so each
   chunk is small enough to be specific but still makes sense alone.

Spoilers: wikis mark them with {{spoiler}} (whole page) or {{spoiler|text}}
(inline). We record both as flags - phase 6 will use them.
"""

import re
from dataclasses import dataclass, field

import mwparserfromhell as mw
from mwparserfromhell.nodes import Template

from app import config

# Templates whose parameters ARE content (quotes, dialogue, spoiler-wrapped text).
TEXT_TEMPLATE_RE = re.compile(r"quot|dialog|spoiler|^w$|^wp$|tooltip|lore|journal|description|"
                              r"text|^sic$|^nowrap$|^lang", re.IGNORECASE)
INFOBOX_RE = re.compile(r"infobox", re.IGNORECASE)
SPOILER_RE = re.compile(r"^spoilers?$", re.IGNORECASE)
# Infobox fields that are presentation, not facts.
INFOBOX_SKIP_KEYS = re.compile(r"image|img|caption|icon|sprite|gallery|^name$|title|map|width|"
                               r"size|color|colour|sound|audio|video|tab\d*$|^id$", re.IGNORECASE)
# Sections that rarely contain lore.
SKIP_SECTIONS = re.compile(r"^(gallery|images?|screenshots|videos?|sound ?files|audio|references|"
                           r"notes|footnotes|external links|see also|navigation|achievements|"
                           r"trivia|patch(es)?|update history|version history|concept art|"
                           r"sprites?|other languages|in other languages)$", re.IGNORECASE)


@dataclass
class Chunk:
    section: str           # "Background > Early life" ("" = lead / intro)
    text: str
    spoiler: bool = False
    kind: str = "text"     # "text" | "infobox"


@dataclass
class ParsedPage:
    chunks: list[Chunk] = field(default_factory=list)
    page_spoiler: bool = False


MEDIA_RE = re.compile(r"\.(wav|mp3|ogg|flac|png|jpe?g|gif|webp|svg|webm|mp4)\s*$", re.IGNORECASE)
MEDIA_PARAM_RE = re.compile(r"sound|audio|file|image|img|icon|video", re.IGNORECASE)
_HTML_HEADING = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def _render(code: mw.wikicode.Wikicode, flags: dict) -> str:
    """Replace templates with their text (or nothing), recursively."""
    out = []
    for node in code.nodes:
        if isinstance(node, Template):
            name = str(node.name).strip()
            if SPOILER_RE.match(name):
                if node.params:
                    flags["spoiler"] = True          # inline {{spoiler|text}}
                    out.append(" ".join(_render(p.value, flags) for p in node.params))
                else:
                    flags["page_spoiler"] = True     # bare {{spoiler}} banner
            elif INFOBOX_RE.search(name):
                continue                             # handled separately
            elif TEXT_TEMPLATE_RE.search(name):
                parts = []
                for p in node.params:
                    if p.showkey and MEDIA_PARAM_RE.search(str(p.name)):
                        continue
                    val = _render(p.value, flags).strip()
                    if MEDIA_RE.search(val):
                        continue
                    # positional params are content; named ones only if they look like prose
                    if val and (not p.showkey or len(val) > 25):
                        parts.append(val)
                if parts:
                    out.append(("\n" if "quot" in name.lower() or "dialog" in name.lower() else " ")
                               + " — ".join(parts) + "\n")
            # any other template (nav boxes, icons, galleries...) is dropped
        else:
            out.append(str(node))
    return "".join(out)


def _plain(wikitext: str) -> str:
    text = mw.parse(wikitext).strip_code(normalize=True, collapse=True)
    text = re.sub(r"<[^>]+>", " ", text)                      # leftover html tags
    text = re.sub(r"^\s*(thumb|left|right|\d+px)\|.*$", "", text, flags=re.MULTILINE)
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.splitlines())
    return _NL.sub("\n\n", text).strip()


def extract_infobox(code: mw.wikicode.Wikicode) -> list[str]:
    for tpl in code.filter_templates(recursive=False):
        if INFOBOX_RE.search(str(tpl.name)):
            facts = []
            for p in tpl.params:
                key = str(p.name).strip()
                if not p.showkey or INFOBOX_SKIP_KEYS.search(key):
                    continue
                val = _plain(_render(p.value, {}))
                val = re.sub(r"\s*\n\s*", ", ", val).strip(" ,")
                if val:
                    facts.append(f"{key.replace('_', ' ')}: {val}")
            return facts
    return []


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    """Split on sentence boundaries into ~size-word pieces with ~overlap words shared."""
    sentences = re.split(r"(?<=[.!?…])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        n = len(s.split())
        if cur and cur_len + n > size:
            chunks.append(" ".join(cur))
            # carry the last sentences (up to `overlap` words) into the next chunk
            keep, kept = [], 0
            for prev in reversed(cur):
                kept += len(prev.split())
                if kept > overlap:
                    break
                keep.insert(0, prev)
            cur, cur_len = keep, sum(len(x.split()) for x in keep)
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def normalize_wikitext(wikitext: str) -> str:
    """- <ref> footnotes: citations, not article text; inline they glue words together.
    - Some wikis write headings as HTML (<h2>Lore</h2>): convert to == Lore ==
      so section splitting works the same everywhere."""
    text = _REF.sub("", wikitext)
    return _HTML_HEADING.sub(lambda m: "\n" + "=" * int(m.group(1)) + " " + m.group(2).strip()
                             + " " + "=" * int(m.group(1)) + "\n", text)


def parse_page(wikitext: str, size: int = config.CHUNK_WORDS,
               overlap: int = config.CHUNK_OVERLAP_WORDS) -> ParsedPage:
    code = mw.parse(normalize_wikitext(wikitext))
    page = ParsedPage()

    facts = extract_infobox(code)
    if facts:
        page.chunks.append(Chunk(section="Infobox", text="\n".join(facts), kind="infobox"))

    path: list[tuple[int, str]] = []          # heading stack: [(level, title)]
    skip_level = None                         # inside a skipped section (and its subsections)
    for section in code.get_sections(flat=True, include_lead=True, include_headings=True):
        headings = section.filter_headings(recursive=False)
        if headings:
            h = headings[0]
            title = _plain(str(h.title))
            path = [(lvl, t) for lvl, t in path if lvl < h.level] + [(h.level, title)]
            section.remove(h)
            if skip_level is not None and h.level <= skip_level:
                skip_level = None
            if skip_level is None and SKIP_SECTIONS.match(title):
                skip_level = h.level
        if skip_level is not None:
            continue
        flags: dict = {}
        text = _plain(_render(section, flags))
        page.page_spoiler |= flags.get("page_spoiler", False)
        if len(text.split()) < config.CHUNK_MIN_WORDS:
            continue
        sec_name = " > ".join(t for _, t in path)
        for piece in _split_words(text, size, overlap):
            page.chunks.append(Chunk(section=sec_name, text=piece, spoiler=flags.get("spoiler", False)))
    return page
