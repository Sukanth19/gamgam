"""Wikitext -> chunks, using markup patterns seen on the real wikis."""

from app.chunking import _split_words, normalize_wikitext, parse_page

PAGE = """{{For|the other one|Hornet (Silksong)}}
{{HK Infobox NPC
|image1 = <gallery>Hornet_Idle.png|Idle</gallery>
|area = [[Greenpath]]<br/>
[[City of Tears]]
|gender = Female<ref>Journal entry: "a footnote that must not leak"</ref>
|va = Makoto Koji
}}
{{Quote|Skilled protector of Hallownest's ruins. Wields a needle and thread.}}
'''Hornet''' is a recurring [[character|NPC]] in the game.<ref>source</ref> She guards the ruins and tests the Knight many times.
<h2>Lore</h2>
Hornet is the daughter of the [[Pale King]] and Herrah the Beast, queen of Deepnest, born of a bargain.
== Gallery ==
[[File:Hornet.png|thumb|A picture]] Lots of captions that are not lore at all here.
=== Concept ===
Still inside the gallery section, so this must be skipped as well entirely.
== Dialogue ==
{{Quotation|GLaDOS|We both said a lot of things that you're going to regret.|sound=GLaDOS wakeup06.wav}}
And then {{spoiler|she turns out to be the villain all along in the story}} of course.
"""


def test_infobox_becomes_facts_without_images_or_footnotes():
    chunks = parse_page(PAGE).chunks
    box = chunks[0]
    assert box.kind == "infobox"
    assert box.text.splitlines() == ["area: Greenpath, City of Tears", "gender: Female", "va: Makoto Koji"]


def test_quote_text_kept_and_refs_dropped():
    lead = next(c for c in parse_page(PAGE).chunks if c.section == "")
    assert "Skilled protector of Hallownest's ruins" in lead.text
    assert "Hornet is a recurring NPC in the game." in lead.text
    assert "footnote" not in lead.text and "source" not in lead.text
    assert "Hornet (Silksong)" not in lead.text               # navigation template dropped


def test_html_headings_become_sections_and_gallery_is_skipped_with_subsections():
    sections = [c.section for c in parse_page(PAGE).chunks]
    assert "Lore" in sections
    assert "Dialogue" in sections
    assert not any(s.startswith("Gallery") for s in sections)


def test_inline_spoiler_text_is_kept_and_flagged():
    page = parse_page(PAGE)
    dlg = next(c for c in page.chunks if c.section == "Dialogue")
    assert "she turns out to be the villain" in dlg.text
    assert dlg.spoiler is True
    assert "GLaDOS — We both said a lot of things" in dlg.text
    assert ".wav" not in dlg.text                             # sound file param dropped
    assert page.page_spoiler is False


def test_bare_spoiler_banner_marks_whole_page():
    page = parse_page("{{spoiler}}\nSolanum is a Nomai on the Quantum Moon, the last of her kind in the system.")
    assert page.page_spoiler is True


def test_normalize_converts_html_headings():
    assert "== Lore ==" in normalize_wikitext("<h2>Lore</h2>")
    assert "=== The Protector ===" in normalize_wikitext('<h3 id="x">The Protector</h3>')


def test_split_words_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(60))
    pieces = _split_words(text, size=50, overlap=10)
    assert len(pieces) > 5
    assert all(len(p.split()) <= 57 for p in pieces)          # size + at most one sentence
    first_last = pieces[0].split(". ")[-1]
    assert first_last.rstrip(".") in pieces[1]                # overlap: last sentence repeats


def test_template_only_heading_does_not_leave_empty_path_part():
    page = parse_page("== {{Icon|x}} ==\n=== Journal entry ===\n" + "A monster lives in the swamp and eats travellers at night. " * 3)
    assert [c.section for c in page.chunks] == ["Journal entry"]
