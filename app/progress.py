"""Story milestones per game, and the user's progress through them.

Each game gets a short ORDERED list of checkpoints. Wiki sections are tagged
with the earliest milestone at which a player would know what they reveal
(app/spoilers.py); questions then only see sections at or before the user's
progress.

Milestones are deliberately coarse: players can say "I'm at City of Tears",
not "I'm 37% through". `aliases` are lowercase phrases that identify a
milestone when the user describes their progress in chat.

Outer Wilds is non-linear (you can go anywhere from the start), so its
milestones are KNOWLEDGE checkpoints - roughly how deep into the mystery
you are - rather than places.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Milestone:
    label: str
    description: str          # given to the tagging model, so be concrete
    aliases: tuple[str, ...] = ()


MILESTONES: dict[str, list[Milestone]] = {
    "Hollow Knight": [
        Milestone("Opening", "King's Pass, Dirtmouth, Forgotten Crossroads, False Knight, Elderbug, Cornifer, Sly.",
                  ("start", "dirtmouth", "crossroads", "false knight", "king's pass")),
        Milestone("Greenpath", "Greenpath, Hornet's first fight, Lake of Unn, Moss Knights, the Stag Station network.",
                  ("greenpath", "hornet fight", "first hornet")),
        Milestone("Fungal Wastes & City of Tears", "Fungal Wastes, Mantis Lords, Mantis Village, City of Tears, "
                  "Soul Sanctum and the Soul Master, Watcher's Spire, Lemm, Quirrel at the Blue Lake.",
                  ("fungal", "mantis", "city of tears", "soul master", "soul sanctum")),
        Milestone("Crystal Peak & Resting Grounds", "Crystal Peak, Resting Grounds, the Dream Nail and the Seer, "
                  "Royal Waterways, Dung Defender, learning about the Dreamers and the seal on the Black Egg.",
                  ("crystal peak", "resting grounds", "dream nail", "seer", "waterways", "dung defender")),
        Milestone("Deepnest, Kingdom's Edge & Gardens", "Deepnest, Distant Village, Kingdom's Edge, the Hive, "
                  "Queen's Gardens, Traitor Lord, Ancient Basin, Hornet's second fight, Broken Vessel.",
                  ("deepnest", "kingdom's edge", "queen's gardens", "gardens", "hive", "ancient basin",
                   "broken vessel", "hornet sentinel")),
        Milestone("The Dreamers", "Killing the three Dreamers (Monomon, Lurien, Herrah) to break the seal; "
                  "Uumuu, the Watcher Knights.",
                  ("dreamers", "monomon", "lurien", "herrah", "uumuu", "watcher knights")),
        Milestone("Abyss & White Palace", "The Abyss, Lifeblood/Void Heart, the Shade Lord, the truth of the Vessels "
                  "and the Hollow Knight's creation, the White Palace, the Pale King's plan, the White Lady.",
                  ("abyss", "void heart", "white palace", "kingsoul", "shade cloak")),
        Milestone("Endings", "Fighting the Hollow Knight in the Black Egg Temple, the Radiance, all endings "
                  "(Hollow Knight, Sealed Siblings, Dream No More).",
                  ("ending", "final boss", "black egg", "radiance", "finished", "beat the game", "completed")),
        Milestone("DLC & Godhome", "The Grimm Troupe, Godmaster, Godhome, the Pantheons, Absolute Radiance, "
                  "Embrace the Void and Delicate Flower endings.",
                  ("grimm", "godmaster", "godhome", "pantheon", "100%", "dlc")),
    ],
    "Outer Wilds": [
        Milestone("Launch", "Timber Hearth, the Hearthians, Outer Wilds Ventures, launch codes, the observatory, "
                  "the first time loop (dying and waking at the campfire).",
                  ("start", "timber hearth", "launch", "first loop", "time loop")),
        Milestone("Meeting the Nomai", "Learning the Nomai existed, Nomai writing, the Nomai statue, other "
                  "Hearthian travelers (Chert, Esker, Riebeck, Gabbro, Feldspar), the signalscope.",
                  ("nomai", "translator", "travelers", "signalscope")),
        Milestone("Exploring the planets", "Visiting the Hourglass Twins, Brittle Hollow and the Hanging City, "
                  "Giant's Deep, Dark Bramble's surface, the Interloper's surface, the Quantum Moon from outside.",
                  ("hourglass", "ember twin", "brittle hollow", "hanging city", "giant's deep", "dark bramble",
                   "interloper", "exploring")),
        Milestone("The Eye's signal", "Why the Nomai came to the solar system: the signal from the Eye of the "
                  "Universe, the Orbital Probe Cannon, the Nomai's search for the Eye.",
                  ("eye's signal", "probe cannon", "why the nomai came")),
        Milestone("Ghost matter & the Sun Station", "Ghost matter, the Interloper's core and what killed the "
                  "Nomai, the Sun Station and its failure.",
                  ("ghost matter", "sun station", "what killed the nomai")),
        Milestone("The Ash Twin Project", "What causes the time loop: the Ash Twin Project, the advanced warp core, "
                  "the supernova, the High Energy Lab, why the loop exists.",
                  ("ash twin project", "warp core", "what causes the loop", "supernova")),
        Milestone("Quantum Moon & the Vessel", "The Quantum Moon's sixth location, Solanum, the quantum rules, "
                  "the Nomai Vessel in Dark Bramble and how to reach the Eye.",
                  ("quantum moon", "solanum", "vessel", "quantum rules")),
        Milestone("The Eye & ending", "Reaching the Eye of the Universe, the campfire with the travelers, "
                  "the end of the universe and the ending.",
                  ("eye of the universe", "ending", "finished", "beat the game", "completed")),
        Milestone("Echoes of the Eye (DLC)", "The Stranger, the Owlks, the Prisoner, the slide reels, the "
                  "Dreamworld, everything in the Echoes of the Eye expansion.",
                  ("echoes of the eye", "stranger", "owlk", "prisoner", "dreamworld", "dlc")),
    ],
    "The Witcher 3: Wild Hunt": [
        Milestone("Background & White Orchard", "General world background a new player learns or that comes from "
                  "the books (Geralt, Yennefer, Ciri as Geralt's adopted daughter, the Continent, Nilfgaard, "
                  "Witchers), plus the prologue: Kaer Morhen dream, White Orchard, the griffin, Emperor Emhyr.",
                  ("start", "prologue", "white orchard", "griffin")),
        Milestone("Velen & the Bloody Baron", "Velen, Crow's Perch, the Bloody Baron's family, the Crones of "
                  "Crookback Bog, Keira Metz.",
                  ("velen", "bloody baron", "crones", "crookback", "keira")),
        Milestone("Novigrad", "Novigrad, Triss, Dandelion, Zoltan, the Witch Hunters, Radovid's persecution of "
                  "mages, the Reason of State conspiracy setup, Dudu, Whoreson Junior.",
                  ("novigrad", "triss", "dandelion", "radovid", "whoreson")),
        Milestone("Skellige", "The Skellige Isles, Yennefer's investigation, Crach an Craite, the King's Gambit, "
                  "Uma's discovery at Kaer Morhen.",
                  ("skellige", "kaer trolde", "crach", "uma")),
        Milestone("Finding Ciri", "Lifting Uma's curse (Avallac'h), the Isle of Mists, finding Ciri, Ciri's "
                  "story of her escape and Avallac'h's role.",
                  ("isle of mists", "found ciri", "finding ciri", "lifting the curse", "ugly baby")),
        Milestone("Battle of Kaer Morhen", "Gathering allies, the Battle of Kaer Morhen, Vesemir's death, the "
                  "Wild Hunt's attack, Ciri's grief.",
                  ("battle of kaer morhen", "kaer morhen", "vesemir")),
        Milestone("Final act & endings", "Reason of State, Radovid's fate, the Sunstone, Skellige battle against "
                  "the Hunt, Eredin's defeat, the White Frost, Ciri's fate and all endings.",
                  ("final act", "eredin", "white frost", "ending", "finished", "beat the game", "completed")),
        Milestone("Hearts of Stone (DLC)", "Gaunter O'Dimm, Olgierd von Everec, the Hearts of Stone expansion.",
                  ("hearts of stone", "o'dimm", "olgierd")),
        Milestone("Blood and Wine (DLC)", "Toussaint, Regis, Anna Henrietta, Syanna, the Blood and Wine expansion.",
                  ("blood and wine", "toussaint", "regis", "syanna")),
    ],
    "Portal 2": [
        Milestone("Portal 1 (prequel)", "Everything that happens in the first Portal game: Chell, GLaDOS, the "
                  "Enrichment Center tests, the cake, GLaDOS's destruction at the end of Portal.",
                  ("portal 1", "first game", "prequel", "start")),
        Milestone("Ch. 1-2: Courtesy Call & Cold Boot", "Waking in the Relaxation Vault, Wheatley, the ruined "
                  "facility, reactivating GLaDOS by accident.",
                  ("chapter 1", "chapter 2", "courtesy call", "cold boot", "wake up")),
        Milestone("Ch. 3-4: The Return & The Surprise", "Testing under a revived, resentful GLaDOS, rebuilt test "
                  "chambers, the 'surprise' escape attempt with Wheatley.",
                  ("chapter 3", "chapter 4", "the return", "the surprise")),
        Milestone("Ch. 5: The Escape", "Escaping with Wheatley, sabotaging the turret and neurotoxin production, "
                  "the core transfer that puts Wheatley in charge and GLaDOS in a potato.",
                  ("chapter 5", "the escape", "core transfer")),
        Milestone("Ch. 6-7: The Fall & The Reunion", "Falling into Old Aperture, Cave Johnson's recordings, "
                  "Caroline, GLaDOS's origin, PotatOS, the gels, climbing back up.",
                  ("chapter 6", "chapter 7", "the fall", "the reunion", "old aperture", "cave johnson",
                   "potatos", "caroline")),
        Milestone("Ch. 8-9 & ending", "Wheatley's tests, the final battle with Wheatley, the portal to the moon, "
                  "GLaDOS deleting Caroline, Chell's release, the ending song.",
                  ("chapter 8", "chapter 9", "the itch", "kills you", "moon", "ending", "finished",
                   "beat the game", "completed")),
        Milestone("Co-op & DLC", "The cooperative campaign with Atlas and P-body, the human vault, Peer Review DLC.",
                  ("co-op", "coop", "atlas", "p-body", "peer review", "dlc")),
    ],
}


def milestones_for(game: str) -> list[Milestone]:
    return MILESTONES.get(game, [])


def finished_index(game: str) -> int:
    """Index of the main-story ending (DLC milestones come after it)."""
    ms = milestones_for(game)
    for i, m in enumerate(ms):
        if "finished" in m.aliases:
            return i
    return len(ms) - 1


def parse_progress(game: str, text: str) -> int | None:
    """Find which milestone a sentence like "I just reached City of Tears" names.
    Takes the LATEST milestone mentioned (the furthest point the user claims)."""
    t = text.lower()
    best = None
    for i, m in enumerate(milestones_for(game)):
        names = (m.label.lower(),) + m.aliases
        if any(re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])", t) for a in names):
            best = i
    return best
