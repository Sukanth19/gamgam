"""Which wikis to pull lore from. Adding a game = adding one entry here.

Any MediaWiki site works (Fandom or not): we only need its api.php URL.
`categories` are fetched in order; within each category the LONGEST pages
come first (major characters/places have long pages, one-line NPCs don't),
and the whole game is capped at `max_pages`.

All four wikis are Creative Commons licensed (CC BY-SA / CC BY): we store each
page's URL so answers can cite and attribute their source.
"""

WIKIS = [
    {
        "game": "Hollow Knight",
        "api": "https://hollowknight.fandom.com/api.php",
        "categories": ["Lore", "NPCs (Hollow Knight)", "Bosses (Hollow Knight)",
                       "Areas (Hollow Knight)", "Points of Interest (Hollow Knight)",
                       "Items (Hollow Knight)", "Quests (Hollow Knight)",
                       "Enemies (Hollow Knight)"],   # enemies carry Hunter's Journal lore
        "max_pages": 400,
    },
    {
        "game": "Outer Wilds",
        "api": "https://outerwilds.fandom.com/api.php",
        "categories": ["Characters", "Nomai", "Locations", "Planets", "Technology",
                       "Echoes of the Eye", "Races"],
        "max_pages": 250,
    },
    {
        "game": "The Witcher 3: Wild Hunt",
        "api": "https://witcher.fandom.com/api.php",
        # ~1000 pages total; the wiki also covers the books and show, so we
        # stick to the game's own categories and take the longest pages.
        "categories": ["The Witcher 3 characters", "The Witcher 3 locations",
                       "The Witcher 3 bestiary", "The Witcher 3 quests"],
        "max_pages": 250,
    },
    {
        "game": "Portal 2",
        "api": "https://theportalwiki.com/w/api.php",
        "categories": ["Characters", "Portal 2"],
        "max_pages": 80,
    },
]

# Pages whose titles match this are skipped (translations like "GLaDOS/fr",
# list/meta pages).
SKIP_TITLE_RE = r"/[a-z]{2}(-[a-z]+)?$|^List of |\(disambiguation\)$"
