import re
from html.parser import HTMLParser
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "shot_predict" / "labeler_frontend"
VOID = {"meta", "link", "input", "br", "img", "hr"}


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.ids, self.errors, self.scripts = [], set(), [], []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            if attributes["id"] in self.ids:
                self.errors.append(f"duplicate id {attributes['id']}")
            self.ids.add(attributes["id"])
        if tag == "script" and "src" in attributes:
            self.scripts.append(attributes["src"])
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> inside {self.stack[-3:]}")
        else:
            self.stack.pop()


def test_markup_is_balanced_and_every_element_the_scripts_use_exists():
    parsed = Structure()
    parsed.feed((FRONTEND / "index.html").read_text(encoding="utf-8"))
    assert parsed.stack == [] and parsed.errors == []
    source = "".join((FRONTEND / name).read_text(encoding="utf-8") for name in ("labeler.js", "review.js", "validate.js"))
    used = set(re.findall(r'byId\("([^"]+)"\)', source))
    assert used - parsed.ids == set(), f"scripts reference missing elements: {sorted(used - parsed.ids)}"


def test_scripts_load_in_dependency_order_and_all_ship_in_the_package():
    parsed = Structure()
    parsed.feed((FRONTEND / "index.html").read_text(encoding="utf-8"))
    assert parsed.scripts == ["review_logic.js", "labeler.js", "review.js", "validate.js"]
    assert all((FRONTEND / name).is_file() for name in parsed.scripts + ["style.css"])
