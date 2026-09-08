#!/usr/bin/env python3
from html.parser import HTMLParser
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()
    def handle_starttag(self, _tag, attrs):
        for key, value in attrs:
            if key in {'src', 'href', 'poster'} and value and not value.startswith(('http:', 'https:', '#', 'mailto:')):
                self.paths.add(value.split('#', 1)[0].split('?', 1)[0])

parser = Links()
parser.feed((ROOT / 'index.html').read_text())
css = (ROOT / 'assets/css/style.css').read_text()
parser.paths.update(re.findall(r'url\(["\']?([^"\')]+)', css))
missing = sorted(path for path in parser.paths if not (ROOT / path).exists())
assert not missing, f'Missing local assets: {missing}'
assert (ROOT / 'data/main_results.json').is_file()
print(f'OK: {len(parser.paths)} local references and results data exist')
