"""Verify the portable report assets, source-free JSON and export allowlist."""
from pathlib import Path
from html.parser import HTMLParser
import json
from PIL import Image

class Assets(HTMLParser):
    def __init__(self):super().__init__();self.images=[]
    def handle_starttag(self,tag,attrs):
        if tag=='img':self.images.append(dict(attrs)['src'])

def main():
    root=Path(__file__).resolve().parents[2]
    publication=root/'dxa_project/team_demo'
    parser=Assets();parser.feed((publication/'index.html').read_text(encoding='utf-8'))
    for relative in set(parser.images):
        target=(publication/relative).resolve();assert target.is_relative_to(publication.resolve())
        assert target.is_file();Image.open(target).verify()
    assert len(json.loads((publication/'full_examples.json').read_text(encoding='utf-8')))==5
    for path in publication.glob('*.json'):
        value=json.loads(path.read_text(encoding='utf-8'))
        def check(obj):
            if isinstance(obj,dict):
                assert not ({'source','relative_path','source_relative_path','study','study_uid','source_study_uid'} & set(obj))
                for v in obj.values():check(v)
            elif isinstance(obj,list):
                for v in obj:check(v)
        check(value)
    bundle=root/'dxa_project/outputs/git_team_bundle'
    inventory=json.loads((bundle/'FILES.json').read_text(encoding='utf-8'))
    for item in inventory:
        path=bundle/item['path'];assert path.is_file()
        assert path.suffix.lower() not in ('.dcm','.dicom','.pt','.pth','.xlsx') and path.name!='.env'
        assert not item['path'].startswith(('Размеченные/','Исследования/','dxa_project/outputs/'))
    print(json.dumps({'report_images':len(set(parser.images)),'full_examples':5,'exported_files':len(inventory),'valid':True}))

if __name__=='__main__':main()
