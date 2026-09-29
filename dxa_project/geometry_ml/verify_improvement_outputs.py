"""Final integrity checks for the executed notebook and portable improvement report."""
from pathlib import Path
from html.parser import HTMLParser
import json,nbformat

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('src','href'))

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2';public=root/'dxa_project/team_demo'
    notebook=nbformat.read(root/'dxa_project/DXA_improvement_variants.ipynb',as_version=4);nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert code and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    parser=Links();parser.feed((public/'improvements.html').read_text(encoding='utf-8'))
    for link in parser.links:
        if not link.startswith(('#','http:','https:')):assert (public/link).is_file(),link
    raw=(public/'improvements_metrics.json').read_text(encoding='utf-8');metrics=json.loads(raw)
    json.dumps(metrics,allow_nan=False)
    assert all(s not in raw for s in ('C:\\Users\\','PatientName','StudyInstanceUID','relative_path'))
    assert metrics['original_evaluation']['processed_files']==100
    assert not metrics['original_evaluation']['failed_files']
    assert metrics['synthetic_evaluation']['selected_images']==135 and not metrics['synthetic_evaluation']['failures']
    assert len(metrics['examples'])==5
    module_page=public/'model_examples.html'
    if module_page.exists():
        links=Links();links.feed(module_page.read_text(encoding='utf-8'))
        for link in links.links:
            if not link.startswith(('#','http:','https:')):assert (public/link).is_file(),link
        module_raw=(public/'model_examples.json').read_text(encoding='utf-8');module_data=json.loads(module_raw)
        assert all(s not in module_raw for s in ('relative_path','StudyInstanceUID','C:\\Users\\'))
        assert len(module_data['modules'])==8
        for examples in module_data['modules'].values():
            assert examples and all(1<=e['image_number']<=499 for e in examples)
            for e in examples:
                if e['category']=='Успехи':assert e['success']
                if e['category']=='Ошибки':assert not e['success']
        assert [e['image_number'] for e in module_data['manual_review']]==[52,55,58,102,121,122,287,288]
        assert module_data['roi_nominal_review']==[]
    report={'notebook_executed_cells':len(code),'linked_files':len(parser.links),'examples':5,
            'original_files_processed':100,'synthetic_files_processed':135,'finite_public_json':True,'passed':True}
    (out/'final_integrity.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
