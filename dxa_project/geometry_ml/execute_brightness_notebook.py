"""Execute the report notebook with training disabled, then verify artifacts."""
from pathlib import Path
import re
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'DXA_Spine_Brightness_20260929.ipynb'
b=nbformat.read(p,as_version=4)
assert any('RUN_TRAINING=False' in c.source for c in b.cells)
if not any('decoded_geometry_audit' in c.source for c in b.cells):
    b.cells.insert(4,nbformat.v4.new_code_cell('''print("Ограничения всех декодированных линий")
display(pd.DataFrame({k:{a:b for a,b in v.items() if a!="details"} for k,v in r["decoded_geometry_audit"].items()}).T)
display(pd.DataFrame(r["contour_geometry_audit"]).T)'''))
NotebookClient(b,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
assert not any(o.output_type=='error' for c in b.cells if c.cell_type=='code' for o in c.outputs)
nbformat.write(b,p)
for name in ['spine_brightness_study.py','experimental_spine_axes.py','finalize_brightness_study.py']:
    f=Path(__file__).parent/name
    compile(f.read_text(encoding='utf-8'),str(f),'exec')
page=ROOT/'dxa_project/team_demo/spine_brightness_study_20260929.html'
html=page.read_text(encoding='utf-8')
images=re.findall(r'src="(spine_brightness_assets/[^\"]+)"',html)
assert images and all((page.parent/p).is_file() for p in images)
assert html.count('<table>')==html.count('</table>')
assert 'final-audit-note' in html
print({'notebook_cells':len(b.cells),'linked_images':len(images),'sources_compile':'OK','notebook_errors':0})
