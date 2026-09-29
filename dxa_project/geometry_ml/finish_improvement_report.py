"""Build the reviewable report after all improved training/evaluation finishes."""
from pathlib import Path
import json,subprocess,sys,time
import nbformat
from nbclient import NotebookClient

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    while not (out/'complete.json').exists():time.sleep(5)
    for module in ('localization_comparison','geometry_rules','report_improvements','make_variant_notebook','experiment_provenance'):
        subprocess.run([sys.executable,'-m','dxa_project.geometry_ml.'+module],cwd=root,check=True)
    notebook=nbformat.read(root/'dxa_project/DXA_improvement_variants.ipynb',as_version=4)
    NotebookClient(notebook,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(notebook,out/'variants_executed.ipynb')
    nbformat.write(notebook,root/'dxa_project/DXA_improvement_variants.ipynb')
    subprocess.run([sys.executable,'-m','dxa_project.geometry_ml.export_team_bundle'],cwd=root,check=True)
    (out/'report_complete.json').write_text(json.dumps({'report_complete':True,'executed_notebook':True}),encoding='utf-8')
    print(json.dumps({'report_complete':True}),flush=True)

if __name__=='__main__':main()
