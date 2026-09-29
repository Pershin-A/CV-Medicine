"""Capture code, annotation, split and checkpoint fingerprints for this run."""
from pathlib import Path
import hashlib,json,platform
import torch,torchvision

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    source={p.relative_to(root).as_posix():digest(p) for p in (root/'dxa_project/geometry_ml').glob('*.py')}
    datafiles=[root/'Размеченные/labels.csv',root/'dxa_project/outputs/manifest.csv',out/'protocol.json']
    annotations=hashlib.sha256()
    for p in sorted((root/'Размеченные/geometry').glob('*.json')):
        annotations.update(p.name.encode());annotations.update(p.read_bytes())
    report={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
            'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'code_sha256':source,'data_sha256':{p.relative_to(root).as_posix():digest(p) for p in datafiles},
            'geometry_collection_sha256':annotations.hexdigest(),
            'pipeline_checkpoints_sha256':{p.name:digest(p) for p in (out/'pipeline').glob('*.pt')},
            'note':'Local reproducibility manifest; snapshot after training. CUDA kernels may not be bitwise deterministic.'}
    (out/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'provenance':str(out/'provenance.json'),'gpu':report['gpu']}))

if __name__=='__main__':main()
