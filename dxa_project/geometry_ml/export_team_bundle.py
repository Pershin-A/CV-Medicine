"""Export an explicit allowlist of code and the portable report, without raw data."""
from pathlib import Path
import hashlib,json,shutil

def main():
    root=Path(__file__).resolve().parents[2]
    destination=root/'dxa_project/outputs/git_team_bundle'
    destination.mkdir(parents=True,exist_ok=True)
    selected=set()
    for folder in ('dxa_project/geometry_ml','dxa_project/augmentation','dxa_project/tests','labeler'):
        selected.update(p for p in (root/folder).glob('*') if p.is_file() and p.suffix in ('.py','.md'))
    selected.update((root/'dxa_project/team_demo').rglob('*'))
    selected.update(root/path for path in ('dxa_project/README.md','dxa_project/prepare.py',
        'dxa_project/run_augmented_experiment.py','dxa_project/DXA_augmented_pipeline.ipynb','dxa_project/DXA_improvement_variants.ipynb',
        'dxa_project/GIT_PUBLICATION.md','dxa_project/NEXT_STEPS.md','dxa_project/requirements-geometry.txt',
        'labeler/requirements.txt','labeler/Dockerfile','labeler/docker-compose.yml',
        'labeler/.env.example','labeler/.env.augmented.example','labeler/.env.pilot.example'))
    inventory=[]
    for path in sorted(p for p in selected if p.is_file()):
        relative=path.relative_to(root)
        assert path.suffix.lower() not in ('.dcm','.pt','.pth','.xlsx')
        assert path.name!='.env'
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        inventory.append({'path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination/'FILES.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'README.md').write_text('# DXA: код и отчёт для команды\n\nОткройте `dxa_project/team_demo/index.html` в браузере. Новые эксперименты: `dxa_project/team_demo/improvements.html`. Для просмотра не нужны Python, GPU, данные или веса.\n\nИнструкции публикации: `dxa_project/GIT_PUBLICATION.md`. Модели, исходные DICOM, личные настройки и полная разметка не входят в эту папку. Для собственного инференса нужны четыре основных чекпойнта, новый `hip_points.pt` и DICOM, передаваемые отдельно; калибровку и `landmark_geometry_bounds.json` также переносите с весами.\n',encoding='utf-8')
    (destination/'.gitignore').write_text('Исследования/\nРазмеченные*/\n*.dcm\n*.dicom\n*.xlsx\n*.pt\n*.pth\n.venv/\n__pycache__/\n.env\ndxa_project/outputs/\n',encoding='utf-8')
    print(json.dumps({'bundle':str(destination),'files':len(inventory),'megabytes':sum(i['bytes'] for i in inventory)/1024**2},ensure_ascii=False))

if __name__=='__main__':main()
