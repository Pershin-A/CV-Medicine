"""Compare the second point-loss variant on held-out synthetic pipeline inputs."""
from pathlib import Path
import json,os
from .evaluate_augmented import run

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    source=out/'pipeline';alternative=out/'coordinate_pipeline';alternative.mkdir(exist_ok=True)
    candidates=[x for x in json.loads((out/'point_selection.json').read_text(encoding='utf-8'))['candidates'] if x['variant']=='coordinate256']
    chosen=min(candidates,key=lambda x:x['metrics']['mean_error_mm'])
    for name in ('router.pt','spine.pt','hip.pt','artifact.pt','landmark_geometry_bounds.json'):
        if not (alternative/name).exists():os.link(source/name,alternative/name)
    if not (alternative/'hip_points.pt').exists():os.link(Path(chosen['path']),alternative/'hip_points.pt')
    (alternative/'evaluation').mkdir(exist_ok=True)
    if not (alternative/'evaluation/calibration.json').exists():os.link(source/'evaluation/calibration.json',alternative/'evaluation/calibration.json')
    # Rotation calibration remains identical: both variants share the same mask model.
    run(root,root/'dxa_project/outputs/augmented_15000_final',alternative,alternative/'synthetic_evaluation')

if __name__=='__main__':main()
