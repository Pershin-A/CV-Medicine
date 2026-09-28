"""Audit the finished dataset, train all models, then evaluate the held-out fold."""
import json,subprocess,sys,time
from pathlib import Path

def run():
    root=Path(__file__).resolve().parents[1]
    augmented=root/'dxa_project/outputs/augmented_15000_final'
    experiment=root/'dxa_project/outputs/geometry_ml_augmented_5epochs'
    experiment.mkdir(parents=True,exist_ok=True)
    while not (augmented/'generation_report.json').exists():
        time.sleep(5)
    def command(stage,args):
        print(json.dumps({'stage':stage,'status':'started'}),flush=True)
        with (experiment/f'{stage}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-W','ignore',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'stage':stage,'status':'finished'}),flush=True)
    command('audit',['-m','dxa_project.augmentation.audit_generated',str(augmented),
                     '--output',str(augmented/'audit_report.json')])
    audit=json.loads((augmented/'audit_report.json').read_text(encoding='utf-8'))
    if audit['problems']:raise RuntimeError(f"Dataset audit failed: {len(audit['problems'])} problems")
    command('training',['-m','dxa_project.geometry_ml.train','--augmented-root',str(augmented),
                        '--output',str(experiment),'--epochs','5','--batch-size','8','--loader-workers','2','--fold','0'])
    command('evaluation',['-m','dxa_project.geometry_ml.evaluate','--checkpoints',str(experiment),
                          '--output',str(experiment/'evaluation'),'--fold','0'])
    command('synthetic_evaluation',['-m','dxa_project.geometry_ml.evaluate_augmented',
            '--augmented-root',str(augmented),'--checkpoints',str(experiment),'--output',str(experiment/'synthetic_evaluation')])
    print(json.dumps({'complete':True,'experiment':str(experiment)}),flush=True)

if __name__=='__main__':run()
