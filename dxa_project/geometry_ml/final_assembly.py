"""Fresh, multilabel-stratified retraining and assembly of metric-selected modules."""
import argparse,csv,json,time,random,hashlib,shutil,gc
from pathlib import Path
import numpy as np
import torch
from .data import load_records,load_augmented_records
from .final_protocol import make_final_protocol,PARTS
from .scoliosis import train_scoliosis
from .train import train_task
from .experiments import train_variant,VARIANTS
from .predict import _load_model

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dxa_project/outputs/final_20260929'
AUG=ROOT/'dxa_project/outputs/augmented_15000_20260929'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def event(stage,**kwargs):
    value={'stage':stage,'updated':time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs};save(OUT/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)


def prepare():
    originals=load_records(ROOT);protocol=make_final_protocol(ROOT,originals,OUT/'protocol.json')
    augmented=load_augmented_records(ROOT,AUG,originals)
    groups={p:[r for r in originals if protocol['partition_by_path'][r.relative_path]==p] for p in PARTS}
    original_by={r.relative_path:r for r in originals};labels=protocol['original_labels'].copy()
    with (AUG/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:augrows={f"aug/{r['image_path']}":r for r in csv.DictReader(f)}
    source_lines={r.relative_path:len(json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['disc_lines']) for r in originals if r.region=='SPINE'}
    for r in augmented:
        row=augrows[r.relative_path];y={}
        for key in ('spine_position','spine_axis','spine_artifact','hip_position','hip_roi','hip_rotation'):
            value=row.get(key,'');y[key]=int(float(value)) if value not in ('',None) else None
        if r.region=='SPINE':
            g=json.loads(r.geometry_path.read_text(encoding='utf-8'));kept=len(g['spine']['disc_lines'])
            y['spine_scoliosis']=labels[r.source_id].get('spine_scoliosis') if kept>=4 and kept==source_lines[r.source_id] else None
        labels[r.relative_path]=y
        if protocol['partition_by_path'][r.source_id]=='train':groups['train'].append(r)
    fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'Размеченные/labels.csv',ROOT/'dxa_project/outputs/manifest.csv',AUG/'manifest.csv')}
    old=OUT/'data_fingerprints.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8'))!=fingerprints:raise RuntimeError('Data changed: create a new version instead of resuming old weights')
    save(old,fingerprints);save(OUT/'labels.json',labels)
    save(OUT/'data_summary.json',{'original_split':protocol['summary'],'training_images':len(groups['train']),
         'training_augmented':sum(bool(r.source_id) for r in groups['train']),'augmented_total':len(augmented),
         'scoliosis_training':{str(v):sum(labels[r.relative_path].get('spine_scoliosis')==v for r in groups['train'] if r.region=='SPINE') for v in (0,1,None)},
         'scoliosis_inheritance':'Rigid affine variants only; exclude examples losing manual dividers or having <4 dividers. Inherited labels remain approximate under crop.',
         'validation_test_augmentation':'Source-group held out; generated examples never enter training across partitions.'})
    save(OUT/'module_plan.json',{'fresh_initialization':'ImageNet / COCO generic pretraining only; no previous project-trained weights because split changed',
         'router':'ResNet18; historical validation accuracy=1 for both pipelines',
         'spine':'ResNet50 spatial decoder + all-peak priors; compare c=2 and c=.5 using actual angular error and divider F1; no direct angle head',
         'spine_crests':'Optional independent ResNet50 point-only module; compare original validation PCK10/presence against joint angle-selected checkpoint before test',
         'hip':'Retrain ResNet18 and ResNet50; select ROI by validation IoU and mask by validation Dice separately',
         'hip_points':'shared256 full frame, masked-logit argmax; previous six-option validation winner; compare light/heavy on new common validation',
         'artifact':'MobileNetV3 / ResNet50 FasterRCNN retrained on common split; select validation box F1, calibrate image presence threshold on original validation',
         'scoliosis':'Separate ResNet18 annotation classifier; epoch / threshold select original validation F1 + balanced accuracy',
         'test_used_for_selection':False})
    return originals,augmented,groups,labels,protocol


def clear():
    _load_model.cache_clear();gc.collect();torch.cuda.empty_cache()


def hip_selection(model,device,stats):return -(.5*stats['pixel_dice']+.5*stats['mean_roi_iou'])


def train_all():
    originals,augmented,groups,labels,protocol=prepare()
    torch.hub.set_dir(str(ROOT/'dxa_project/outputs/torch_hub'));torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for complete final retraining')
    device=torch.device('cuda');bundle=OUT/'bundle';bundle.mkdir(exist_ok=True)
    save(OUT/'runtime.json',{'python':__import__('sys').version,'torch':torch.__version__,'device':torch.cuda.get_device_name(),'maximum_epochs':24,'patience':6,'training_original_groups_only':True})
    if not (bundle/'scoliosis_report.json').exists():
        event('training',task='scoliosis');train_scoliosis(groups['train'],groups['validation'],labels,bundle,device)
    for task,architecture,size,batch in [('router','light',256,8),('artifact','light',256,8),('hip','light',256,8),('hip','heavy',384,2)]:
        folder=OUT/(task+'_'+architecture);report_path=folder/(task+'_report.json')
        folder.mkdir(parents=True,exist_ok=True)
        if not report_path.exists():
            clear();event('training',task=task,architecture=architecture);random.seed(42);np.random.seed(42);torch.manual_seed(42)
            report=train_task(task,groups['train'],groups['validation'],folder,device,12 if task=='router' else 24,size,batch,None,True,
                architecture,2,1e-5 if task=='artifact' else 2e-5,3e-5 if task=='artifact' else 1e-3 if task=='router' else 2e-4,
                'cosine',6,True,selection_callback=hip_selection if task=='hip' else None,mixed_precision=architecture=='heavy')
            save(report_path,report)
        if task!='hip':shutil.copy2(folder/(task+'.pt'),bundle/(task+'.pt'))
    hip_candidates={}
    for arch in ('light','heavy'):
        report=json.loads((OUT/('hip_'+arch)/'hip_report.json').read_text(encoding='utf-8'))
        hip_candidates[arch]=next(h['validation_metrics'] for h in report['history'] if h['epoch']==report['best_epoch'])
    roi=max(hip_candidates,key=lambda k:hip_candidates[k]['mean_roi_iou']);mask=max(hip_candidates,key=lambda k:hip_candidates[k]['pixel_dice'])
    shutil.copy2(OUT/('hip_'+roi)/'hip.pt',bundle/'hip.pt');shutil.copy2(OUT/('hip_'+mask)/'hip.pt',bundle/'hip_mask.pt')
    save(OUT/'hip_selection.json',{'validation':hip_candidates,'roi':roi,'mask':mask,'test_used':False})
    clear()
    from .final_spine import train_final_spine
    if not (bundle/'spine_report.json').exists():
        event('training',task='spine');train_final_spine(groups,augmented,protocol,bundle,device)
    hiptrain=[r for r in groups['train'] if r.region!='SPINE'];valid=[r for r in groups['validation'] if r.region!='SPINE']
    scores={}
    for arch in ('light','heavy'):
        clear();folder=OUT/('points_'+arch)
        if not (folder/'report.json').exists():
            event('training',task='hip_points',architecture=arch)
            train_variant({**VARIANTS['shared256'],'architecture':arch,'batch_size':8 if arch=='light' else 2},hiptrain,valid,folder,device,24,patience=6,validation_evaluator=PointSelection(valid),mixed_precision=arch=='heavy')
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'));scores[arch]=min(report['history'],key=lambda h:h['metrics']['selection_loss'])['metrics']
    points=min(scores,key=lambda k:scores[k]['selection_loss']);shutil.copy2(OUT/('points_'+points)/'hip_points.pt',bundle/'hip_points.pt')
    save(OUT/'point_selection.json',{'validation':scores,'selected':points,'test_used':False})
    from .landmark_geometry import fit_geometry_ranges
    save(bundle/'landmark_geometry_bounds.json',fit_geometry_ranges([r for r in hiptrain if not r.source_id]))
    shutil.copy2(OUT/'protocol.json',bundle/'protocol.json');clear();event('trained',bundle=str(bundle))


class PointSelection:
    def __init__(self,records):self.records=records
    def __call__(self,model,config,device):
        from .experiments import evaluate
        from .landmarks import PointDataset
        from .data import collate
        from torch.utils.data import DataLoader
        metrics,rows,loss=evaluate(model,DataLoader(PointDataset(self.records,config['size'],config['crop']),batch_size=8,collate_fn=collate),device,'masked_logit_argmax',config['loss'])
        # PCK already counts missed visible targets; use primary localization metric.
        metrics['selection_loss']=1-metrics['pck_10mm'];return metrics,rows,loss


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','train','evaluate'),default='train');args=parser.parse_args()
    if args.phase=='prepare':prepare();event('prepared')
    elif args.phase=='train':train_all()
    else:
        from .final_evaluation import run
        run()
