"""Durable job worker. Run CPU and GPU roles in separate processes."""
import argparse
import contextlib
import json
import os
import time
import uuid
import logging
from pathlib import Path
from .store import Store,ROOT

logger=logging.getLogger('dxa.prediction')

def external_training_busy():
    for version in ('retrained_20260929','final_20260929'):
        path=ROOT/'outputs'/version/'status.json'
        try:
            stage=json.loads(path.read_text(encoding='utf-8'))['stage']
            logs=list(path.parent.glob('training*.log'))+list(path.parent.glob('*/*history.json'))
            changed=max([path.stat().st_mtime,*[p.stat().st_mtime for p in logs]])
            if stage not in ('complete','failed','trained','prepared') and time.time()-changed<600:return True
        except (FileNotFoundError,KeyError,json.JSONDecodeError):continue
    return False

@contextlib.contextmanager
def worker_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

class Worker:
    def __init__(self,store,device='auto'):
        self.store=store;self.device=device;self.engine=None;self.engine_version=None
    def run(self,job):
        try:
            if job['action'] in ('augmentation','enqueue'):
                from .data_ops import augment
                augment(self.store,job,job['action']=='enqueue')
            elif job['action']=='partial_fit':
                import torch
                from .training import partial_fit
                device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if self.device=='auto' else self.device)
                self.engine=None
                partial_fit(self.store,job,device)
            else:self.predict(job)
        except Exception as e:
            import traceback
            folder=self.store.root/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
            (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
            self.store.update(job['id'],'failed',error=f'{type(e).__name__}: {e}')
    def predict(self,job):
        from .training import model
        from .inference import InferenceEngine,save_prediction
        from dxa_project.geometry_ml.predict import _load_model
        import torch
        base=model(self.store,job['payload']['model_version'])
        if self.engine_version!=base['version'] or self.engine is None:
            _load_model.cache_clear();torch.cuda.empty_cache()
            self.engine=InferenceEngine(Path(base['path']),self.device)
            self.engine_version=base['version']
        files=job['payload']['paths'];results=[];errors=[];times={}
        log_path=self.store.root/'jobs'/job['id']/'prediction.jsonl'
        log_path.parent.mkdir(parents=True,exist_ok=True)
        def progress(index,path,status,error=None):
            message=f'Фотография {index}/{len(files)}: '+{'started':'обработка','completed':'готово','failed':'ошибка'}[status]
            entry={'time':time.time(),'job_id':job['id'],'image_index':index,'total':len(files),
                   'file':path.name,'status':status,'message':message}
            if error is not None:entry['error']=error
            with log_path.open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
            logger.info('%s — %s',message,path.name)
            return entry
        # Bound memory for decoded files and dense geometry.
        for offset in range(0,len(files),8):
            chunk=[Path(p) for p in files[offset:offset+8]]
            for index,path in enumerate(chunk,offset+1):
                event=progress(index,path,'started')
                self.store.update(job['id'],result=dict(result_ids=results,errors=errors,processed=offset,total=len(files),current_image=event))
            started=time.perf_counter()
            try:
                values=self.engine.predict(chunk)
                if len(values)!=len(chunk):raise ValueError('Prediction count differs from input count')
            except Exception:
                values=[]
                for path in chunk:
                    try:values.append(self.engine.predict([path])[0])
                    except Exception as e:values.append(None);errors.append({'path':str(path),'error':str(e)})
            forward_seconds=(time.perf_counter()-started)/len(chunk)
            for index,(path,result) in enumerate(zip(chunk,values),offset+1):
                saved=time.perf_counter()
                if result is None:
                    progress(index,path,'failed',next((e['error'] for e in errors if e['path']==str(path)),'Prediction failed'))
                else:
                    try:
                        id=uuid.uuid4().hex;save_prediction(self.store,id,result,path,base['version']);results.append(id)
                    except Exception as e:
                        errors.append({'path':str(path),'error':str(e)});progress(index,path,'failed',str(e))
                    else:progress(index,path,'completed')
                times[str(path)]=forward_seconds+time.perf_counter()-saved
            self.store.update(job['id'],result=dict(result_ids=results,errors=errors,processed=min(offset+8,len(files)),total=len(files),batch_sizes=self.engine.batch_sizes))
        from .deliverables import finish
        delivery=finish(self.store,job['id'],files,results,errors,times)
        self.store.update(job['id'],'completed_with_errors' if errors or any(r['processing_status']=='Failure' for r in delivery['table']) else 'completed',dict(result_ids=results,errors=errors,processed=len(files),total=len(files),batch_sizes=self.engine.batch_sizes,memory_profiles=getattr(self.engine,'memory_profiles',{}),**delivery))

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s',force=True)
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('cpu','gpu','all'),default='all');p.add_argument('--device',choices=('auto','cpu','cuda'),default=os.getenv('DXA_DEVICE','auto'));p.add_argument('--once',action='store_true')
    args=p.parse_args();store=Store();worker=Worker(store,args.device)
    import torch
    torch.set_num_threads(int(os.getenv('DXA_CPU_THREADS','2')))
    actions={'cpu':('augmentation','enqueue'),'gpu':('predict','partial_fit'),'all':('augmentation','enqueue','predict','partial_fit')}[args.role]
    # GPU/all share a lock so accidental second workers cannot duplicate GPU work.
    name='cpu' if args.role=='cpu' else 'gpu'
    with worker_lock(store.root/f'{name}.lock'):
        from .store import write_json
        write_json(store.root/f'worker_{name}_pid.json',{'pid':os.getpid(),'controller_pid':os.getenv('DXA_API_CONTROLLER_PID')})
        with store.connect() as db:
            db.execute('UPDATE jobs SET status="interrupted",error="Worker stopped; create a new request to restart" WHERE status="running" AND action IN ('+','.join('?' for _ in actions)+')',actions)
        for row in store.all('queue'):
            if row['status']=='reserved' and store.job(row['training_job'])['status']=='interrupted':
                row['status']='ready';store.put('queue',row['id'],row)
        while True:
            ready_actions=actions
            if args.device!='cpu' and external_training_busy():ready_actions=tuple(a for a in actions if a not in ('predict','partial_fit'))
            job=store.claim(ready_actions) if ready_actions else None
            if job:worker.run(job)
            elif not args.once:time.sleep(1)
            if args.once:break

if __name__=='__main__':main()
