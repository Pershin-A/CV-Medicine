"""Run from Windows Task Scheduler so closing a terminal does not kill training."""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from . import retrain_reviewed as r
from .heavy_continuation import stop_tree
from dxa_project.service.worker import worker_lock

def main():
    with worker_lock(r.OUT/'unattended_heavy.lock'):
        folder=r.OUT/'heavy_stratified'
        archive=folder/'interrupted'/datetime.now().strftime('%Y%m%d_%H%M%S')
        repeated=[]
        for task in ('router','spine','hip','artifact','hip_points'):
            report=folder/('points_report.json' if task=='hip_points' else f'{task}_report.json')
            if report.exists():continue
            candidates=[folder/f'{task}.pt',folder/f'{task}_best.pt',folder/f'{task}_history.json']
            if any(p.exists() for p in candidates):
                archive.mkdir(parents=True,exist_ok=True)
                for p in candidates:
                    if p.exists():shutil.copy2(p,archive/p.name)
                repeated.append(task)
        # Old checkpoints do not include optimizer state, so only unfinished
        # modules restart. Completed module reports/checkpoints are reused.
        timing=r.OUT/'heavy_timing.json'
        first_started=timing.stat().st_mtime if timing.exists() else time.time()
        deadline=first_started+12*3600
        r.save(r.OUT/'unattended_heavy.json',{'stage':'running','started':time.time(),
               'deadline':deadline,'restart_modules':repeated,'archive':str(archive),
               'launch_method':'Windows Task Scheduler, current interactive user'})
        with (r.OUT/'heavy_unattended.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-m','dxa_project.geometry_ml.heavy_continuation','--run'],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT)
            watcher=subprocess.Popen([sys.executable,'-m','dxa_project.geometry_ml.report_reviewed','--watch'],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                if time.time()>deadline:
                    stop_tree(child.pid);r.event('failed',error='12-hour elapsed limit reached; checkpoints retained');break
                time.sleep(10)
            result=child.poll()
            r.save(r.OUT/'unattended_heavy.json',{'stage':'complete' if result==0 else 'failed',
                   'exit_code':result,'deadline':deadline,'restart_modules':repeated,'archive':str(archive)})

if __name__=='__main__':
    try:main()
    except Exception as e:r.event('failed',error=repr(e));raise
