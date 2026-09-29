"""Start the API and two workers with one local command."""
import argparse
import os
import subprocess
import sys
import threading
import signal
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--data-dir',type=Path);p.add_argument('--no-workers',action='store_true')
    p.add_argument('--bootstrap-bundle',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/final_20260929/bundle',
                   help='Register existing final weights if storage has no active model; never retrain at startup')
    args=p.parse_args()
    if args.data_dir:os.environ['DXA_SERVICE_DATA']=str(args.data_dir.resolve())
    from .store import Store
    store=Store()
    try:store.get('config','active_model')
    except KeyError:
        if args.bootstrap_bundle.is_dir():
            from .training import register
            register(store,dict(checkpoints=str(args.bootstrap_bundle.resolve()),protocol=str(args.bootstrap_bundle.parent/'protocol.json'),epochs=None,activate=True))
    children=[]
    handles=[]
    readers=[]
    os.environ['DXA_API_CONTROLLER_PID']=str(os.getpid())
    def pump(stream,handle):
        for line in stream:
            handle.write(line.encode('utf-8'));print(line,end='',flush=True)
    try:
        if not args.no_workers:
            for role in ('cpu','gpu'):
                log=store.root/'logs'/f'worker_{role}.log';log.parent.mkdir(parents=True,exist_ok=True)
                handle=log.open('ab',buffering=0);handles.append(handle)
                child=subprocess.Popen([sys.executable,'-X','utf8','-m','dxa_project.service.worker','--role',role,'--device',args.device],
                                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                children.append(child)
                reader=threading.Thread(target=pump,args=(child.stdout,handle),daemon=True);reader.start();readers.append(reader)
        import uvicorn
        uvicorn.run('dxa_project.service.app:app',host=args.host,port=args.port,workers=1)
    finally:
        for role in ('cpu','gpu'):
            pid_file=store.root/f'worker_{role}_pid.json'
            try:
                item=json.loads(pid_file.read_text(encoding='utf-8'))
                if item.get('controller_pid')==str(os.getpid()):os.kill(item['pid'],signal.SIGTERM)
            except (FileNotFoundError,ProcessLookupError,OSError):pass
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill()
        for reader in readers:reader.join(timeout=2)
        for handle in handles:handle.close()

if __name__=='__main__':main()
