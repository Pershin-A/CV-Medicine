"""Backend helper: submit multipart DICOM prediction or an explicit staged-fit JSON."""
import argparse,json,time
from contextlib import ExitStack
from pathlib import Path
import requests

def wait(base,job_id):
    offset=0
    while True:
        response=requests.get(base+f'/v1/jobs/{job_id}',timeout=30);response.raise_for_status();job=response.json()
        logs=requests.get(base+f'/v1/jobs/{job_id}/logs',params={'offset':offset},timeout=30);logs.raise_for_status()
        for event in logs.json()['events']:print(event['message'],flush=True)
        offset=logs.json()['next_offset']
        if job['status'] not in ('queued','running'):return job
        time.sleep(1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8765');sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('predict');p.add_argument('dicoms',nargs='+',type=Path);p.add_argument('--output',type=Path,default=Path('prediction_export'))
    p=sub.add_parser('partial_fit');p.add_argument('config',type=Path)
    args=parser.parse_args();base=args.url.rstrip('/')
    if args.command=='predict':
        with ExitStack() as stack:
            files=[('files',(p.name,stack.enter_context(p.open('rb')),'application/dicom')) for p in args.dicoms]
            response=requests.post(base+'/v1/model/predict/upload',files=files,timeout=120)
        response.raise_for_status();job=wait(base,response.json()['job_id']);args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
        if job['status'] in ('completed','completed_with_errors'):
            response=requests.get(base+job['result']['download_url'],timeout=120);response.raise_for_status();(args.output/'predictions.zip').write_bytes(response.content)
        print(json.dumps(job.get('result'),ensure_ascii=False,indent=2))
    else:
        response=requests.post(base+'/v1/model/partial_fit',json=json.loads(args.config.read_text(encoding='utf-8')),timeout=120)
        response.raise_for_status();print(json.dumps(wait(base,response.json()['job_id']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
