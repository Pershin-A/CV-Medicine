import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

def settings():
    return Path(os.getenv('DXA_SERVICE_DATA',str(ROOT/'service_data'))).resolve()

class Store:
    def __init__(self,path=None):
        self.root=Path(path or settings());self.root.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS objects (kind TEXT, id TEXT, data TEXT, PRIMARY KEY(kind,id))')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, action TEXT, status TEXT, payload TEXT, result TEXT, error TEXT, created REAL, updated REAL, request_key TEXT UNIQUE)')
    def connect(self):
        db=sqlite3.connect(self.root/'state.sqlite',timeout=30);db.row_factory=sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA busy_timeout=30000');return db
    def put(self,kind,id,value):
        with self.connect() as db:db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)',(kind,id,json.dumps(value,ensure_ascii=False)))
    def get(self,kind,id):
        with self.connect() as db:r=db.execute('SELECT data FROM objects WHERE kind=? AND id=?',(kind,id)).fetchone()
        if not r:raise KeyError(f'{kind}/{id}')
        return json.loads(r[0])
    def all(self,kind):
        with self.connect() as db:rows=db.execute('SELECT data FROM objects WHERE kind=? ORDER BY id',(kind,)).fetchall()
        return [json.loads(r[0]) for r in rows]
    def job(self,id):
        with self.connect() as db:r=db.execute('SELECT * FROM jobs WHERE id=?',(id,)).fetchone()
        if not r:raise KeyError(id)
        result=dict(r)
        for key in ('payload','result'):result[key]=json.loads(result[key]) if result[key] else None
        return result
    def create_job(self,action,payload,request_id=None):
        key=f'{action}:{request_id}' if request_id else None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if key:
                old=db.execute('SELECT id,payload FROM jobs WHERE request_key=?',(key,)).fetchone()
                if old:
                    if json.loads(old['payload'])!=payload:raise ValueError('request_id already used with another payload')
                    return old['id']
            id=uuid.uuid4().hex;now=time.time()
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',(id,action,'queued',json.dumps(payload),None,None,now,now,key))
        return id
    def update(self,id,status=None,result=None,error=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=COALESCE(?,status), result=COALESCE(?,result), error=COALESCE(?,error), updated=? WHERE id=?',
                       (status,json.dumps(result,ensure_ascii=False) if result is not None else None,error,time.time(),id))
    def claim(self,actions):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            r=db.execute('SELECT id FROM jobs WHERE status="queued" AND action IN ('+','.join('?' for _ in actions)+') ORDER BY created LIMIT 1',actions).fetchone()
            if r:db.execute('UPDATE jobs SET status="running",updated=? WHERE id=?',(time.time(),r[0]))
        return self.job(r[0]) if r else None
    def artifact(self,id,name):
        # Filenames are server-owned; callers cannot traverse the storage root.
        if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Invalid artifact name')
        if not id.isalnum():raise ValueError('Invalid result ID')
        return self.root/'results'/id/name

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def allowed_path(raw):
    path=Path(raw).resolve()
    roots=[PROJECT.resolve(),settings()]
    roots.extend(Path(r).resolve() for r in os.getenv('DXA_ALLOWED_ROOTS','').split(';') if r)
    if not any(path==r or path.is_relative_to(r) for r in roots):raise ValueError('Path is outside DXA_ALLOWED_ROOTS')
    if not path.exists():raise ValueError(f'Path does not exist: {path}')
    return path
