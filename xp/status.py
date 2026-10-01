import json
from pathlib import Path
import time

from common import BASE, alive


def main():
    file=BASE/'heartbeat.json'
    if file.exists():
        h=json.loads(file.read_text())
        h['runner_running']=alive(h.get('owner'))
        h['heartbeat_age_seconds']=round(time.time()-h['timestamp'],1)
        print(json.dumps(h,indent=2))
    else:
        print('Runner stopped / no heartbeat yet')
    print('STOP:', 'PRESENT, remove manually before resume' if (BASE/'STOP').exists() else 'absent')
    for path in sorted((BASE/'logs').glob('*.log')):
        print('\n'+str(path.relative_to(BASE)))
        with path.open('rb') as f:
            f.seek(0,2)
            size=f.tell()
            f.seek(max(0,size-8192))
            lines=f.read().decode(errors='replace').splitlines()
        print('\n'.join(lines[-8:]))


if __name__=='__main__':
    main()
