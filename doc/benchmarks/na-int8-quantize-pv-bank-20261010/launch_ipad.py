"""Launch a long-running benchmark with stdout and an attached debugger.

Attachment avoids iOS's foreground launch watchdog for this command-line probe.
GPU timestamps are still measured inside the application with no breakpoints.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

def run(device, values, log, timeout=1800, bundle="com.liu.NAFloatPVAttentionExperiment"):
    with tempfile.TemporaryDirectory(prefix='ccv-pv-ipad-') as tmp:
        def processes():
            path=Path(tmp)/'processes.json'
            subprocess.run(['xcrun','devicectl','device','info','processes','--device',device,'--json-output',str(path)],stdout=subprocess.DEVNULL,check=True)
            return {p['processIdentifier']:p.get('executable','') for p in json.loads(path.read_text())['result']['runningProcesses']}
        before=processes()
        cmd=['xcrun','devicectl','device','process','launch','--device',device,'--start-stopped','--terminate-existing','--console',bundle,*map(str,values)]
        with open(log,'w') as output:
            child=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT)
            try:
                pid=None
                for _ in range(30):
                    time.sleep(.2)
                    matches=[pid for pid,path in processes().items() if pid not in before and path.endswith('/NAInt8TuningApp.app/NAInt8TuningApp')]
                    if len(matches)==1: pid=matches[0]; break
                    if child.poll() is not None: break
                if pid is None: raise RuntimeError('No unique benchmark process appeared')
                with open(str(log)+'.lldb','w') as debug:
                    result=subprocess.run(['xcrun','lldb','--batch',
                        '-o','settings set target.preload-symbols false',
                        '-o',f'device select {device}', '-o','script import time; time.sleep(1)',
                        '-o',f'device process attach --pid {pid}', '-o','script time.sleep(4)',
                        '-o','continue','-o','quit'],stdout=debug,stderr=subprocess.STDOUT,timeout=timeout)
                status=child.wait(timeout=30)
                return status if status else result.returncode
            finally:
                if child.poll() is None: child.terminate(); child.wait(timeout=30)

if __name__=='__main__':
    sys.exit(run(sys.argv[1],sys.argv[3:],sys.argv[2]))
