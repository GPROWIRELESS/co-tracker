"""Run the delivered TL adapter with frozen numerical modules beside this file."""
from pathlib import Path
import sys,runpy
sys.dont_write_bytecode=True
home=Path(__file__).resolve().parent
sys.path.insert(0,str(home))
import tl_geometry,tl_common,tl_stats
for module in (tl_geometry,tl_common,tl_stats):
    assert Path(module.__file__).resolve().parent==home,'Frozen modules must be beside this wrapper'
sys.argv[0]=str(home/'target_locking_run.py')
runpy.run_path(sys.argv[0],run_name='__main__')
