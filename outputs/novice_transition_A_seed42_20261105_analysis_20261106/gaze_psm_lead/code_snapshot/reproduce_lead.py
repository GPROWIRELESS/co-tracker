"""Reproduce coordinate-only lead results and verify the zero-horizon control."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,sys
from pathlib import Path
import lead_psm_postprocess_driver as driver
import lead_psm_supplement as finish
import lead_psm_zero_horizon_fix as zero
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
driver.TOOLS=Path(__file__).resolve().parent
sys.argv=[__file__,'--root',str(args.root),'--workers',str(args.workers)];driver.main()
sys.argv=[__file__,'--root',str(args.root)];finish.main()
sys.argv=[__file__,'--root',str(args.root)];zero.main()
