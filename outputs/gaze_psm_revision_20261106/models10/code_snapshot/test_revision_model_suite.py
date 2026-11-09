from gaze_psm_modes import mode_rows
import csv
import tempfile
import contextlib
import shutil
import uuid
import unittest
from pathlib import Path
import numpy as np
from analyze_model_xcorr_dlc_v2 import SOURCES, selected_rule_outputs
from gaze_psm_reliability_v2 import peak_features


@contextlib.contextmanager
def checked_test_directory():
    # Ordinary mkdir avoids restrictive temporary-directory ACLs on Windows.
    parent=Path(__file__).resolve().parent
    path=parent/('_selected_rule_check_'+uuid.uuid4().hex)
    path.mkdir()
    try:
        yield str(path)
    finally:
        if path.resolve().parent!=parent or not path.name.startswith('_selected_rule_check_'):
            raise RuntimeError('Unsafe temporary cleanup path')
        shutil.rmtree(path)


class SelectedRuleTests(unittest.TestCase):
    def test_gt_side_is_shared_and_missing_gt_is_preserved(self):
        curves=[];rows=[]
        for cid,missing in [('User1_1_1',False),('User2_1_1',True)]:
            for source in SOURCES:
                for side in ('l','r'):
                    # GT wins left, all models win right: legacy changes instrument.
                    r=np.array([.1,.8,.2]) if (source=='gt')==(side=='l') else np.array([.1,.2,.5])
                    if source=='gt' and missing:r=np.full(3,np.nan)
                    c=dict(variant='primary',content_id=cid,user=cid.split('_')[0],task='A',
                           side=side,signal='position',stage='reach ring',mode='PSM in stage',
                           source=source,lag_s=np.array([-1.,0.,1.]),r=r)
                    curves.append(c);rows.append(dict(c,**peak_features(c),lag_identifiable=False))
        with checked_test_directory() as tmp:
            selected_rule_outputs(curves,rows,tmp,0)
            with (Path(tmp)/'selected_rule_paired_metrics.csv').open(encoding='utf-8-sig') as f:
                output=list(mode_rows(csv.DictReader(f)))
        self.assertEqual(len(output),20)
        valid=[r for r in output if r['content_id']=='User1_1_1']
        shared=[r for r in valid if r['side']=='gt_selected_side']
        legacy=[r for r in valid if r['side']=='independent_max_side_legacy']
        self.assertTrue(all(r['gt_side']==r['model_side']=='l' and r['same_side']=='True' for r in shared))
        self.assertTrue(all(r['gt_side']=='l' and r['model_side']=='r' and r['same_side']=='False' for r in legacy))
        missing=[r for r in output if r['content_id']=='User2_1_1' and r['side']=='gt_selected_side']
        self.assertEqual(len(missing),5)
        self.assertTrue(all(r['common_input']=='False' and not r['gt_side'] and not r['model_side'] for r in missing))


if __name__=='__main__':unittest.main()
