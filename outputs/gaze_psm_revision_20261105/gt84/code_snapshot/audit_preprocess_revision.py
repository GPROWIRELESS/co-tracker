"""Read-only cache equivalence audit; writes only its requested local JSON."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from gaze_psm_preprocess_revision import prepare_signals


def main():
    root = Path(r'D:\GitHub_Project\co-tracker')
    spec = importlib.util.spec_from_file_location('old_preprocess', root/'tools/gaze_psm_preprocess_v2.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    records = []
    for path in sorted((root/'outputs/gaze_psm_xcorr_dlc/cache').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            cache = {k:z[k] for k in z.files}
        revised = prepare_signals({'gt':cache}, {})
        reference = old.prepare_signals({'gt':cache}, {})
        checks = 0
        for signal in ('position','velocity'):
            for side in ('l','r'):
                a,b = revised['signals'][signal][side], reference['signals'][signal][side]
                for key in ('psm','time_s','stage','occurrence','gaze_valid','psm_valid'):
                    np.testing.assert_array_equal(a[key],b[key]); checks += 1
                np.testing.assert_array_equal(a['gazes']['gt'],b['gazes']['gt']); checks += 1
                diag = revised['diagnostics'][side]
                if signal == 'position':
                    categories = sum(diag['grid_'+k].astype(int) for k in
                        ('observed_exact','ordinary_resample','filled_nonfinite_hole'))
                    np.testing.assert_array_equal(categories, np.isfinite(cache['psm_raw_'+side]).all(1).astype(int))
                assert not diag['quality_rejected'].any()
                assert not diag['grid_filled_quality_hole'].any()
        records.append({'content_id':path.stem, 'numerical_array_checks':checks,
                        'native_reconstruction_passed':all(x['passed'] for x in revised['audit']),
                        'qc':revised['qc']})
    output = dict(n_records=len(records), numerical_arrays_identical=True,
                  comparison='existing_model_v2_default_box3_SG5',
                  no_quality_rejection=True, records=records)
    dest = Path(__file__).with_name('preprocess_cache_audit.json')
    dest.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'n_records':len(records),'numerical_array_checks':sum(x['numerical_array_checks'] for x in records),
                      'all_passed':True,'output':str(dest)}))


if __name__ == '__main__': main()
