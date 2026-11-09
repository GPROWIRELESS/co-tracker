"""Delegate the source and same-new-input audit to the delivered verification.

The complete verification independently checks native NPY coordinates, frame
times, 41 Hz interpolation and both preprocessing implementations on the same
new input. Its output stays in verification/; no raw-CSV-GT audit is invoked.
"""
import importlib.util
from pathlib import Path


def main():
    script=Path(__file__).resolve().parents[2]/'verification'/'audit_preprocess_revision.py'
    if not script.exists():
        script=Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\outputs\gaze_psm_revision_20261106\verification\audit_preprocess_revision.py')
    if not script.exists():
        raise FileNotFoundError('Run the delivered verification/audit_preprocess_revision.py on the new NPY input')
    spec=importlib.util.spec_from_file_location('revision_20261106_source_audit',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.audit_inputs()


if __name__=='__main__':main()
