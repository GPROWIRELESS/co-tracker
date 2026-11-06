"""Input-contract regression checks; fixtures are temporary, no dataset writes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import types
import unittest
import uuid
from unittest.mock import patch

import numpy as np

import tl_inputs as inputs


class InputChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("_input_test_"+uuid.uuid4().hex)
        self.root.mkdir()
        self.frames = np.arange(10,14,dtype=np.int64)
        self.ns = np.int64(1750000000000000000)+np.arange(4,dtype=np.int64)*20_000_000
        self.rec = dict(content_id="User1_1_1",participant="User1",task="A",task_num=1,
                        attempt=1,score=50.,status="Completed",round=1)
        self.data = dict(**self.rec,frames=self.frames,timestamp_ns=self.ns,width=1280,height=1024,
                         valid_code=np.array([0,1,2,3]))

    def tearDown(self):
        # Verify the absolute cleanup target is an owned test directory.
        parent = Path(__file__).resolve().parent
        if self.root.resolve().parent != parent or not self.root.name.startswith("_input_test_"):
            raise RuntimeError("Unsafe temporary cleanup path")
        shutil.rmtree(self.root)

    def prediction(self):
        p = np.zeros((4,32,40),dtype=np.float32)
        p[0,0,0] = 1
        p[1,31,39] = 1
        p[2,0,0] = .5
        p[2,31,39] = .5
        p[3,12,7] = 1
        s = np.array([[[0.,0.],[1.,1.],[7/39,12/31]]] * 4,dtype=np.float32)
        path = self.root/"User1_1_1_predictions.npz"
        np.savez_compressed(path,probability=p,samples=s,frame_number=self.frames,
                            frame_index=np.arange(4),timestamp_ns=self.ns,valid_code=self.data["valid_code"])
        return dict(model="all_ABCD",content_id="User1_1_1",path=str(path),
                    prediction_sha256=inputs._sha256(path),prediction_frames=4), p, s

    def test_grid_scale_expected_argmax_and_sampling(self):
        entry, p, s = self.prediction()
        b = inputs.prediction_bundle(entry,self.data,trajectory_seed=42)
        np.testing.assert_allclose(b["expectedxy"][:3],[[0,0],[1280,1024],[640,512]])
        np.testing.assert_allclose(b["argmaxxy"][2],[0,0])
        np.testing.assert_allclose(b["samples_xy"],s.astype(float)*[1280,1024])
        np.testing.assert_array_equal(b["samples"],s)
        np.testing.assert_array_equal(b["sampledxy"],b["samples_xy"][np.arange(4),b["sample_index"]])
        np.testing.assert_array_equal(b["valid_maps"],np.ones(4,dtype=bool))
        self.assertFalse(b["metadata"]["sample_column_temporal_identity"])
        self.assertEqual(b["metadata"]["argmax_tied_frames"],1)

    def test_cache_and_frame_timestamp_validation(self):
        entry,_,_ = self.prediction()
        cache = self.root/"cache"
        first = inputs.prediction_bundle(entry,self.data,cache_dir=cache,include_probability=False)
        second = inputs.prediction_bundle(entry,self.data,cache_dir=cache,include_probability=False)
        self.assertFalse(first["metadata"]["prediction_bundle_cache_hit"])
        self.assertTrue(second["metadata"]["prediction_bundle_cache_hit"])
        np.testing.assert_array_equal(first["sampledxy"],second["sampledxy"])
        shifted = dict(self.data,timestamp_ns=self.ns+1)
        with self.assertRaisesRegex(ValueError,"timestamp_ns mismatch"):
            inputs.prediction_bundle(entry,shifted,cache_dir=cache,include_probability=False)
        with Path(entry["path"]).open("ab") as out:
            out.write(b"tamper")
        with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
            inputs.prediction_bundle(entry,self.data,cache_dir=cache,include_probability=False)

    def test_legacy_reuse_requires_verified_sha_and_indices(self):
        entry,_,_ = self.prediction()
        legacy = self.root/"legacy"
        (legacy/"frames").mkdir(parents=True)
        expected = inputs.prediction_bundle(entry,self.data,include_probability=False)["expectedxy"]
        np.savez_compressed(legacy/"frames/User1_1_1.npz",frame_number=self.frames,
                            timestamp_ns=self.ns,resolution=[1280,1024],xy_all_ABCD=expected)
        with (legacy/"prediction_audit.csv").open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=["content_id","model","observed_sha256","prediction_sha256"])
            writer.writeheader()
            writer.writerow(dict(content_id="User1_1_1",model="all_ABCD",
                                 observed_sha256=entry["prediction_sha256"],prediction_sha256=entry["prediction_sha256"]))
        b = inputs.prediction_bundle(entry,self.data,legacy_cache_dir=legacy,include_probability=False)
        self.assertEqual(b["metadata"]["expectedxy_origin"],"verified legacy frame cache")
        np.savez_compressed(legacy/"frames/User1_1_1.npz",frame_number=self.frames+1,
                            timestamp_ns=self.ns,resolution=[1280,1024],xy_all_ABCD=expected)
        b = inputs.prediction_bundle(entry,self.data,legacy_cache_dir=legacy,include_probability=False)
        self.assertEqual(b["metadata"]["expectedxy_origin"],"normalized probability map mean")

    def test_record_validity_policies_and_raw_state_evidence(self):
        base = self.root/"Active_Data/User1"
        for directory in ("npy","csv","sam","stimuli"):
            (base/directory).mkdir(parents=True)
        dtype = [("Frame_Number","i8"),("Timestamp(ns)","i8"),("gaze_x","f8"),
                 ("gaze_y","f8"),("valid","i4"),("stage","U12"),("point_1_x","f8"),("point_1_y","f8")]
        a = np.zeros(4,dtype=dtype)
        a["Frame_Number"] = self.frames
        a["Timestamp(ns)"] = self.ns[0]
        a["gaze_x"] = .5
        a["gaze_y"] = .25
        a["valid"] = [0,1,2,3]
        a["stage"] = "rr"
        np.save(base/"npy/User1_1_1_gaze.npy",a)
        (base/"csv/User1_1_1_video_ts.csv").write_text("Frame_Number,Timestamp(ns)\n"+"\n".join(f"{f},{t}" for f,t in zip(self.frames,self.ns)))
        (base/"sam/dlc.csv").write_text("fixture")
        (base/"csv/User1_1_1_gaze.csv").write_text("Capture Unix timestamp,Left Status,Right Status,Status,norm_x,norm_y\n"+"\n".join(f"{t},{c},2,2,.5,.25" for t,c in zip(self.ns,range(4))))
        fake = types.SimpleNamespace(dlc_path=lambda *args:base/"sam/dlc.csv",
                                     timeline=lambda *args:(self.frames,self.ns,0),
                                     read_dlc=lambda *args,**kwargs:({k:np.zeros((4,2)) for k in ("l","r")},{k:np.ones(4) for k in ("l","r")}),
                                     video_metadata=lambda *args:(1280,1024,50),normalized_stage=lambda s:s)
        with patch.object(inputs,"_pipeline",return_value=fake):
            d = inputs.read_record(self.rec,self.root,"unused",audit_cache_dir=self.root/"cache")
            strict = inputs.read_record(self.rec,self.root,"unused",gaze_allowed_codes=(3,),audit_cache_dir=self.root/"cache")
            legacy = inputs.read_record(self.rec,self.root,"unused",gaze_allowed_codes=(1,2,3))
        np.testing.assert_array_equal(d["gaze_valid"],[False,False,True,True])
        np.testing.assert_array_equal(strict["gaze_valid"],[False,False,False,True])
        np.testing.assert_array_equal(legacy["gaze_valid"],[False,True,True,True])
        self.assertTrue(np.isfinite(d["gaze_unmasked"]).all())
        self.assertEqual(d["audit"]["raw_left_status_match_n"],4)
        self.assertEqual(d["audit"]["npy_video_timestamp_differences"],3)
        self.assertTrue(strict["audit"]["raw_state_audit_cache_hit"])
        self.assertEqual(strict["audit"]["gaze_allowed_codes"],[3])


if __name__ == "__main__":
    unittest.main()
