"""Tests of weighting, complete support, exact block nulls and deterministic resampling."""
import unittest
import numpy as np
from gaze_psm_statistics_v2 import summarize_pairs, group_curves, exact_model_comparisons, identity_reference, paired_group_differences, OBSERVED


def pair(user, stage='one', model='m1', value=.1, cid=None, common=True):
    return dict(variant='primary',task='A',side='l',signal='position',stage=stage,mode='PSM in stage',
        model=model,user=user,content_id=cid or user,curve_mae=value,delta_lag_s=value,
        abs_lag_difference_s=abs(value),input_eligible=common,common_input=common,
        gt_localizable=False,model_localizable=False,gt_status='broad_uncertainty',model_status='boundary_limited')


def curve(user, stage='one', source='gt', value=.1, finite=True):
    rr=np.array([value-.05,value,value-.03])
    if not finite: rr[0]=np.nan
    return dict(variant='primary',task='A',side='l',signal='position',stage=stage,mode='PSM in stage',
        user=user,content_id=user,source=source,lag_s=np.array([-1.,0.,1.]),r=rr)


class StatisticsTests(unittest.TestCase):
    def test_stage_then_person_equal(self):
        data=[pair('a','s1',value=0),*[pair('a','s2',value=2,cid='a'+str(i)) for i in range(3)],pair('b','s1',value=10)]
        got=[x for x in summarize_pairs(data,bootstrap=20) if x['stage']==OBSERVED][0]
        self.assertEqual(got['participant_equal_curve_mae'],5.5)
        self.assertEqual(got['n_conditions'],5)
        self.assertEqual(got['n_model_localizable'],0)
        self.assertEqual(got['model_boundary_rate'],1.)

    def test_common_input_only_not_peaks(self):
        data=[pair('a',value=.1),pair('b',value=.3),pair('c',value=.8,common=False)]
        got=summarize_pairs(data,bootstrap=10)[0]
        self.assertEqual(got['n_conditions'],2)
        self.assertEqual(got['input_coverage'],2/3)
        self.assertAlmostEqual(got['participant_equal_curve_mae'],.2)

    def test_shared_model_resample_draws(self):
        data=[pair(str(u),model=m,value=u*.01+(1 if m=='m2' else 0)) for u in range(5) for m in ['m1','m2']]
        s=[r for r in summarize_pairs(data,100,19) if r['stage']==OBSERVED]
        self.assertAlmostEqual(s[1]['curve_mae_resample_low']-s[0]['curve_mae_resample_low'],1.)
        self.assertAlmostEqual(s[1]['curve_mae_resample_high']-s[0]['curve_mae_resample_high'],1.)

    def test_group_fixed_support(self):
        cs=[curve(str(i),value=.1+i*.02) for i in range(4)]+[curve('bad',finite=False)]
        summary,long=group_curves(cs,bootstrap=20)
        self.assertEqual(summary[0]['n_stable_users'],4)
        self.assertEqual(summary[0]['n_total_users'],5)
        self.assertEqual(len(long),3)
        self.assertEqual(summary[0]['peak_lag_s'],0.)

    def test_group_reproducible(self):
        cs=[curve(str(i),value=.1+i*.02) for i in range(5)]
        a=group_curves(cs,bootstrap=30,seed=12)
        b=group_curves(list(reversed(cs)),bootstrap=30,seed=12)
        self.assertEqual(a,b)

    def test_exact_swap_small_n_floor(self):
        rows=[pair(str(i),model=m,value=.1+(m=='m2')*.2) for i in range(5) for m in ['m1','m2']]
        out=exact_model_comparisons(rows)
        self.assertEqual(len(out),2)
        for r in out:
            self.assertEqual(r['n_permutations'],32)
            self.assertEqual(r['p_exact'],.0625)
            self.assertGreaterEqual(r['p_holm'],r['p_exact'])
            self.assertAlmostEqual(r['mean_curve_mae_b_minus_a'],.2)

    def test_identity_blocks_and_missing_stage(self):
        cs=[curve(str(i),stage=s,source=src,value=.1+i*.2) for i in range(3) for s in ['one','two'] for src in ['gt','m1']]
        out=identity_reference(cs)[0]
        self.assertEqual(out['n_permutations'],6)
        self.assertEqual(out['observed_curve_mae'],0.)
        self.assertEqual(out['empirical_lower_tail_rank'],1/6)
        cs=[c for c in cs if not(c['user']=='0' and c['stage']=='two' and c['source']=='m1')]
        out=identity_reference(cs)[0]
        self.assertEqual(out['n_common_stages'],1)
        self.assertIn('one',out['stages_json'])

    def test_paired_group_shared_users_and_draws(self):
        cs=[]
        for u in range(4):
            a=curve(str(u),source='gt');a['r']=np.array([.0,.5,.1])
            b=curve(str(u),source='m1');b['r']=np.array([.0,.1,.5])
            if u==3:b['r'][0]=np.nan
            cs.extend([a,b])
        row=paired_group_differences(cs,bootstrap=20)[0]
        self.assertEqual(row['n_shared_users'],3)
        self.assertEqual(row['delta_group_peak_lag_s'],1.)
        self.assertEqual(row['delta_group_peak_lag_resample_low'],1.)
        self.assertEqual(row['delta_group_peak_lag_resample_high'],1.)
        self.assertEqual(row['model_bootstrap_boundary_frequency'],1.)

    def test_explicit_boundary_field_overrides_status(self):
        data=[pair('a'),pair('b')]
        for r in data:r.update(gt_boundary=True,model_boundary=False,model_status='boundary_limited')
        row=summarize_pairs(data,bootstrap=5)[0]
        self.assertEqual(row['gt_boundary_rate'],1.)
        self.assertEqual(row['model_boundary_rate'],0.)

    def test_identity_lag_reference_uses_same_person_maps(self):
        cs=[]
        for i in range(3):
            for src in ['gt','m1']:
                c=curve(str(i),source=src);c['r']=np.array([.1,.1,.1]);c['r'][i]=.5
                cs.append(c)
        row=identity_reference(cs)[0]
        self.assertEqual(row['n_permutations'],6)
        self.assertEqual(row['observed_abs_lag_difference_s'],0.)
        self.assertGreater(row['null_mean_abs_lag_difference_s'],0.)
        self.assertEqual(row['lag_empirical_lower_tail_rank'],1/6)

    def test_no_identity_profile_returns_status(self):
        cs=[curve(str(i),source=s,finite=(s=='gt')) for i in range(3) for s in ['gt','m1']]
        out=identity_reference(cs)[0]
        self.assertEqual(out['status'],'insufficient_common_profile')
        self.assertEqual(out['n_permutations'],0)

if __name__=='__main__':unittest.main()
