"""Regression cases for shared Linux memory and simultaneous hall training/evaluation."""
import unittest

from run_interactive_world import admission, scheduled_variants, GIB, TRAIN_ORDER


def worker(variant,phase="train"):
    return dict(variant=variant,phase=phase,active_phase=phase,active=True,first_batch=True,
                training_complete=phase=="evaluate",verified_updates_lower_bound=40000,
                thread_budget=3,tree_memory={"pss_bytes":3*GIB,"uss_bytes":3*GIB})


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        self.host=dict(ram_available_bytes=41*GIB,gpu_free_bytes=86*GIB,cpu_quota=25)
        # Two jobs reported the same concurrent container file-cache high water
        # as their personal incremental peak in the original November ledger.
        self.peaks={k:dict(pss_bytes=3*GIB,uss_bytes=3*GIB,torch_reserved_peak=14*GIB,
                           incremental_working_peak=64*GIB) for k in ("R0C","R1")}
        self.active=[worker("R0C"),worker("R1")]

    def decide(self,name="R2",phase="train"):
        return admission(name,{"phase":phase},self.active,self.peaks,self.host)

    def test_migrated_container_lifetime_drift_is_not_private_demand(self):
        self.assertTrue(self.decide()[0])

    def test_insufficient_real_ram_or_gpu_keeps_job_queued(self):
        self.host["ram_available_bytes"]=8*GIB
        self.assertFalse(self.decide()[0])
        self.host["ram_available_bytes"]=41*GIB
        self.host["gpu_free_bytes"]=10*GIB
        self.assertFalse(self.decide()[0])

    def test_eval_does_not_consume_one_of_three_training_slots(self):
        self.active.append(worker("VR2","evaluate"))
        self.host["ram_available_bytes"]=90*GIB
        self.assertTrue(self.decide()[0])
        self.active[-1]=worker("R3")
        self.assertFalse(self.decide()[0])

    def test_attributed_startup_pressure_still_reserves_headroom(self):
        self.peaks["R1"]["phase_peaks"]={"train":{"startup_pressure_bytes":34*GIB,"pss_bytes":3*GIB,
                                                 "gpu_peak_bytes":14*GIB}}
        self.assertFalse(self.decide()[0])

    def test_hall_mesh_retains_volume_headroom_and_protects_a_second_volume(self):
        self.assertFalse(self.decide("R3","evaluate")[0])
        self.host["ram_available_for_mesh_bytes"]=82*GIB
        self.assertTrue(self.decide("R3","evaluate")[0])
        self.active.append(worker("R0C","evaluate"))
        self.assertFalse(self.decide("R3","evaluate")[0])

    def test_once_only_verified_handoff_prefers_r5_then_restores_r3(self):
        rows={k:{"log":None,"training_complete":False} for k in TRAIN_ORDER}
        rows["R3"]["checkpoint_handoff"]=True
        order=scheduled_variants(rows)
        self.assertLess(order.index("R5"),order.index("R3"))
        rows["R5"]["log"]="started"
        order=scheduled_variants(rows)
        self.assertLess(order.index("R3"),order.index("R5"))


if __name__=="__main__":
    unittest.main()
