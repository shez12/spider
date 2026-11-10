"""Check custom-hand actuation and passive-joint reference handling."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from spider.robot_model import actuator_qpos_indices, project_joint_equalities

ROOT = Path(__file__).resolve().parents[2]


class YimuIntegrationTest(unittest.TestCase):
    """Preserve the supplied mechanism through SPIDER adapters."""

    def test_adapter_actuation_and_coupling(self):
        """Keep passive DIP joints out of motor commands on both hands."""
        for side in ["right", "left"]:
            with self.subTest(side=side):
                model = mujoco.MjModel.from_xml_path(
                    str(ROOT / f"spider/assets/robots/yimu_v2/{side}.xml")
                )
                self.assertEqual((model.nq, model.nu, model.neq), (27, 23, 4))
                indices = actuator_qpos_indices(model)
                self.assertEqual(len(set(indices)), 23)
                np.testing.assert_allclose(
                    model.actuator_forcerange[6:], np.tile([-0.16, 0.16], (17, 1))
                )
                self.assertAlmostEqual(model.body_mass.sum(), 0.72, places=5)
                batch = np.tile(model.qpos0, (3, 1))
                for finger in range(2, 6):
                    driver = model.joint(f"{side}_{finger}-3").qposadr[0]
                    passive = model.joint(f"{side}_{finger}-4").qposadr[0]
                    self.assertNotIn(passive, indices)
                    batch[:, driver] = [0.1, 0.4, 0.8]
                project_joint_equalities(model, batch)
                for row in batch:
                    np.testing.assert_allclose(
                        project_joint_equalities(model, row.copy()), row
                    )
                for eq in range(model.neq):
                    driver = model.jnt_qposadr[model.eq_obj2id[eq]]
                    passive = model.jnt_qposadr[model.eq_obj1id[eq]]
                    expected = (
                        np.polynomial.polynomial.polyval(
                            batch[:, driver] - model.qpos0[driver],
                            model.eq_data[eq, :5],
                        )
                        + model.qpos0[passive]
                    )
                    np.testing.assert_allclose(batch[:, passive], expected)
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    self.assertGreaterEqual(model.site(f"{side}_{finger}_tip").id, 0)


if __name__ == "__main__":
    unittest.main()
