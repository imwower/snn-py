import unittest

from snn_py.seed import SeedManager


class SeedManagerTests(unittest.TestCase):
    def test_same_seed_same_sequence(self) -> None:
        sm1 = SeedManager(base=123)
        sm2 = SeedManager(base=123)
        r1 = [sm1.rng("gate").random() for _ in range(5)]
        r2 = [sm2.rng("gate").random() for _ in range(5)]
        self.assertEqual(r1, r2)
        self.assertEqual(sm1.seed_for("gate"), sm2.seed_for("gate"))

    def test_different_names_different_streams(self) -> None:
        sm = SeedManager(base=456)
        a = [sm.rng("gate").random() for _ in range(3)]
        b = [sm.rng("segments").random() for _ in range(3)]
        self.assertNotEqual(a, b)  # 概率极低相同
        seeds = sm.describe()
        self.assertIn("gate", seeds)
        self.assertIn("segments", seeds)
        self.assertNotEqual(seeds["gate"], seeds["segments"])

    def test_none_base_still_deterministic_per_process(self) -> None:
        sm = SeedManager(base=None)
        # None base 时，仍应返回可用 RNG，只要 name 相同就一致（在同一进程中缓存）
        x = [sm.rng("x").random() for _ in range(2)]
        y = [sm.rng("x").random() for _ in range(2)]
        self.assertNotEqual(x, y)  # 第二次接着抽，不应重置
        # 同个实例重复获取应该继续同一流
        self.assertGreater(sm.rng("x").random(), 0.0)
