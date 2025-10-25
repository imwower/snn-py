import unittest

from snn_py.seed import SeedManager


def test_same_seed_same_sequence():
    sm1 = SeedManager(base=123)
    sm2 = SeedManager(base=123)
    r1 = [sm1.rng("gate").random() for _ in range(5)]
    r2 = [sm2.rng("gate").random() for _ in range(5)]
    assert r1 == r2


def test_different_names_different_streams():
    sm = SeedManager(base=456)
    a = [sm.rng("gate").random() for _ in range(3)]
    b = [sm.rng("segments").random() for _ in range(3)]
    assert a != b  # 概率极低相同


def test_none_base_still_deterministic_per_process():
    sm = SeedManager(base=None)
    # None base 时，仍应返回可用 RNG，只要 name 相同就一致（在同一进程中缓存）
    x = [sm.rng("x").random() for _ in range(2)]
    y = [sm.rng("x").random() for _ in range(2)]
    assert x != y  # 第二次接着抽，不应重置


class SeedManagerTests(unittest.TestCase):
    def test_same_seed_same_sequence(self) -> None:
        test_same_seed_same_sequence()

    def test_different_names_different_streams(self) -> None:
        test_different_names_different_streams()

    def test_none_base_still_deterministic_per_process(self) -> None:
        test_none_base_still_deterministic_per_process()
