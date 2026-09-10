#!/usr/bin/env python3
import unittest
from validate_host_contract_release import validate


class HostContractReleaseTests(unittest.TestCase):
    def call(self, **overrides):
        context = dict(event="push", ref_type="tag", ref_name="kapsl-managed-abi-v0.1.0",
                       publish=False, on_main=True)
        context.update(overrides)
        return validate("kapsl-managed-abi", "0.1.0", **context)

    def test_exact_main_merged_tag_can_publish(self):
        self.assertTrue(self.call())
        self.assertTrue(self.call(event="workflow_dispatch", publish=True))

    def test_branch_checks_cannot_publish(self):
        self.assertFalse(self.call(event="workflow_dispatch", ref_type="branch", ref_name="feature/contract"))
        with self.assertRaises(ValueError):
            self.call(event="workflow_dispatch", ref_type="branch", ref_name="main", publish=True)

    def test_prerelease_mismatch_unmerged_and_foreign_event_fail(self):
        for changes in (
            dict(ref_name="kapsl-managed-abi-v0.1.0-beta.1"),
            dict(ref_name="kapsl-kv-abi-v0.1.0"),
            dict(ref_name="kapsl-managed-abi-v0.1.1"),
            dict(on_main=False), dict(event="pull_request", publish=True),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(**changes)

    def test_only_allowed_contracts_and_stable_versions_are_supported(self):
        for package, version in (("arbitrary-package", "1.0.0"), ("kapsl-kv-abi", "0.6.1-rc.1"),
                                 ("kapsl-kv-abi", "00.6.1")):
            with self.assertRaises(ValueError):
                validate(package, version, event="workflow_dispatch", ref_type="branch",
                         ref_name="main", publish=False, on_main=True)


if __name__ == "__main__":
    unittest.main()
