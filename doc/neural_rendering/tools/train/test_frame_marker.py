# SPDX-License-Identifier: Apache-2.0
"""Reject legacy/mixed frame dumps before torch training can consume them."""
import unittest
import dataset


class FrameMarkerTest(unittest.TestCase):
    def test_uv_map_matches_render_frame(self):
        dataset.require_render_uv_layer('UVMap', 'UVMap')
        for requested, render in [('other', 'UVMap'), ('UVMap', None), ('UVMap', 'automap')]:
            with self.subTest(requested=requested, render=render), self.assertRaises(dataset.DumpError):
                dataset.require_render_uv_layer(requested, render)

    def test_current_frame(self):
        dataset.require_uv_tangent_frame([{'frame': 'uv_tangent'}])

    def test_legacy_or_mixed_frames(self):
        for meta in ({}, {'frame': 'edge'}, {'frame': 'world'}):
            with self.subTest(meta=meta), self.assertRaises(dataset.DumpError):
                dataset.require_uv_tangent_frame([{'frame': 'uv_tangent'}, meta])


if __name__ == '__main__':
    unittest.main()
