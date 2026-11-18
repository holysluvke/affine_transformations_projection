"""Проверки попиксельного Phong/toon-шейдинга без запуска Tkinter."""

import unittest

from geometry import create_polyhedron
from zbuffer import ZBuffer, render_scene as legacy_render_scene
from phong_toon import (
    render_scene, toon_band, normalize_vector, phong_toon_color,
    transform_normal_to_view,
)
import transforms


class PhongToonTests(unittest.TestCase):
    def _render_triangle(self, normal, light_position):
        buffer = ZBuffer(20, 20, background=(0, 0, 0))
        buffer.triangle_phong_toon(
            (2, 2, 0), (17, 2, 0), (2, 17, 0),
            normal, normal, normal,
            light_position, (200, 100, 50),
            positions=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ambient=0.2, diffuse=0.8,
        )
        offset = (5 * buffer.width + 5) * 3
        return tuple(buffer.rgb[offset:offset + 3])

    def test_toon_uses_brightest_band_when_normal_faces_light(self):
        self.assertEqual(self._render_triangle((0, 0, 1), (0, 0, 5)), (200, 100, 50))

    def test_toon_keeps_ambient_band_when_normal_faces_away(self):
        self.assertEqual(self._render_triangle((0, 0, -1), (0, 0, 5)), (40, 20, 10))

    def test_zero_normal_is_safe_and_uses_ambient(self):
        self.assertEqual(self._render_triangle((0, 0, 0), (0, 0, 5)), (40, 20, 10))

    def test_interpolated_normals_produce_different_pixel_bands(self):
        buffer = ZBuffer(22, 22, background=(0, 0, 0))
        buffer.triangle_phong_toon(
            (2, 2, 0), (18, 2, 0), (2, 18, 0),
            (0, 0, 1), (1, 0, 0), (0, 1, 0),
            (0.5, 0.5, 5), (255, 255, 255),
            positions=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ambient=0.1, diffuse=0.9,
        )
        near_vertex = tuple(buffer.rgb[(3 * 22 + 3) * 3:(3 * 22 + 3) * 3 + 3])
        toward_edge = tuple(buffer.rgb[(15 * 22 + 3) * 3:(15 * 22 + 3) * 3 + 3])
        self.assertNotEqual(near_vertex, toward_edge)

    def test_scene_supports_all_requested_shading_modes(self):
        for shading in ("flat", "gouraud", "toon", "texture"):
            with self.subTest(shading=shading):
                cube = create_polyhedron("Гексаэдр (куб)")
                buffer = render_scene(
                    [cube], lambda point: tuple(point), 100, 100, 25,
                    "persp", 7.0, shading=shading,
                    light_position=(4, 6, 8), object_color=(160, 90, 210),
                )
                self.assertTrue(any(owner >= 0 for owner in buffer.owner))

    def test_transformed_model_renders_with_transformed_normals(self):
        cube = create_polyhedron("Гексаэдр (куб)")
        cube.transform(transforms.scaling(1.8, 0.6, 1.1))
        cube.transform(transforms.rotation_y(35))
        buffer = render_scene(
            [cube], lambda point: tuple(point), 100, 100, 25,
            "axon", 7.0, shading="toon", light_position=(4, 6, 8),
        )
        self.assertTrue(any(owner >= 0 for owner in buffer.owner))


    def test_normalize_vector_returns_unit_vector_and_handles_zero(self):
        self.assertEqual(normalize_vector((0, 0, 0)), (0.0, 0.0, 0.0))
        normalized = normalize_vector((0, 3, 4))
        self.assertAlmostEqual(sum(v * v for v in normalized), 1.0)

    def test_toon_band_returns_four_discrete_levels(self):
        self.assertEqual(toon_band(0.1), 0.0)
        self.assertAlmostEqual(toon_band(0.3), 1.0 / 3.0)
        self.assertAlmostEqual(toon_band(0.6), 2.0 / 3.0)
        self.assertEqual(toon_band(0.9), 1.0)

    def test_transform_normal_to_view_ignores_translation(self):
        def translated_view(point):
            return (point[0] + 10, point[1] - 4, point[2] + 7)
        self.assertEqual(
            transform_normal_to_view((0, 0, 2), translated_view),
            (0.0, 0.0, 1.0),
        )

    def test_obj_viewer_imports_dedicated_scene_renderer(self):
        import obj_viewer
        self.assertIs(obj_viewer.render_scene, render_scene)

    def test_zbuffer_legacy_render_scene_api_still_works(self):
        cube = create_polyhedron("Гексаэдр (куб)")
        image = legacy_render_scene(
            [cube], lambda point: tuple(point), 60, 60, 15,
            "axon", 7.0, shading="toon",
        )
        self.assertTrue(any(owner >= 0 for owner in image.owner))

    def test_phong_toon_color_applies_ambient_and_object_color(self):
        color = phong_toon_color(
            (0, 0, 0), (0, 0, -1), (0, 0, 5), (200, 100, 50),
            ambient=0.2, diffuse=0.8,
        )
        self.assertEqual(color, (40.0, 20.0, 10.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
