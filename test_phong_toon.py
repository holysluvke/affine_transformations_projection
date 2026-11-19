"""Проверки попиксельного Phong/toon-шейдинга без запуска Tkinter."""

import math
import unittest

from geometry import create_polyhedron
from zbuffer import ZBuffer, render_scene as legacy_render_scene
import phong_toon as phong_toon_module
from phong_toon import (
    render_scene, toon_band, normalize_vector, phong_toon_color, phong_color,
    transform_normal_to_view,
)
from obj_viewer import camera_light_to_world
from viewer import view_matrix, mat_apply
import transforms


class PhongToonTests(unittest.TestCase):
    def _render_triangle(self, normal, light_position):
        buffer = ZBuffer(20, 20, background=(0, 0, 0))
        buffer.triangle_phong_toon(
            (2, 2, 0), (17, 2, 0), (2, 17, 0),
            normal, normal, normal,
            light_position, (200, 100, 50),
            positions=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        )
        offset = (5 * buffer.width + 5) * 3
        return tuple(buffer.rgb[offset:offset + 3])

    def test_toon_uses_brightest_band_when_normal_faces_light(self):
        self.assertEqual(self._render_triangle((0, 0, 1), (0, 0, 5)), (255, 130, 65))

    def test_toon_uses_lowest_band_when_normal_faces_away(self):
        self.assertEqual(self._render_triangle((0, 0, -1), (0, 0, 5)), (60, 30, 15))

    def test_zero_normal_is_safe_and_uses_lowest_band(self):
        self.assertEqual(self._render_triangle((0, 0, 0), (0, 0, 5)), (60, 30, 15))

    def test_interpolated_normals_produce_different_pixel_bands(self):
        buffer = ZBuffer(22, 22, background=(0, 0, 0))
        buffer.triangle_phong_toon(
            (2, 2, 0), (18, 2, 0), (2, 18, 0),
            (0, 0, 1), (1, 0, 0), (0, 1, 0),
            (0.5, 0.5, 5), (255, 255, 255),
            positions=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        )
        near_vertex = tuple(buffer.rgb[(3 * 22 + 3) * 3:(3 * 22 + 3) * 3 + 3])
        toward_edge = tuple(buffer.rgb[(15 * 22 + 3) * 3:(15 * 22 + 3) * 3 + 3])
        self.assertNotEqual(near_vertex, toward_edge)

    def test_scene_supports_all_requested_shading_modes(self):
        for shading in ("flat", "gouraud", "phong", "toon", "texture"):
            with self.subTest(shading=shading):
                cube = create_polyhedron("Гексаэдр (куб)")
                buffer = render_scene(
                    [cube], lambda point: tuple(point), 100, 100, 25,
                    "persp", 7.0, shading=shading,
                    light_position=(4, 6, 8), object_color=(160, 90, 210),
                )
                self.assertTrue(any(owner >= 0 for owner in buffer.owner))

    def test_platonic_solid_uses_flat_normals_per_face(self):
        tetra = create_polyhedron("Тетраэдр")
        for face_index, face in enumerate(tetra.faces):
            expected = face.unit_normal()
            for normal in tetra.face_vertex_normals(face_index):
                for actual_component, expected_component in zip(normal, expected):
                    self.assertAlmostEqual(actual_component, expected_component, places=7)

    def test_flat_face_normals_remain_correct_after_rotation(self):
        tetra = create_polyhedron("Тетраэдр")
        tetra.transform(transforms.rotation_y(37))
        tetra.transform(transforms.rotation_x(-21))
        for face_index, face in enumerate(tetra.faces):
            expected = face.unit_normal()
            for normal in tetra.face_vertex_normals(face_index):
                for actual_component, expected_component in zip(normal, expected):
                    self.assertAlmostEqual(actual_component, expected_component, places=7)

    def test_camera_relative_light_stays_at_same_camera_coordinates(self):
        light_camera = (2.0, 3.0, 4.0)
        center = (0.5, -0.25, 1.5)
        radius = 2.75
        for yaw, pitch in ((0.0, 0.0), (0.7, -0.3), (-1.2, 0.45)):
            rotation = view_matrix(yaw, pitch)
            light_world = camera_light_to_world(light_camera, rotation, center, radius)
            normalized = tuple((light_world[i] - center[i]) / radius for i in range(3))
            projected = mat_apply(rotation, normalized)[:3]
            for actual, expected in zip(projected, light_camera):
                self.assertAlmostEqual(actual, expected, places=7)

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

    def test_toon_band_matches_three_levels_from_lecture(self):
        # diff = 0.2 + max(N·L, 0); пороги 0.4/0.7, множители 0.3/1.0/1.3.
        self.assertEqual(toon_band(0.0), 0.3)
        self.assertEqual(toon_band(0.199999), 0.3)
        self.assertEqual(toon_band(0.2), 1.0)
        self.assertEqual(toon_band(0.499999), 1.0)
        self.assertEqual(toon_band(0.5), 1.3)
        self.assertEqual(toon_band(1.0), 1.3)

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

    def test_phong_toon_color_matches_lecture_formula(self):
        color_away = phong_toon_color(
            (0, 0, 0), (0, 0, -1), (0, 0, 5), (200, 100, 50),
        )
        color_mid = phong_toon_color(
            (0, 0, 0), (math.sqrt(1 - 0.3 ** 2), 0, 0.3), (0, 0, 5), (100, 80, 60),
        )
        color_bright = phong_toon_color(
            (0, 0, 0), (0, 0, 1), (0, 0, 5), (100, 80, 60),
        )
        self.assertEqual(color_away, (60.0, 30.0, 15.0))
        self.assertEqual(color_mid, (100.0, 80.0, 60.0))
        self.assertEqual(color_bright, (130.0, 104.0, 78.0))
        self.assertEqual(
            phong_toon_color((0, 0, 0), (0, 0, 1), (0, 0, 5), (255, 200, 100)),
            (255.0, 255.0, 130.0),
        )


    def test_phong_uses_continuous_diffuse_intensity(self):
        # Оба значения N·L лежат в одной toon-полосе, но Фонг должен
        # сохранять непрерывную разницу яркости между ними.
        color_a = phong_color(
            (0, 0, 0), (0, 0, 1), (math.sqrt(1 - 0.3 ** 2), 0, 0.3),
            None, (200, 100, 50), ambient=0, diffuse=1, specular=0,
        )
        color_b = phong_color(
            (0, 0, 0), (0, 0, 1), (math.sqrt(1 - 0.4 ** 2), 0, 0.4),
            None, (200, 100, 50), ambient=0, diffuse=1, specular=0,
        )
        self.assertAlmostEqual(color_a[0], 60.0)
        self.assertAlmostEqual(color_b[0], 80.0)
        self.assertNotEqual(color_a, color_b)

    def test_phong_adds_specular_highlight_when_view_and_reflection_align(self):
        color = phong_color(
            (0, 0, 0), (0, 0, 1), (0, 0, 5), (0, 0, 10),
            (100, 80, 60), ambient=0, diffuse=0, specular=0.5, shininess=16,
        )
        self.assertEqual(color, (127.5, 127.5, 127.5))

    def test_numpy_rasterizer_matches_reference_pixel_loop(self):
        if phong_toon_module.np is None:
            self.skipTest("NumPy не установлен; проверяется резервный Python-рендерер")

        common = dict(
            p0=(2.4, 3.2, -0.1), p1=(33.2, 4.7, -0.2), p2=(5.1, 30.4, -0.3),
            n0=(0.1, 0.2, 1.0), n1=(1.0, 0.0, 0.0), n2=(0.2, 1.0, 0.5),
            light_position=(4.0, 6.0, 8.0), object_color=(170, 90, 210), owner=3,
            positions=((-0.5, -0.3, 0.0), (0.8, -0.2, 0.1), (-0.4, 0.9, -0.1)),
            reciprocal_w=(0.3, 0.4, 0.2),
        )
        for mode in ("phong", "toon"):
            with self.subTest(mode=mode):
                accelerated = ZBuffer(40, 35, background=(3, 4, 5))
                reference = ZBuffer(40, 35, background=(3, 4, 5))
                if mode == "phong":
                    phong_toon_module.triangle_phong(
                        accelerated, **common, view_position=(0.0, 0.0, 7.0)
                    )
                    shade = lambda position, normal, base: phong_color(
                        position, normal, common["light_position"],
                        (0.0, 0.0, 7.0), base,
                    )
                else:
                    phong_toon_module.triangle_phong_toon(accelerated, **common)
                    shade = lambda position, normal, base: phong_toon_color(
                        position, normal, common["light_position"], base,
                    )
                phong_toon_module._triangle_per_pixel(
                    reference,
                    common["p0"], common["p1"], common["p2"],
                    common["n0"], common["n1"], common["n2"],
                    common["light_position"], common["object_color"], common["owner"],
                    positions=common["positions"],
                    reciprocal_w=common["reciprocal_w"], shade_pixel=shade,
                )
                self.assertEqual(accelerated.rgb, reference.rgb)
                self.assertTrue((accelerated.owner == reference.owner).all())
                finite = phong_toon_module.np.isfinite(reference.depth)
                self.assertTrue(
                    phong_toon_module.np.allclose(
                        accelerated.depth[finite], reference.depth[finite], atol=1e-12
                    )
                )

    def test_zbuffer_phong_adapter_rasterizes_triangle(self):
        buffer = ZBuffer(20, 20, background=(0, 0, 0))
        buffer.triangle_phong(
            (2, 2, 0), (17, 2, 0), (2, 17, 0),
            (0, 0, 1), (0, 0, 1), (0, 0, 1),
            (0, 0, 5), (200, 100, 50), positions=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            view_position=(0, 0, 10), ambient=0, diffuse=1, specular=0,
        )
        offset = (5 * buffer.width + 5) * 3
        self.assertEqual(tuple(buffer.rgb[offset:offset + 3]), (200, 100, 50))



if __name__ == "__main__":
    unittest.main(verbosity=2)
