#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit-тесты отсечения нелицевых граней без запуска Tkinter."""

import unittest
from pathlib import Path

from backface_culling import (
    classify_faces,
    face_cross_normal,
    filtered_model,
    visible_and_culled,
)
from geometry import Point, Polygon, Polyhedron, create_polyhedron
from obj_viewer import load_obj
import transforms


class BackFaceCullingTests(unittest.TestCase):
    def setUp(self):
        self.identity = lambda point: point

    def test_cube_parallel_projection_keeps_only_camera_facing_face(self):
        cube = create_polyhedron("Гексаэдр (куб)")
        faces = classify_faces(cube, self.identity, "axon")
        visible, culled = visible_and_culled(faces)
        self.assertEqual(len(visible), 1)
        self.assertEqual(len(culled), 5)
        self.assertEqual(len(filtered_model(cube, visible).faces), 1)
        center = cube.center()
        for face in faces:
            outward = sum(
                face.normal[axis] * (face.centroid[axis] - center[axis])
                for axis in range(3)
            )
            self.assertGreater(outward, 0)

    def test_cube_oblique_camera_keeps_front_faces(self):
        cube = create_polyhedron("Гексаэдр (куб)")
        faces = classify_faces(
            cube,
            lambda p: (
                p[0] * 0.8 + p[1] * 0.12 + p[2] * 0.59,
                p[0] * -0.10 + p[1] * 0.98 + p[2] * 0.15,
                p[0] * -0.60 + p[1] * -0.13 + p[2] * 0.79,
            ),
            "axon",
        )
        visible, culled = visible_and_culled(faces)
        self.assertEqual(len(visible), 3)
        self.assertEqual(len(culled), 3)

    def test_sphere_keeps_approximately_camera_facing_half(self):
        sphere = load_obj(Path(__file__).resolve().parent / "examples" / "sphere.obj")
        faces = classify_faces(sphere, self.identity, "axon")
        visible, culled = visible_and_culled(faces)
        self.assertGreater(len(visible), 70)
        self.assertLess(len(visible), 125)
        self.assertEqual(len(visible) + len(culled), len(sphere.faces))

    def test_translated_torus_keeps_outward_normals(self):
        torus = load_obj(Path(__file__).resolve().parent / "examples" / "torus.obj")
        torus.transform(transforms.translation(1000, -300, 250))
        faces = classify_faces(torus, self.identity, "axon")
        visible, culled = visible_and_culled(faces)
        self.assertEqual(len(visible), 96)
        self.assertEqual(len(culled), 96)

    def test_reversed_closed_mesh_is_still_oriented_outward(self):
        cube = create_polyhedron("Гексаэдр (куб)")
        reversed_faces = [Polygon(list(reversed(face.points))) for face in cube.faces]
        reversed_cube = Polyhedron(cube.vertices, reversed_faces, cube.name)
        faces = classify_faces(reversed_cube, self.identity, "axon")
        visible, culled = visible_and_culled(faces)
        self.assertEqual(len(visible), 1)
        self.assertEqual(len(culled), 5)

    def test_open_revolution_mesh_with_inward_winding_is_corrected(self):
        vase = load_obj(Path(__file__).resolve().parent / "examples" / "vase.obj")
        faces = classify_faces(vase, self.identity, "axon")
        self.assertTrue(any(face.visible for face in faces))
        self.assertTrue(any(not face.visible for face in faces))

    def test_cross_product_normal_is_perpendicular_to_face_edges(self):
        normal = face_cross_normal(((0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0)))
        self.assertAlmostEqual(normal[0], 0)
        self.assertAlmostEqual(normal[1], 0)
        self.assertGreater(normal[2], 0)

    def test_perspective_classification_uses_camera_position(self):
        # Нормаль направлена по (+X, 0, +Z). Параллельная камера видит
        # грань по положительной Z-компоненте, но перспектива с камерой
        # в (0, 0, 7) смотрит на грань с противоположной стороны.
        points = [
            Point(10.5, -0.5, -0.5), Point(10.5, 0.5, -0.5),
            Point(9.5, 0.5, 0.5), Point(9.5, -0.5, 0.5),
        ]
        face = Polygon(points)
        model = Polyhedron(points, [face])
        parallel = classify_faces(model, self.identity, "axon")
        perspective = classify_faces(model, self.identity, "persp", camera_distance=7)
        self.assertTrue(parallel[0].visible)
        self.assertFalse(perspective[0].visible)


if __name__ == "__main__":
    unittest.main(verbosity=2)
