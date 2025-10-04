"""SVG rendering helpers."""
from __future__ import annotations

import os.path
from typing import Iterable

import svgwrite


class SvgRenderer:
    """Render a collection of geometric primitives to SVG."""

    def __init__(self, filename: str = "output.svg", size: tuple[int, int] = (900, 700)) -> None:
        self.filename = filename
        self.size = size

        self.output_dir = "output_svg"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def draw(self, objects: Iterable[object]) -> None:
        from geo_dsl import GeoArc
        from sympy import Circle, Line, Point, Segment

        dwg = svgwrite.Drawing(
            os.path.join(self.output_dir, self.filename),
            profile="full",
            size=self.size,
        )

        for element in objects:
            if isinstance(element, Point):
                SvgRenderer.draw_point(dwg, element)
            elif isinstance(element, Line):
                start = (float(element.p1.x), float(element.p1.y))
                end = (float(element.p2.x), float(element.p2.y))
                dwg.add(dwg.line(start=start, end=end, stroke="blue", stroke_width=1))
            elif isinstance(element, Segment):
                start = (float(element.p1.x), float(element.p1.y))
                end = (float(element.p2.x), float(element.p2.y))
                dwg.add(
                    dwg.line(
                        start=start,
                        end=end,
                        stroke="green",
                        stroke_width=1,
                        stroke_dasharray="5,1",
                    )
                )
            elif isinstance(element, Circle):
                SvgRenderer.draw_circle(dwg, element)
            elif isinstance(element, GeoArc):
                element.draw_svg(dwg)
            else:
                raise ValueError(f"A non-drawable object passed: {element}")

        SvgRenderer.rotate_drawing(dwg, 90)
        dwg.save()

    @staticmethod
    def draw_circle(dwg: svgwrite.Drawing, element) -> None:
        center = (float(element.center.x), float(element.center.y))
        radius = float(element.radius)
        dwg.add(dwg.circle(center=center, r=radius, fill="none", stroke="orange", stroke_width=1))

    @staticmethod
    def draw_point(dwg: svgwrite.Drawing, element, size: int = 3) -> None:
        x, y = float(element.x), float(element.y)
        dwg.add(
            dwg.line(
                start=(x - size, y - size),
                end=(x + size, y + size),
                stroke="red",
                stroke_width=1,
            )
        )
        dwg.add(
            dwg.line(
                start=(x - size, y + size),
                end=(x + size, y - size),
                stroke="red",
                stroke_width=1,
            )
        )

    @staticmethod
    def rotate_drawing(dwg: svgwrite.Drawing, angle: int) -> None:
        if angle % 90 != 0:
            raise ValueError("Only multiples of 90° supported")

        width = float(str(dwg["width"]).replace("px", ""))
        height = float(str(dwg["height"]).replace("px", ""))

        if angle % 180 != 0:
            dwg["width"], dwg["height"] = dwg["height"], dwg["width"]
            width_new, height_new = height, width
        else:
            width_new, height_new = width, height

        cx_old, cy_old = width / 2, height / 2
        cx_new, cy_new = width_new / 2, height_new / 2

        group = dwg.g()
        for elem in list(dwg.elements):
            group.add(elem)
        dwg.elements.clear()

        transform = f"translate({cx_new},{cy_new}) rotate({angle}) translate({-cx_old},{-cy_old})"
        group.attribs["transform"] = transform
        dwg.add(group)


__all__ = ["SvgRenderer"]
