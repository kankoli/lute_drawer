from __future__ import annotations
import os.path
import svgwrite
from typing import Iterable

class SvgRenderer:
    """Render a list of primitives (('line', (...)), ('circle', (...)), ('arc', (GeoArc,))) to SVG."""
    def __init__(self, filename: str='output.svg', size: tuple[int,int]=(900,700)) -> None:
        self.filename = filename
        self.size = size

        self.output_dir = "output_svg"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def draw(self, objects):
        from geo_dsl import GeoArc
        from sympy import Point, Circle, Line, Segment

        dwg = svgwrite.Drawing(self.output_dir + "/" + self.filename, profile="full", size=self.size)

        for element in objects:
            if isinstance(element, Point):
                SvgRenderer.draw_point(dwg, element)
            elif isinstance(element, Line):
                start = (float(element.p1.x), float(element.p1.y))
                end = (float(element.p2.x), float(element.p2.y))
                dwg.add(dwg.line(start=start, end=end, stroke='blue', stroke_width=1))
            elif isinstance(element, Segment):
                start = (float(element.p1.x), float(element.p1.y))
                end = (float(element.p2.x), float(element.p2.y))
                dwg.add(dwg.line(start=start, end=end, stroke='green', stroke_width=1, stroke_dasharray='5,1'))
            elif isinstance(element, Circle):
                SvgRenderer.draw_circle(dwg, element)
            elif isinstance(element, GeoArc):
                element.draw_svg(dwg)  # Call the Arc's draw method
            else:
                raise ValueError("A non-drawable object passed: ", element)

        SvgRenderer.rotate_drawing(dwg, 90)

        dwg.save()

    @staticmethod
    def draw_circle(dwg, element):
        center = (float(element.center.x), float(element.center.y))
        radius = float(element.radius)
        dwg.add(dwg.circle(center=center, r=radius, fill='none', stroke='orange', stroke_width=1))

    @staticmethod
    def draw_point(dwg, element, size=3):
        """
        Draw a diagonal cross (X) centered at the element.

        Args:
            dwg: The SVG drawing object
            element: The point with x and y attributes
            size: Half-length of the cross arms
        """
        x, y = float(element.x), float(element.y)

        # Diagonal line: top-left to bottom-right
        dwg.add(dwg.line(start=(x - size, y - size), end=(x + size, y + size),
                         stroke='red', stroke_width=1))

        # Diagonal line: bottom-left to top-right
        dwg.add(dwg.line(start=(x - size, y + size), end=(x + size, y - size),
                         stroke='red', stroke_width=1))

    @staticmethod
    def rotate_drawing(dwg: svgwrite.Drawing, angle: int):
        """
        Rotate all elements in a drawing by multiples of 90°, keeping them
        fully centered.
        """
        if angle % 90 != 0:
            raise ValueError("Only multiples of 90° supported")

        # Extract numeric width/height
        width = float(str(dwg['width']).replace('px',''))
        height = float(str(dwg['height']).replace('px',''))

        # Swap width/height for 90°/270°
        if angle % 180 != 0:
            dwg['width'], dwg['height'] = dwg['height'], dwg['width']
            width_new, height_new = height, width
        else:
            width_new, height_new = width, height

        # Compute centers
        cx_old, cy_old = width / 2, height / 2
        cx_new, cy_new = width_new / 2, height_new / 2

        # Wrap elements in a group
        g = dwg.g()
        for elem in dwg.elements:
            g.add(elem)
        dwg.elements.clear()

        # Apply combined transform: translate to old center → rotate → translate to new center
        transform = f"translate({cx_new},{cy_new}) rotate({angle}) translate({-cx_old},{-cy_old})"
        g.attribs['transform'] = transform

        dwg.add(g)
 