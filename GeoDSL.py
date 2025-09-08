import warnings
import svgwrite
import svgutils
from sympy import Point, Circle, Line, Segment, intersection as sympy_intersection
import numpy as np

class GeoArc:
    def __init__(self, center, p1, p2):
        self.center = center
        self.radius = center.distance(p1)
        self.angle1 = self._calculate_angle(center, p1)
        self.angle2 = self._calculate_angle(center, p2)
        self.point1 = p1
        self.point2 = p2

    @staticmethod
    def _calculate_angle(center, point):
        """Angle of point relative to center in degrees."""
        cx, cy = map(float, center.args)
        px, py = map(float, point.args)
        return (180.0 / np.pi) * np.arctan2(py - cy, px - cx)

    def _is_angle_between(self, test_angle, start_angle, end_angle, eps=1e-9):
        """
        Return True if test_angle lies on the directed arc start_angle → end_angle.
        Endpoint-safe and handles CW/CCW arcs.
        """
        a = test_angle % 360
        s = start_angle % 360
        e = end_angle % 360

        sweep = (e - s) % 360
        if sweep > 180:
            sweep -= 360  # CW
        span = (a - s) % 360
        if sweep < 0:
            span = ((a - s) % 360) - 360  # adjust for CW

        return -eps <= span <= sweep + eps

    def draw(self, dwg):
        """Draw the arc with correct SVG flags respecting point order."""
        cx, cy = float(self.center.x), float(self.center.y)
        r = float(self.radius)

        start_rad = np.deg2rad(self.angle1)
        end_rad   = np.deg2rad(self.angle2)
        start_point = (cx + r * np.cos(start_rad), cy + r * np.sin(start_rad))
        end_point   = (cx + r * np.cos(end_rad),   cy + r * np.sin(end_rad))

        # Compute signed delta for sweep direction
        delta = (self.angle2 - self.angle1)
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360

        sweep_flag = 1 if delta > 0 else 0       # CCW = 1, CW = 0
        large_arc_flag = 1 if abs(delta) > 180 else 0

        d = f'M {start_point[0]} {start_point[1]} ' \
            f'A {r} {r} 0 {large_arc_flag} {sweep_flag} {end_point[0]} {end_point[1]}'

        arc_path = dwg.path(d=d, fill='none', stroke='purple', stroke_width=1)
        dwg.add(arc_path)
        return arc_path

class GeoDSL:
    def __init__(self, display_size):
        self.display_size = display_size
        self.svg_size = [display_size*9, display_size*6]

    def point(self, x, y):
        """Create a point."""
        # p = Point(float(x), float(y), evaluate=False)
        p = Point(x, y)
        return p

    def simple_point(self, p):
        return Point(float(p.x), float(p.y), evaluate=False)

    def line(self, p1, p2):
        """Create a line through two points."""
        l = Line(p1, p2)
        return l

    def segment(self, p1, p2):
        """Create a segment between two points."""
        s = Segment(p1, p2)
        return s

    def circle_by_compass(self, center, p1, p2):
        """Create a circle with radius p1.distance(p2)"""
        radius = p1.distance(p2)
        c = Circle(center, radius)
        return c

    def circle_by_center_and_radius(self, center, radius):
        """Create a circle given the center and radius."""
        c = Circle(center, radius)
        return c

    def circle_by_center_and_point(self, center, point):
        """Create a circle given the center and a point on the circumference."""
        radius = center.distance(point)
        return self.circle_by_center_and_radius(center, radius)

    def arc_by_center_and_two_points(self, center, p1, p2):
        """Create a circular arc given a center and two points."""
        arc = GeoArc(center, p1, p2)  # Create Arc with center and two points
        return arc


    def midpoint(self, p1, p2):
        """Find the midpoint of two points."""
        mid = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        return mid

    def perpendicular_segment(self, line, point):
        perp = line.perpendicular_segment(point)
        return perp

    def perpendicular_line(self, line, point):
        """Create a line perpendicular to a given line at a specified point."""
        perp = line.perpendicular_line(point)
        return perp

    def parallel_line(self, line, point):
        """Create a line parallel to a given line through a specified point."""
        par = line.parallel_line(point)
        return par

    def intersection(self, obj1, obj2):
        """Find the intersection points between two geometric objects."""
        points = sympy_intersection(obj1, obj2)  # Directly use sympy's intersection
        return points

    def translate_point(self, p1, ref1, ref2):
        """Translate point by the distance between the reference points"""
        distance_x = ref2.x - ref1.x
        distance_y = ref2.y - ref1.y
        p = self.point(p1.x + distance_x, p1.y + distance_y)
        return p

    def translate_point_x(self, p1, distance):
        p = self.point(p1.x + distance, p1.y)
        return p

    def translate_point_y(self, p1, distance):
        p = self.point(p1.x, p1.y + distance)
        return p

    def reflect(self, obj, reflection_obj):
        """Reflect a geometric object across a line or around a point."""
        if isinstance(reflection_obj, Line):
            # Reflection across a line
            return self._reflect_across_line(obj, reflection_obj)
        elif isinstance(reflection_obj, Point):
            # Reflection around a point
            return self._reflect_around_point(obj, reflection_obj)
        else:
            raise ValueError(f"Reflection object must be a Line or Point, got {type(reflection_obj)}")

    def divide_distance(self, p1, p2, count):
        """ Proper geometrical division would be by drawing circles and intersecting them.
        That creates trouble with calculating intersections though, so this method is simplified and uses
        simple distance division and translation """
        distance_x = p2.x - p1.x
        distance_y = p2.y - p1.y

        dividing_unit_x = distance_x / count
        dividing_unit_y = distance_y / count
        dividing_points = []
        for i in range(count-1):
            dividing_points.append(self.point(p1.x + (i+1) * dividing_unit_x, p1.y + (i+1) * dividing_unit_y))

        return dividing_points

    def _reflect_across_line(self, obj, line_of_reflection):
        """Reflect a geometric object across a given line."""
        if isinstance(obj, Point):
            return obj.reflect(line_of_reflection)
        elif isinstance(obj, Line):
            p1_reflected = obj.p1.reflect(line_of_reflection)
            p2_reflected = obj.p2.reflect(line_of_reflection)
            return self.line(p1_reflected, p2_reflected)
        elif isinstance(obj, Segment):
            p1_reflected = obj.p1.reflect(line_of_reflection)
            p2_reflected = obj.p2.reflect(line_of_reflection)
            return self.segment(p1_reflected, p2_reflected)
        elif isinstance(obj, Circle):
            center_reflected = obj.center.reflect(line_of_reflection)
            return self.circle(center_reflected, obj.radius)
        elif isinstance(obj, GeoArc):
            center_reflected = obj.center.reflect(line_of_reflection)
            p1_reflected = obj.point1.reflect(line_of_reflection)
            p2_reflected = obj.point2.reflect(line_of_reflection)
            return GeoArc(center_reflected, p2_reflected, p1_reflected)
        else:
            raise ValueError("Reflection not supported for object type: {type(obj)}")

    def _reflect_around_point(self, obj, reflection_point):
        """Reflect a geometric object around a given point."""
        if isinstance(obj, Point):
            # Reflect a point around another point
            dx = reflection_point.x - obj.x
            dy = reflection_point.y - obj.y
            return Point(reflection_point.x + dx, reflection_point.y + dy)
        elif isinstance(obj, Line):
            # Reflect two points on the line around the point
            p1_reflected = self._reflect_around_point(obj.p1, reflection_point)
            p2_reflected = self._reflect_around_point(obj.p2, reflection_point)
            return self.line(p1_reflected, p2_reflected)
        elif isinstance(obj, Segment):
            # Reflect both endpoints of the segment around the point
            p1_reflected = self._reflect_around_point(obj.p1, reflection_point)
            p2_reflected = self._reflect_around_point(obj.p2, reflection_point)
            return self.segment(p1_reflected, p2_reflected)
        elif isinstance(obj, Circle):
            # Reflect the center and retain the radius
            center_reflected = self._reflect_around_point(obj.center, reflection_point)
            return self.circle_by_center_and_radius(center_reflected, obj.radius)
        elif isinstance(obj, GeoArc):
            # Reflect the center and the two points defining the arc
            center_reflected = self._reflect_around_point(obj.center, reflection_point)
            p1_reflected = self._reflect_around_point(obj.point1, reflection_point)
            p2_reflected = self._reflect_around_point(obj.point2, reflection_point)
            return GeoArc(center_reflected, p2_reflected, p1_reflected)
        else:
            raise ValueError(f"Reflection not supported for object type: {type(obj)}")

    def __point_is_between(self, a, b, c):
        "Return true iff point c intersects the line segment from a to b."
        # (or the degenerate case that all 3 points are coincident)
        return (self.__points_are_collinear(a, b, c)
                and (self.within(a.x, c.x, b.x) if a.x != b.x else
                     self.within(a.y, c.y, b.y)))

    def __points_are_collinear(self, a, b, c):
        "Return true iff a, b, and c all lie on the same line."
        return (b.x - a.x) * (c.y - a.y) == (c.x - a.x) * (b.y - a.y)

    def __coordinate_is_within(p, q, r):
        "Return true iff q is between p and r (inclusive)."
        return p <= q <= r or r <= q <= p

    def pick_point_closest_to(self, reference, lst):
        # turn points into simple points for perfomance
        lst = [self.simple_point(p) for p in lst]

        minimum_distance = lst[0].distance(reference)
        closest_point = lst[0]

        for p in lst:
            dst = p.distance(reference)
            if minimum_distance > dst:
                minimum_distance = dst
                closest_point = p

        return closest_point

    def pick_point_furthest_from(self, reference, lst):
        maximum_distance = lst[0].distance(reference)
        furthest_point = lst[0]

        for p in lst:
            dst = p.distance(reference)
            if maximum_distance < dst:
                maximum_distance = dst
                furthest_point = p

        return furthest_point

    def pick_west_point(self, p1, p2=None):
        if p2 == None:
            print("Warning: only one argument passed to", pick_west_point)
            return p1

        if p2.x < p1.x:
            return p2
        elif p2.x > p1.x:
            return p1
        else:
            print("Warning when finding west point: both points have the same x. Picking the south point")
            if p1.y < p2.y:
                return p1
            else:
                return p2

    def golden_ratio_divider(self, p1, p2):
        """
        Dividing towards p2:
        p1 ---------- G ----- p2
        """
        dwg = svgwrite.Drawing(filename='golden_ratio_divider.svg', profile='tiny', size=(2000, 2000))
        line_between = self.line(p1, p2)

        big_circle = self.circle_by_compass(p2, p1, p2)
        perpendicular_line = self.perpendicular_line(line_between, p2)
        big_intersection_point = self.intersection(big_circle, perpendicular_line)[0]

        midpoint = self.midpoint(big_intersection_point, p2)
        small_circle = self.circle_by_center_and_point(midpoint, p2)

        cross_line = self.line(midpoint, p1)
        cross_intersections = self.intersection(cross_line, small_circle)

        cross_intersection = self.pick_point_closest_to(p1, cross_intersections)
        golden_circle = self.circle_by_center_and_point(p1, cross_intersection)
        golden_intersections = self.intersection(line_between, golden_circle)
        golden_point = self.pick_point_closest_to(p2, golden_intersections)

        return golden_point

    def get_tangent_circle(self, circle, line, radius, closest_point, inner=True):
        '''
        Given a line and a circle, creates a new circle, with
        a given radius, that is tangential at the intersection point
        closest to a given point.

        The given line should pass through the center of the given
        circle. Otherwise, a warning is thrown.

        inner parameter sets whether the new circle should be inside or
        outside the given circle. True by default.

        Returns the new circle and its intersection with the given circle.

        '''
        if not self.intersection(circle.center, line):
            warnings.warn("Calculating tangent circle with a non-radial line", Warning, stacklevel=2)

        dwg = svgwrite.Drawing(filename='tangent_output.svg', profile='tiny', size=(1400, 1100))
        GeoDSL.draw_circle(dwg, circle)
        GeoDSL.draw_point(dwg, circle.center)

        intersections = self.intersection(circle, line)
        intersection = self.pick_point_closest_to(closest_point, intersections)

        helper_circle = self.circle_by_center_and_radius(intersection, radius)
        helper_intersections = self.intersection(helper_circle, line)
        if inner:
            new_circle_center = self.pick_point_closest_to(circle.center, helper_intersections)
        else:
            new_circle_center = self.pick_point_furthest_from(circle.center, helper_intersections)

        new_circle = self.circle_by_center_and_radius(new_circle_center, radius)
        GeoDSL.draw_circle(dwg, new_circle)
        dwg.save()
        return new_circle, intersection

    def blend_two_circles(self, blender_radius, circle_1, circle_2):
        helper_1_radius = circle_1.radius - blender_radius
        helper_1_circle = self.circle_by_center_and_radius(circle_1.center, helper_1_radius)
        helper_2_radius = circle_2.radius - blender_radius
        helper_2_circle = self.circle_by_center_and_radius(circle_2.center, helper_2_radius)

        helper_circles_intersections = self.intersection(helper_1_circle, helper_2_circle)

        ''' TODO: don't attempt to pick the right intersection here.
        Return two sets of blend circle and points instead and leave
        it to the caller to deduct the right blending circle. Or take
        a 'closest_point' parameter to resolve it.
        '''
        if helper_circles_intersections[0].y < helper_circles_intersections[1].y:
            blender_center = helper_circles_intersections[0]
        else:
            blender_center = helper_circles_intersections[1]
        blender_center = self.simple_point(blender_center)

        blender_circle = self.circle_by_center_and_radius(blender_center, blender_radius+0.0001)

        blender_intersections_1 = self.intersection(blender_circle, circle_1)
        blender_intersections_2 = self.intersection(blender_circle, circle_2)

        return blender_circle, blender_intersections_1[0], blender_intersections_2[0]

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

    def draw_svg(self, objects, filename='output.svg'):
        """Draw specified geometric objects to an SVG file."""
        dwg = svgwrite.Drawing(filename, profile='full', size=self.svg_size)

        for element in objects:
            if isinstance(element, Point):
                GeoDSL.draw_point(dwg, element)
            elif isinstance(element, Line):
                start = (float(element.p1.x), float(element.p1.y))
                end = (float(element.p2.x), float(element.p2.y))
                dwg.add(dwg.line(start=start, end=end, stroke='blue', stroke_width=1))
            elif isinstance(element, Segment):
                start = (float(element.p1.x), float(element.p1.y))
                end = (float(element.p2.x), float(element.p2.y))
                dwg.add(dwg.line(start=start, end=end, stroke='green', stroke_width=1, stroke_dasharray='5,1'))
            elif isinstance(element, Circle):
                GeoDSL.draw_circle(dwg, element)
            elif isinstance(element, GeoArc):
                element.draw(dwg)  # Call the Arc's draw method
            else:
                raise ValueError("A non-drawable object passed: ", element)

        GeoDSL.rotate_drawing(dwg, 90)

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

def main():
    dwg = svgwrite.Drawing('test_output.svg', profile='full', size=[700,600])

    geo = GeoDSL()
    geo.draw_point(dwg, geo.point(200,100))
    arc = geo.arc_by_center_and_two_points(geo.point(150,150), geo.point(150,200), geo.point(200,150))

    p1 = geo.point(300,370)
    p2 = geo.point(370,300)
    arc = geo.arc_by_center_and_two_points(geo.point(300,300), p2, p1)
    arc.draw(dwg)
    dwg.save()

if __name__ == '__main__':
    main()