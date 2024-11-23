import svgwrite
from sympy import Point, Circle, Line, Segment, nsimplify, intersection as sympy_intersection
import numpy as np

svg_size = [1400, 1100]

class GeoArc:
    def __init__(self, center, p1, p2):
        self.center = center
        self.radius = center.distance(p1)  # Calculate radius based on center and one point
        self.angle1 = self._calculate_angle(center, p1)  # Calculate angle for p1
        self.angle2 = self._calculate_angle(center, p2)  # Calculate angle for p2
        self.point1 = p1
        self.point2 = p2

    def intersect(self, other):
        """Calculate the intersection points between the arc and another geometric object."""
        points = sympy_intersection(Circle(self.center, self.radius), other)  # Use Circle's intersection

        valid_points = []
        for point in points:
            if isinstance(point, Point):
                angle = self._calculate_angle(self.center, point)
                if min(self.angle1, self.angle2) <= angle <= max(self.angle1, self.angle2):
                    valid_points.append(point)

        return valid_points

    def _calculate_angle(self, center, point):
        """Calculate the angle of a point relative to the center."""
        cx, cy = map(float, center.args)  # Convert to float
        px, py = map(float, point.args)    # Convert to float
        return (180 / np.pi) * np.arctan2(py - cy, px - cx)

    def draw(self, dwg, size):
        """Draw the arc on the provided SVG drawing."""
        start_angle = np.deg2rad(self.angle1)
        end_angle = np.deg2rad(self.angle2)

        start_point = (
            float(self.center.x) + float(self.radius) * np.cos(start_angle),
            size[1] - (float(self.center.y) + float(self.radius) * np.sin(start_angle))
        )

        end_point = (
            float(self.center.x) + float(self.radius) * np.cos(end_angle),
            size[1] - (float(self.center.y) + float(self.radius) * np.sin(end_angle))
        )

        # Ensure radius is evaluated correctly as a float
        radius_float = float(self.radius)

        # Create the SVG path for the arc
        arc_path = dwg.path(
            d=f'M {start_point[0]} {start_point[1]} A {radius_float} {radius_float} 0 0 1 {end_point[0]} {end_point[1]}',
            fill='none', stroke='purple', stroke_width=1
        )

        # Add the path to the drawing
        dwg.add(arc_path)

        # Return the created arc_path for further use if needed
        return arc_path

class GeoDSL:
    def __init__(self):
        pass

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

    def _pick_point_closest_to(self, reference, lst):
        minimum_distance = lst[0].distance(reference)
        closest_point = lst[0]

        for p in lst:
            dst = p.distance(reference)
            if minimum_distance > dst:
                minimum_distance = dst
                closest_point = p

        return closest_point

    def _pick_point_between(self, lst, p1, p2):
        # NOT SAFE
        print("pick point in between")

        print ("lst", lst)
        print ("lst length: ", len(lst))

        print ("p1", p1)

        print ("p2", p2)
        for p in lst:
            if ((p1.x <= p.x <= p2.x) or \
                (p1.x >= p.x >= p2.x)) and \
                ((p1.y <= p.y <= p2.y) or \
                (p1.y >= p.y >= p2.y)):
                print("point picked", p)
                return p
        print("Warning: no points are between")

    def pick_south_point(self, p1, p2=None):
        if p2 == None:
            print("Warning: only one argument passed to",pick_south_point)
            return p1

        if p2.y < p1.y:
            return p2
        elif p2.y > p1.y:
            return p1
        else:
            print("Warning when finding southern point: both points have the same y. Picking the east point")
            if p1.x > p2.x:
                return p1
            else:
                return p2

    def pick_north_point(self, p1, p2=None):
        if p2 == None:
            print("Warning: only one argument passed to",pick_north_point)
            return p1

        if p2.y < p1.y:
            return p1
        elif p2.y > p1.y:
            return p2
        else:
            print("Warning when finding northern point: both points have the same y. Picking the east point")
            if p1.x > p2.x:
                return p1
            else:
                return p2

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

    def pick_east_point(self, p1, p2=None):
        if p2 == None:
            print("Warning: only one argument passed to",pick_east_point)
            return p1

        if p2.x > p1.x:
            return p2
        elif p2.x < p1.x:
            return p1
        else:
            print("Warning when finding east point: both points have the same x. Picking the south point")
            if p1.y < p2.y:
                return p1
            else:
                return p2


    def golden_ratio_divider(self, p1, p2):
        dwg = svgwrite.Drawing(filename='golden_ratio_divider.svg', profile='tiny', size=(2000, 2000))
        line_between = self.line(p1, p2)
        # GeoDSL.draw_point(dwg, p1)
        # GeoDSL.draw_point(dwg, p2)
        big_circle = self.circle_by_compass(p2, p1, p2)
        perpendicular_line = self.perpendicular_line(line_between, p2)
        big_intersection_point = self.intersection(big_circle, perpendicular_line)[0]
        # GeoDSL.draw_point(dwg, big_intersection_point)
        # GeoDSL.draw_circle(dwg, big_circle)

        midpoint = self.midpoint(big_intersection_point, p2)
        # GeoDSL.draw_point(dwg, midpoint)
        small_circle = self.circle_by_center_and_point(midpoint, p2)
        # GeoDSL.draw_circle(dwg, small_circle)

        cross_line = self.line(midpoint, p1)
        cross_intersections = self.intersection(cross_line, small_circle)

        cross_intersection = self._pick_point_closest_to(p1, cross_intersections)
        # GeoDSL.draw_point(dwg, cross_intersection)
        golden_circle = self.circle_by_center_and_point(p1, cross_intersection)
        # GeoDSL.draw_circle(dwg, golden_circle)
        golden_intersections = self.intersection(line_between, golden_circle)
        golden_point = self._pick_point_closest_to(p2, golden_intersections)
        # GeoDSL.draw_point(dwg, golden_point)

        # dwg.save()

        return golden_point

    def blend_two_circles(self, blender_radius, circle_1, circle_2):
        # dwg = svgwrite.Drawing(filename='test_output.svg', profile='tiny', size=(1400, 1100))
        # GeoDSL.draw_circle(dwg, circle_1)
        # GeoDSL.draw_circle(dwg, circle_2)

        helper_1_radius = circle_1.radius - blender_radius
        helper_1_circle = self.circle_by_center_and_radius(circle_1.center, helper_1_radius)
        # GeoDSL.draw_circle(dwg, helper_1_circle)
        helper_2_radius = circle_2.radius - blender_radius
        helper_2_circle = self.circle_by_center_and_radius(circle_2.center, helper_2_radius)
        # GeoDSL.draw_circle(dwg, helper_2_circle)

        # dwg.save()
        helper_circles_intersections = self.intersection(helper_1_circle, helper_2_circle)

        if helper_circles_intersections[0].y < helper_circles_intersections[1].y:
            blender_center = helper_circles_intersections[0]
        else:
            blender_center = helper_circles_intersections[1]
        blender_center = self.simple_point(blender_center)

        # GeoDSL.draw_point(dwg, blender_center)

        blender_circle = self.circle_by_center_and_radius(blender_center, blender_radius+0.0000000001)
        # GeoDSL.draw_circle(dwg, blender_circle)
        # dwg.save()

        blender_intersections_1 = self.intersection(blender_circle, circle_1)
        blender_intersections_2 = self.intersection(blender_circle, circle_2)

        return blender_circle, blender_intersections_1[0], blender_intersections_2[0]

    @staticmethod
    def draw_svg(objects, filename='output.svg'):
        """Draw specified geometric objects to an SVG file."""
        dwg = svgwrite.Drawing(filename, profile='tiny', size=svg_size)

        for element in objects:
            if isinstance(element, Point):
                GeoDSL.draw_point(dwg, element)
            elif isinstance(element, Line):
                start = (float(element.p1.x), svg_size[1] - float(element.p1.y))
                end = (float(element.p2.x), svg_size[1] - float(element.p2.y))
                dwg.add(dwg.line(start=start, end=end, stroke='blue', stroke_width=2))
            elif isinstance(element, Segment):
                start = (float(element.p1.x), svg_size[1] - float(element.p1.y))
                end = (float(element.p2.x), svg_size[1] - float(element.p2.y))
                dwg.add(dwg.line(start=start, end=end, stroke='green', stroke_width=2, stroke_dasharray='1,5'))
            elif isinstance(element, Circle):
                GeoDSL.draw_circle(dwg, element)
            elif isinstance(element, GeoArc):
                element.draw(dwg, svg_size)  # Call the Arc's draw method
            else:
                raise ValueError("A non-drawable object passed: ", element)

        dwg.save()

    @staticmethod
    def draw_circle(dwg, element):
        center = (float(element.center.x), svg_size[1] - float(element.center.y))
        radius = float(element.radius)
        dwg.add(dwg.circle(center=center, r=radius, fill='none', stroke='orange', stroke_width=1))


    @staticmethod
    def draw_point(dwg, element):
        dwg.add(dwg.circle(center=(float(element.x), svg_size[1] - float(element.y)), r=2, fill='red'))


