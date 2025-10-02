import warnings
import svgwrite
import svgutils
from sympy import pi, Point, Circle, Line, Segment, intersection as sympy_intersection
import numpy as np

class GeoArc:
    def __init__(self, center, p1, p2):
        self.center = center
        self.radius = center.distance(p1)
        self.angle1 = GeoArc.__calculate_angle(center, p1)
        self.angle2 = GeoArc.__calculate_angle(center, p2)
        self.point1 = p1
        self.point2 = p2

    def rotate(self, p):
        center_reflected = self.center.rotate(pi, p)
        p1_reflected = self.point1.rotate(pi, p)
        p2_reflected = self.point2.rotate(pi, p)
        return GeoArc(center_reflected, p2_reflected, p1_reflected)

    def reflect(self, line):
        center_reflected = self.center.reflect(line)
        p1_reflected = self.point1.reflect(line)
        p2_reflected = self.point2.reflect(line)
        return GeoArc(center_reflected, p2_reflected, p1_reflected)

    @staticmethod
    def __calculate_angle(center, point):
        """Angle of point relative to center in degrees."""
        cx, cy = map(float, center.args)
        px, py = map(float, point.args)
        return (180.0 / np.pi) * np.arctan2(py - cy, px - cx)

    def draw_svg(self, dwg):
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

    def sample_points(self, n=100):
        """Return n points along the arc as a list of sympy.Point2D."""
        cx, cy = float(self.center.x), float(self.center.y)
        r = float(self.radius)
        a1_deg = float(self.angle1); a2_deg = float(self.angle2)
        delta = a2_deg - a1_deg
        if delta > 180: delta -= 360
        elif delta < -180: delta += 360
        ts = np.linspace(0.0, 1.0, n)
        ang = np.deg2rad(a1_deg + ts*delta)
        x = cx + r*np.cos(ang); y = cy + r*np.sin(ang)
        return np.column_stack([x,y])

class GeoDSL:
    def __init__(self, display_size):
        self.display_size = display_size

    def point(self, x, y):
        return Point(x, y)

    def line(self, a, b):
        return Line(a, b)

    def simple_point(self, p):
        # Used for evaluating the x and y symbolic expresssions to floats.
        # To be used for performance issues if symbolic expressions become
        # overly complex.
        return Point(float(p.x), float(p.y), evaluate=False)

    def circle_by_compass(self, center, p1, p2):
        return Circle(center, p1.distance(p2))

    def circle_by_center_and_radius(self, c: Point, r: float) -> Circle:
        return Circle(c, r)

    def circle_by_center_and_point(self, center, point):
        """Create a circle given the center and a point on the circumference."""
        return self.circle_by_center_and_radius(center, center.distance(point))

    def arc_by_center_and_two_points(self, center, p1, p2):
        """Create a circular arc given a center and two points."""
        return GeoArc(center, p1, p2) 

    def perpendicular_segment(self, line, point):
        perp = line.perpendicular_segment(point)
        return perp

    def translate_x(self, obj, distance):
        return obj.translate(distance, 0)

    def translate_y(self, obj, distance):
        return obj.translate(0, distance)

    def reflect(self, obj, reflection_obj):
        """Reflect a geometric object across a line or around a point."""
        if isinstance(reflection_obj, Line):
            return obj.reflect(reflection_obj)
        elif isinstance(reflection_obj, Point):
            return obj.rotate(pi, reflection_obj)
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
        perpendicular_line = line_between.perpendicular_line(p2)
        big_intersection_point = big_circle.intersection(perpendicular_line)[0]

        midpoint = big_intersection_point.midpoint(p2)
        small_circle = self.circle_by_center_and_point(midpoint, p2)

        cross_line = self.line(midpoint, p1)
        cross_intersections = cross_line.intersection(small_circle)

        cross_intersection = self.pick_point_closest_to(p1, cross_intersections)
        golden_circle = self.circle_by_center_and_point(p1, cross_intersection)
        golden_intersections = line_between.intersection(golden_circle)
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
        if not circle.center.intersection(line):
            warnings.warn("Calculating tangent circle with a non-radial line", Warning, stacklevel=2)

        intersections = circle.intersection(line)
        intersection = self.pick_point_closest_to(closest_point, intersections)

        helper_circle = self.circle_by_center_and_radius(intersection, radius)
        helper_intersections = helper_circle.intersection(line)
        if inner:
            new_circle_center = self.pick_point_closest_to(circle.center, helper_intersections)
        else:
            new_circle_center = self.pick_point_furthest_from(circle.center, helper_intersections)

        new_circle = self.circle_by_center_and_radius(new_circle_center, radius)

        return new_circle, intersection

    def blend_two_circles(self, blender_radius, circle_1, circle_2):
        helper_1_radius = circle_1.radius - blender_radius
        helper_1_circle = self.circle_by_center_and_radius(circle_1.center, helper_1_radius)
        helper_2_radius = circle_2.radius - blender_radius
        helper_2_circle = self.circle_by_center_and_radius(circle_2.center, helper_2_radius)

        helper_circles_intersections = helper_1_circle.intersection(helper_2_circle)

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

        blender_intersections_1 = blender_circle.intersection(circle_1)
        blender_intersections_2 = blender_circle.intersection(circle_2)

        return blender_circle, blender_intersections_1[0], blender_intersections_2[0]
   
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
