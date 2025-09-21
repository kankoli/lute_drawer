from abc import ABC, abstractmethod
from typing import final, override
from geo_dsl import GeoDSL
import sympy

from abc import ABC, abstractmethod
import numpy as np

from lute_renderers import SvgRenderer

class TopArc(ABC):
    @abstractmethod
    def _get_top_arc_radius(self):
        pass

    def _make_top_arc_circle(self):
        self.top_arc_radius = self._get_top_arc_radius()
        self.top_arc_center = self.geo.translate_y(self.form_side, self.top_arc_radius)
        self.top_arc_circle = self.geo.circle_by_center_and_radius(self.top_arc_center, self.top_arc_radius)

class TopArc_Type1(TopArc):
    @override
    def _get_top_arc_radius(self):
        return 4 * self.unit

class TopArc_Type2(TopArc):
    @override
    def _get_top_arc_radius(self):
        return 5 * self.unit

class TopArc_Type3(TopArc):
    @override
    def _get_top_arc_radius(self):
        return 6 * self.unit

class TopArc_Type4(TopArc):
    @override
    def _get_top_arc_radius(self):
        return 7 * self.unit

class TopArc_Type10(TopArc):
    @override
    def _get_top_arc_radius(self):
        return 13 * self.unit


class Neck(ABC):
    @abstractmethod
    def _make_neck_joint_fret(self):
        pass

class Neck_ThruTop2(Neck):
    """
    Intersect the top arc with a line through top_2 and top_arc_center.
    Draw a circle through this point with center top_2.
    Intersect the circle with the spine to get the neck joint.

    Warning: top_arc_center depends on the particular TopArc class used
    """
    @abstractmethod
    def _make_top_2_point():
        pass

    @override
    def _make_neck_joint_fret(self):
        self._make_top_2_point()

        helper_line = self.geo.line(self.top_2, self.top_arc_center)
        helper_point = self.geo.pick_point_closest_to(self.form_top, helper_line.intersection(self.top_arc_circle))
        helper_circle = self.geo.circle_by_center_and_point(self.top_2, helper_point)
        self.point_neck_joint = self.geo.pick_point_closest_to(self.form_top, helper_circle.intersection(self.spine))

class Neck_DoubleGolden(Neck):
    @abstractmethod
    def _make_top_2_point():
        pass

    @override
    def _make_neck_joint_fret(self):
        self._make_top_2_point()

        first_golden_point = self.geo.golden_ratio_divider(self.top_2, self.form_top)
        self.point_neck_joint = self.geo.golden_ratio_divider(self.form_top, first_golden_point)

class Neck_Quartered(Neck):
    @override
    def _make_neck_joint_fret(self):
        self._make_top_2_point()

        self.point_neck_joint = self.geo.translate_x(self.form_top, self.unit / 4)

class SoundholeAt(ABC):
    @abstractmethod
    def _get_soundhole_center(self):
        pass

class SoundholeAt_NeckBridgeMidpoint(SoundholeAt):
    @override
    def _get_soundhole_center(self):
        return self.point_neck_joint.midpoint(self.bridge)


class Soundhole(ABC):
    @abstractmethod
    def _get_soundhole_radius(self):
        pass

class Soundhole_OneThirdOfSegment(Soundhole):
    @override
    def _get_soundhole_radius(self):
        self.soundhole_perpendicular = self.spine.perpendicular_line(self._get_soundhole_center())
        soundhole_perpendicular_intersection = self.geo.pick_point_closest_to(self.spine, self.top_arc_circle.intersection(self.soundhole_perpendicular))

        # Let the whole line segment between the two top-arcs be 6 units, then the radius should be 1 unit
        # We worked with half of that line segment, only measuring towards one side. Hence division by 3
        return self._get_soundhole_center().distance(soundhole_perpendicular_intersection) / 3

class Soundhole_GoldenRatiofOfSegment(Soundhole):
    @override
    def _get_soundhole_radius(self):
        self.soundhole_perpendicular = self.spine.perpendicular_line(self._get_soundhole_center())
        soundhole_perpendicular_intersection = self.geo.pick_point_closest_to(self.spine, self.top_arc_circle.intersection(self.soundhole_perpendicular))

        self.halfsegment_length = self._get_soundhole_center().distance(soundhole_perpendicular_intersection)
        return self.halfsegment_length - self.halfsegment_length / sympy.S.GoldenRatio # small side of golden ratio


class Soundhole_HalfUnit(Soundhole):
    @override
    def _get_soundhole_radius(self):
        return self.unit / 2

class Soundhole_ThreeQuarters(Soundhole):
    @override
    def _get_soundhole_radius(self):
        return 3 * self.unit / 4

class NoSoundhole(Soundhole):
    @override
    def _make_soundhole(self):
        pass

    @override
    def _get_soundhole_center(self):
        pass

    @override
    def _get_soundhole_radius(self):
        pass

class SmallSoundhole(ABC):
    @override
    def _make_soundhole(self):
        super()._make_soundhole()
        self._make_small_soundholes()

    @abstractmethod
    def _make_small_soundholes(self):
        pass

    @override
    def _make_template_points(self):
        super()._make_template_points()
        self.template_points.extend([self.small_soundhole_centers[0]])

    @override
    def _make_template_objects(self):
        super()._make_template_objects()
        self.template_objects.extend([*self.small_soundhole_centers, *self.small_soundhole_circles])

    @override
    def _make_full_view_objects(self):
        super()._make_full_view_objects()
        self.full_view_objects.extend([*self.small_soundhole_centers, *self.small_soundhole_circles])

class SmallSoundhole_Turkish(SmallSoundhole):
    @override
    def _make_small_soundholes(self):
        soundholes_axis_point = self._get_soundhole_center().midpoint(self.form_center)
        soundholes_line = self.spine.perpendicular_line(soundholes_axis_point)
        self.small_soundhole_locator = self.geo.circle_by_center_and_point(soundholes_axis_point, self.form_center)
        self.small_soundhole_centers = self.small_soundhole_locator.intersection(soundholes_line)
        self.small_soundhole_circles = [self.geo.circle_by_center_and_radius(x, self._get_soundhole_radius()/2) for x in self.small_soundhole_centers]

class SmallSoundhole_Brussels0164(SmallSoundhole):
    @override
    def _make_small_soundholes(self):
        triangle_unit = self._get_soundhole_radius() / 3
        # travel 4 units towards the bridge
        soundholes_axis_point = self.geo.translate_x(self._get_soundhole_center(), 4 * triangle_unit)
        soundholes_line = self.spine.perpendicular_line(soundholes_axis_point)
        self.small_soundhole_locator = self.geo.circle_by_center_and_radius(soundholes_axis_point, 3 * triangle_unit)
        self.small_soundhole_centers = self.small_soundhole_locator.intersection(soundholes_line)
        self.small_soundhole_circles = [self.geo.circle_by_center_and_radius(x, self._get_soundhole_radius() / 4) for x in self.small_soundhole_centers]


class TangentParameter(ABC):
    def __init__(self, geo, previous_circle, point, radius, closest_point):
        self.geo = geo
        self.previous_circle = previous_circle
        self.point = point
        self.radius = radius
        self.closest_point = closest_point

    @property
    def center(self):
        return self.tangent_circle.center

    def calculate_new_circle(self):
        self.line = self.geo.line(self.previous_circle.center, self.point)

        if (isinstance(self.previous_circle, TangentParameter)):
            previous_circle = self.previous_circle.tangent_circle
        else:
            previous_circle = self.previous_circle
        self.tangent_circle, self.tangent_point = self.geo.get_tangent_circle(previous_circle, self.line, self.radius, self.closest_point, True)

        return self.tangent_circle, self.tangent_point

class LowerArcBuilder(ABC):
    ''' Superclass for building lower arcs of a lute,
    connecting the top arc to the lower arc.

    Subclasses define the parameter sets to create tangential
    arcs and a final circle-to-circle blender-circle.
    '''
    @override
    def _build_lower_arcs(self):
        # When there are no tangents defined, blend with top circle
        current_circle = self.top_arc_circle

        tangent_circles = [self.top_arc_circle]
        tangent_points = [self.form_top]
        for tp in self._get_tangent_parameters():
            tangent_circle, tangent_point = tp.calculate_new_circle()
            tangent_circles.append(tangent_circle)
            tangent_points.append(tangent_point)

            current_circle = tangent_circle

        blender_circle, blender_intersection_1, blender_intersection_2 = self.geo.blend_two_circles(self._get_blender_radius(), current_circle, self.bottom_arc_circle)
        tangent_circles.append(blender_circle)
        tangent_circles.append(self.bottom_arc_circle)
        tangent_points.append(blender_intersection_1)
        tangent_points.append(blender_intersection_2)
        tangent_points.append(self.form_bottom)

        self.tangent_circles = tangent_circles
        self.tangent_points = tangent_points

        self.arc_params = []
        for i in range(len(self.tangent_circles)):
            self.arc_params.append([self.tangent_circles[i].center, self.tangent_points[i+1], self.tangent_points[i]])


    @abstractmethod
    def _get_tangent_parameters(self):
        pass

    @abstractmethod
    def _get_blender_radius(self):
        pass

class SimpleBlend(LowerArcBuilder):
    # These classes come AFTER the ones defining _get_tangent_parameters
    # so the list doesn't get overwritten to []
    @override
    def _get_tangent_parameters(self):
        self.tangents = []
        return self.tangents

class SimpleBlend_Unit(SimpleBlend):
    @override
    def _get_blender_radius(self):
        return self.unit

class SimpleBlend_OneAndHalfUnit(SimpleBlend):
    @override
    def _get_blender_radius(self):
        return 1.5 * self.unit

class SimpleBlend_DoubleUnit(SimpleBlend):
    @override
    def _get_blender_radius(self):
        return 2 * self.unit

class SideCircle_TwoUnits(LowerArcBuilder):
    @override
    def _get_tangent_parameters(self):
        super()._get_tangent_parameters()
        self.tangents.append(TangentParameter(self.geo, self.top_arc_circle, self.form_center, 2 * self.unit, self.form_side))
        return self.tangents

class SideStep(LowerArcBuilder):
    @abstractmethod
    def _get_step_circle_radius(self):
        pass

    @override
    def _get_tangent_parameters(self):
        super()._get_tangent_parameters()
        self.step_circle = self.geo.circle_by_center_and_radius(self.form_side, self._get_step_circle_radius())

        step_intersections = self.step_circle.intersection(self.top_arc_circle)
        self.top_arc_finish = self.geo.pick_point_furthest_from(self.form_top, step_intersections)

        self.connector_1 = self.geo.line(self.top_arc_finish, self.top_arc_center)
        self.connector_intersections = self.connector_1.intersection(self.spine)
        second_arc_center = self.connector_intersections[0] # single intersection
        second_arc_radius = second_arc_center.distance(self.top_arc_finish)

        # Special case where the new circle's center is determined first, in order to calculate the radius.
        # The superclass will re-calculate the center itself.
        self.tangents.append(TangentParameter(self.geo, self.top_arc_circle, second_arc_center, second_arc_radius, self.form_bottom))

        return self.tangents


class Lute(ABC):
    def __init__(self, geo: GeoDSL | None = None, display_size: int = 100):
		# explicit dependency injection; no global singletons
        self.geo = geo if geo is not None else GeoDSL(display_size)

        self._base_construction()

        self._make_spine_points()
        self._make_bottom_arc_circle()
        self._make_soundhole()
        self._build_lower_arcs()
        self.__generate_arcs()

	# ---------------------------------------------------------------------
	# Canonical construction
	# ---------------------------------------------------------------------

    def _get_unit_in_mm(self):
        """ In mm's.
        It will be used to when printing measurements.
        """
        return 100 # dummy value

    def _base_construction(self):
        self.unit = self.geo.display_size #  1/4th of the belly

        self.form_center = self.geo.point(self.geo.display_size * 5.5, self.geo.display_size * 3)

        self.A = self.geo.point(self.form_center.x-self.unit, self.form_center.y-self.unit)
        self.B = self.geo.point(self.A.x + self.unit, self.A.y)

        self.waist_2 = self.geo.translate_y(self.form_center, -self.unit)
        self.form_side = self.geo.translate_y(self.form_center, -2 * self.unit)

        self._make_top_arc_circle()

        self.centerline = self.geo.line(self.top_arc_center, self.form_side)
        self.spine = self.centerline.perpendicular_line(self.form_center)

        # Finding the form_top
        intersections = self.top_arc_circle.intersection(self.spine)
        self.form_top = self.geo.pick_west_point(*intersections)

    @abstractmethod
    def _make_top_arc_circle(self):
        pass

    @abstractmethod
    def _make_spine_points(self):
        pass

    def _make_bottom_arc_circle(self):
        self.bottom_arc_circle = self.geo.circle_by_center_and_point(self.form_top, self.form_bottom)

    @abstractmethod
    def _get_soundhole_center(self):
        pass

    @abstractmethod
    def _get_soundhole_radius(self):
        pass

    def _make_soundhole(self):
        self.soundhole_circle = self.geo.circle_by_center_and_radius(self._get_soundhole_center(), self._get_soundhole_radius())

        self._make_small_soundholes()

    # Will be overridden by SmallSoundhole classes, or not
    def _make_small_soundholes(self):
        return None

    @abstractmethod
    def _build_lower_arcs(self):
        pass

    def __generate_arcs(self):
        self.final_arcs = [self.geo.arc_by_center_and_two_points(*params) for params in self.arc_params]
        self.final_reflected_arcs = [self.geo.reflect(x, self.spine) for x in self.final_arcs]

	# ---------------------------------------------------------------------
	# Measurements
	# ---------------------------------------------------------------------

    def print_measurement(self, name, value):
        converter = self._get_unit_in_mm() / self.unit
        print(f"{name:<30} {converter * value:.2f}")

    def print_measurements(self):
        self._make_measurements()

        print(37 * "=")
        print(f"{type(self).__name__:<30} mm")

        [self.print_measurement(name, value) \
            for (name, value) in self.measurements]

        print(37 * "=")

    def _make_measurements(self):
        self.measurements = [
            ("Unit:", self.unit),
            ("Form Width:", self.__find_form_width()),
            ("Form Length:", self.form_bottom.distance(self.form_top)),
            ("Neck to Bottom:", self.form_bottom.distance(self.point_neck_joint)),
            ("Neck to Bridge:", self.point_neck_joint.distance(self.bridge)),
            ("(1/3-Neck) Scale:", (3 / 2) * self.point_neck_joint.distance(self.bridge)),
            ("(1/3-Neck) Neck length:", (1 / 2) * self.point_neck_joint.distance(self.bridge)),
            ("(Half-Neck) Scale:", 2 * self.point_neck_joint.distance(self.bridge)),
            ("(Half-Neck) Neck:", self.point_neck_joint.distance(self.bridge)),
            ("Neck-joint width:", self.__get_neck_joint_width())
        ]

    def __find_form_width(self):
        current_farthest_distance = 0
        for arc in self.arc_params:
            distance = arc[1].distance(self.spine)
            if distance > current_farthest_distance:
                current_farthest_distance = distance

        return 2 * current_farthest_distance

    def __get_neck_joint_width(self):
        return self.get_form_width_at_point(self.point_neck_joint)

    def get_form_width_at_point(self, point):
        perpendicular_line = self.spine.perpendicular_line(point)

        arc_intersection = self.geo.pick_point_closest_to(self.spine,self.top_arc_circle.intersection(perpendicular_line))

        width_at_point = 2 * point.distance(arc_intersection)

        return width_at_point

	# ---------------------------------------------------------------------
	# Drawing
	# ---------------------------------------------------------------------

    def draw_all(self):
        self.draw_template()
        self.draw_helpers()
        self.draw_full_view()

    def draw_template(self):
        template_top = self.geo.translate_x(self.form_top, -self.unit / 4)
        template_bottom = self.geo.translate_x(self.form_bottom, self.unit / 4)
        self.template_spine = self.geo.line(template_top, template_bottom)

        end_neck_block = self.geo.translate_x(self.point_neck_joint, self.unit)
        intermediate_points = self.geo.divide_distance(end_neck_block, self.form_bottom, 10)

        self.template_points = [
            template_top, \
            self.form_top, \
            self.point_neck_joint, \
            end_neck_block,
            self.form_center, \
            self.bridge, \
            # beginning of end-block
            self.geo.translate_x(self.form_bottom, -self.unit/8), \
            self.form_bottom, \
            template_bottom, \
        ]

        soundhole_center = self._get_soundhole_center()
        if soundhole_center is not None:
            self.template_points.append(soundhole_center)

        self.template_points.extend(intermediate_points)

        self.template_lines = [self.geo.perpendicular_segment(self.spine, self.geo.point(p.x, p.y - 3 * self.unit)) for p in self.template_points]

        self.template_objects = [
            self.A, self.B, \
            self.template_spine, \
            *self.final_arcs
        ]

        if hasattr(self, 'soundhole_circle'):
            self.template_objects.append(self.soundhole_circle)

        self.template_objects.extend(self.template_lines)
        self.template_objects.extend(self.template_points)

        self.__render_svg(self.template_objects, '_template.svg')

    def draw_helpers(self):
        self.helper_objects = [
            self.A, self.B, \
            self.form_top, self.form_center, self.form_bottom, self.form_side, self.spine, self.centerline, \
            self.point_neck_joint, \
            self.top_arc_circle, self.bottom_arc_circle, \
            self.bridge
        ]

        self.helper_objects.extend(self.tangent_circles)
        self.helper_objects.extend(self.tangent_points)

        if hasattr(self, 'soundhole_circle'):
            self.helper_objects.append(self.soundhole_circle)

        soundhole_center = self._get_soundhole_center()
        if soundhole_center is not None:
            self.helper_objects.append(soundhole_center)

        self.__render_svg(self.helper_objects, '_helpers.svg')

    def draw_full_view(self):
        self.full_view_objects = [
            self.A, self.B, \
            self.form_top, self.form_center, self.form_bottom, self.form_side, \
            self.point_neck_joint, \
            self.bridge, \
            *self.final_arcs, \
            *self.final_reflected_arcs
        ]

        if hasattr(self, 'soundhole_circle'):
            self.full_view_objects.append(self.soundhole_circle)

        soundhole_center = self._get_soundhole_center()
        if soundhole_center is not None:
            self.full_view_objects.append(soundhole_center)

        self.__render_svg(self.full_view_objects, '_fullview.svg')

    def __render_svg(self, objs, file_suffix):
        svg_size = [self.geo.display_size*9, self.geo.display_size*6]
        renderer = SvgRenderer(type(self).__name__ + file_suffix, svg_size)
        renderer.draw(objs)

class LuteType3(TopArc_Type3, Lute):
    pass

class Brussels0404(SimpleBlend_DoubleUnit, Soundhole_HalfUnit, LuteType3):
    override
    def _make_spine_points(self):
        half_vesica_piscis_circle = self.geo.circle_by_center_and_radius(self.waist_2, 2*self.unit)
        self.vesica_piscis_intersections = self.spine.intersection(half_vesica_piscis_circle)
        # form bottom is described by jdowning as 2 units from form center, unlike below
        self.form_bottom = self.geo.pick_point_furthest_from(self.form_top, self.vesica_piscis_intersections)
        self.bridge = self.geo.translate_x(self.form_bottom, -self.unit)

    @override
    def _get_soundhole_center(self):
        return self.geo.pick_point_closest_to(self.form_top, self.vesica_piscis_intersections)

    @override
    def _make_soundhole(self):
        super()._make_soundhole()
        self._make_neck_joint_fret()

    @override
    def _make_neck_joint_fret(self):
        self.point_neck_joint = self.geo.reflect(self.bridge, self._get_soundhole_center())


class LuteType4(TopArc_Type4, Lute):
    @override
    def _make_top_2_point(self):
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)


class ManolLavta(SimpleBlend, Soundhole_OneThirdOfSegment, SoundholeAt_NeckBridgeMidpoint, LuteType4, Neck_ThruTop2):
    # https://www.mikeouds.com/messageboard/viewthread.php?tid=12255
    # https://www.mikeouds.com/messageboard/files.php?pid=80797&aid=18637

    @override
    def _get_unit_in_mm(self):
        return 285 / 4

    @override
    def _make_bottom_arc_circle(self):
        bottom_arc_center = self.geo.translate_x(self.form_top, -2 * self.unit)
        self.bottom_arc_circle = self.geo.circle_by_center_and_point(bottom_arc_center, self.form_bottom)

    @override
    def _make_spine_points(self):
        self._make_neck_joint_fret()

        # self.form_bottom = self.geo.translate_x(self.form_top, 6 * self.unit)
        self.form_bottom = self.geo.reflect(self.geo.divide_distance(self.form_top, self.form_center, 3)[-1], self.form_center)

        self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 5

        self.bridge = self.geo.translate_x(self.form_bottom, -self.vertical_unit) # negation is important

    @override
    def _get_tangent_parameters(self):
        super()._get_tangent_parameters()

        tangent = self.top_arc_circle

        spine_point = self.form_center
        radius = self.unit * 3
        tangent = TangentParameter(self.geo, tangent, spine_point, radius, self.form_center)
        self.tangents.append(tangent)

        return self.tangents

    @override
    def _get_blender_radius(self):
        return self.vertical_unit

    @override
    def _make_measurements(self):
        super()._make_measurements()
        self.measurements.append(("Soundhole radius:", self._get_soundhole_radius()))
        # self.measurements.append(("Soundhole segment half:", self.halfsegment_length))
        self.measurements.append(("Vertical Unit:", self.vertical_unit))

class ManolLavta_Type3(LuteType3, ManolLavta):
    pass

class ManolLavta_Athens(Soundhole_GoldenRatiofOfSegment, ManolLavta):
    pass


class LuteType2(TopArc_Type2, Lute):
    """
                                                                top_arc_center


                                                                 =======



                                                                 =======



                                                                 =======

    form_top ======= top_2 ======= top_3  ======= top_4 ======= form_center

                                                                 =======

                                                                 waist_2

                                                                 =======

                                                                 form_side

    """

    @override
    def _make_top_2_point(self):
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)
        self.top_3 = self.geo.translate_x(self.form_top, 2 * self.unit)
        self.top_4 = self.geo.translate_x(self.form_top, 3 * self.unit)


class VesicaPiscesOud(SimpleBlend_Unit, SmallSoundhole_Turkish, Soundhole_HalfUnit, TopArc_Type2, Neck_DoubleGolden, Lute):
    @override
    def _get_unit_in_mm(self):
        return 366 / 4

    @override
    def _make_top_2_point(self):
        # Dividing the distance by 4 and getting the top_3 is a shortcut to
        # creating a half-sized vesica pisces
        self.top_2, self.top_3, self.top_4 = self.geo.divide_distance(self.form_top, self.form_center, 4)


    @override
    def _make_spine_points(self):
        self._make_top_2_point() # We need those points already

        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.reflect(self.top_4, self.form_center)

        self._make_neck_joint_fret()

    @override
    def _get_soundhole_center(self):
        return self.top_3.midpoint(self.top_4)

    @override
    def _make_helper_objects(self):
        super()._make_helper_objects()
        self.helper_objects.extend([self.top_2, self.top_3, self.top_4])

class TurkishOud2(SmallSoundhole_Turkish, Soundhole_HalfUnit, LuteType2, Neck_Quartered):
    """ Attempt to construct without a vertical unit
        Placing the neck joint first,
        then placing the soundhole center at midpoint of top_3 and top_4 (experimentation needed).
        Neck joint to soundhole center becomes X. Then the bridge and the form_bottom could be set.

        Reference: Eren Ozek's Turkish Oud proportions
        https://www.mikeouds.com/messageboard/files.php?pid=78208&aid=17317

        P.S. The problem with the reference proportions is that it doesn't
        inform on how the neck joint is placed. I'll experiment with the neck
        classes I got...
    """
    @override
    def _get_unit_in_mm(self):
        return 366 / 4 # Form width in mm / 4

    @override
    def _get_soundhole_center(self):
        # self.geo.midpoint(self.top_3, self.top_4)
        # self.geo.golden_ratio_divider(self.form_top, self.form_center)
        # self.geo.midpoint(self.top_2, self.form_center)
        return self.geo.golden_ratio_divider(self.top_4, self.top_3)

    @override
    def _make_spine_points(self):
        self._make_neck_joint_fret()

        vertical_X = self.point_neck_joint.distance(self._get_soundhole_center())
        # Neck to form bottom is said to be X + X + 0.5 X
        self.bridge = self.geo.translate_x(self._get_soundhole_center(), vertical_X)
        self.form_bottom = self.geo.translate_x(self.bridge, vertical_X / 2)

    @override
    def _make_helper_objects(self):
        super()._make_helper_objects()
        self.helper_objects.extend([self.top_2, self.top_3, self.top_4])

class TurkishOud2_1(SimpleBlend, TurkishOud2):
    @override
    def _get_blender_radius(self):
        return self.small_soundhole_centers[0].distance(self._get_soundhole_center())

class TurkishOud2_2(SideCircle_TwoUnits, SimpleBlend_Unit, TurkishOud2):
    pass

class TurkishOud2_3(SideStep, SimpleBlend_Unit, TurkishOud2):
    @override
    def _get_step_circle_radius(self):
        return self.unit / 4

class TurkishOud2_4(SimpleBlend_Unit, TurkishOud2):
    pass

class TurkishOud(LuteType2, Neck_Quartered):
    @override
    def _get_unit_in_mm(self):
        return 366 / 4

    @override
    def _get_soundhole_center(self):
        """ Width of the soundboard (4 regular unit)
        should be 3/4th of the segment from neck-joint to form-bottom
        which is then divided into 5 and placing the bridge at 1 unit
        and the soundhole at 3 units (half-way)

        So, 4 regular units equals 3 /4 (5 vertical units), which means
        1 vertical unit is equal to 16/15 regular unit
        """
        self.vertical_unit = 16 * self.unit / 15

        return self.geo.translate_x(self.point_neck_joint, 2 * self.vertical_unit)

    @override
    def _make_spine_points(self):
        self._make_neck_joint_fret()

        """ Width of the soundboard (4 regular unit)
        should be 3/4th of the segment from neck-joint to form-bottom
        which is then divided into 5 and placing the bridge at 1 unit
        and the soundhole at 3 units (half-way)

        So, 4 regular units equals 3 /4 (5 vertical units), which means
        1 vertical unit is equal to 16/15 regular unit
        """
        self.vertical_unit = 16 * self.unit / 15

        soundhole_center = self._get_soundhole_center()
        self.bridge = self.geo.translate_x(soundhole_center, 2 * self.vertical_unit)
        self.form_bottom = self.geo.translate_x(self.bridge, self.vertical_unit)

class TurkishOudSingleMiddleArc(SimpleBlend, SmallSoundhole_Turkish, Soundhole_HalfUnit, TurkishOud):
    @override
    def _get_blender_radius(self):
        return 3 * self.small_soundhole_centers[0].distance(self.small_soundhole_centers[1]) / 4

class TurkishOudDoubleMiddleArcs(SideCircle_TwoUnits, SimpleBlend_OneAndHalfUnit, SmallSoundhole_Turkish, Soundhole_HalfUnit, TurkishOud):
    pass

class TurkishOudComplexLowerBout(SideStep, SimpleBlend_Unit, SmallSoundhole_Turkish, Soundhole_HalfUnit, TurkishOud):
    @override
    def _get_step_circle_radius(self):
        return self.unit / 4

class TurkishOudSoundholeThird(SimpleBlend, SmallSoundhole_Turkish, Soundhole_OneThirdOfSegment, TurkishOud):
    @override
    def _get_blender_radius(self):
        return 3 * self.small_soundhole_centers[0].distance(self.small_soundhole_centers[1]) / 4

class IstanbulLavta(SideStep, SimpleBlend, Soundhole_OneThirdOfSegment, SoundholeAt_NeckBridgeMidpoint, LuteType2, Neck_ThruTop2):
    @override
    def _get_unit_in_mm(self):
        return 300 / 4

    @override
    def _make_spine_points(self):
        self._make_neck_joint_fret()

        self.form_bottom = self.geo.translate_x(self.form_top, 6 * self.unit)

        self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 4

        self.bridge = self.geo.translate_x(self.form_bottom, -self.vertical_unit) # negation is important

    @override
    def _get_step_circle_radius(self):
        return self.unit / 2

    @override
    def _get_blender_radius(self):
        return 5*self.unit/4

class IkwanAlSafaOud(SimpleBlend_DoubleUnit, Soundhole_HalfUnit, SoundholeAt_NeckBridgeMidpoint, LuteType2, Neck_Quartered):
    @override
    def _make_spine_points(self):
        self._make_neck_joint_fret()

        self.bridge = self.geo.translate_x(self.form_center, 3 * self.unit / 4)
        self.form_bottom = self.geo.translate_x(self.form_center, 2*self.unit)

class HannaNahatOud(SideCircle_TwoUnits, SimpleBlend, Soundhole_ThreeQuarters, SoundholeAt_NeckBridgeMidpoint, LuteType2, Neck_Quartered):
    @override
    def _get_unit_in_mm(self):
        return 365/4

    @override
    def _make_spine_points(self):
        self._make_neck_joint_fret()

        self.bridge = self.geo.translate_x(self.form_center, 3 * self.unit / 4)
        self.form_bottom = self.geo.translate_x(self.bridge, self.unit)

    @override
    def _get_blender_radius(self):
        return (2 - 3 /4 ) * self.unit # difference of the bridge-form_center difference from the form half (2 units)


class LuteType1(TopArc_Type1, Lute):
    """
    Based on a vesica pisces construction. form_top is the top vesica pisces point.
    top_3 is the top point of a half-sized vesica pisces, and form_bottom is the bottom point.

                                    form_top


                                    top_2


                                    top_3


                                    top_4


    form_size        <2 units>        form_center     <2 units> arc_center (radius = 4 units)


                                    bridge (sometimes)


                                    form_bottom
    """
    @override
    def _make_spine_points(self):
        # Dividing the distance by 4 and getting the top_3 is a shortcut to
        # creating a second, half-sized vesica pisces
        self.top_2, self.top_3, self.top_4 = self.geo.divide_distance(self.form_top, self.form_center, 4)

        # Following the double vesica pisces construction...
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.reflect(self.top_4, self.form_center)

class HochLavta(SimpleBlend, LuteType1, Neck_ThruTop2):
    @override
    def _make_top_2_point(self):
        # top_2 is already made within base _make_spine_points
        pass

    @override
    def _get_soundhole_center(self):
        return self.top_3.midpoint(self.top_4.midpoint(self.form_center))

    @override
    def _get_soundhole_radius(self):
        return self._get_soundhole_center().distance(self.top_3)

    @override
    def _make_soundhole(self):
        super()._make_soundhole()
        self._make_neck_joint_fret()

    @override
    def _get_blender_radius(self):
        return float(self._get_soundhole_center().distance(self.waist_2))

    @override
    def _make_helper_objects(self):
        super()._make_helper_objects()
        experimental_neck_joint = self.geo.reflect(self.bridge, self._get_soundhole_center())
        self.helper_objects.extend([experimental_neck_joint])

        self.print_measurement("Neck at experimental point", converter * self.get_form_width_at_point(experimental_neck_joint))

class LavtaSmallThreeCourse(SimpleBlend_Unit, Soundhole_OneThirdOfSegment, SoundholeAt_NeckBridgeMidpoint, LuteType1, Neck_ThruTop2):
    @override
    def _make_top_2_point(self):
        # top_2 is already made within base _make_spine_points
        pass

    @override
    def _make_soundhole(self):
        self._make_neck_joint_fret()
        super()._make_soundhole()

class Brussels0164(SimpleBlend, SmallSoundhole_Brussels0164, LuteType1):
    @override
    def _make_spine_points(self):
        super()._make_spine_points()

        self.bridge = self.geo.translate_x(self.form_bottom, -self.unit)

        self._make_neck_joint_fret()

    @override
    def _get_soundhole_center(self):
        return self.geo.divide_distance(self.top_3, self.form_center, 3)[0]

    @override
    def _get_soundhole_radius(self):
        return self.top_3.distance(self._get_soundhole_center())

    @override
    def _make_neck_joint_fret(self):
        self.point_neck_joint = self.geo.reflect(self.bridge, self._get_soundhole_center())

    @override
    def _get_blender_radius(self):
        return float(self._get_soundhole_center().distance(self.form_center))


class LuteType10(TopArc_Type10, Lute):
    @override
    def _make_top_2_point(self):
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)

    @override
    def _make_spine_points(self):
        self.bridge = self.geo.translate_x(self.form_center, self.unit)
        self.form_bottom = self.geo.reflect(self.form_center, self.bridge)

        self._make_neck_joint_fret()

class BaltaSaz(NoSoundhole, SimpleBlend_OneAndHalfUnit, LuteType10, Neck_ThruTop2):
    @override
    def _get_unit_in_mm(self):
        return 200/4

def test_all_lutes():
	lutes = []
	lutes.extend([ lute() for lute in TurkishOud.__subclasses__() ])
	lutes.extend([ IstanbulLavta(),IkwanAlSafaOud(), HannaNahatOud() ])
	lutes.extend([ lute() for lute in LuteType3.__subclasses__() ])
	lutes.extend([ lute() for lute in LuteType4.__subclasses__() ])
	lutes.extend([ lute() for lute in LuteType1.__subclasses__() ])
	lutes.extend([ lute() for lute in LuteType10.__subclasses__() ])
	lutes.extend([ lute() for lute in TurkishOud2.__subclasses__() ])

	print("\n\n\n\n\n")
	[lute.print_measurements() for lute in lutes]
	[lute.drawl_all() for lute in lutes]

def test_single_lute():
	lute = ManolLavta()
	lute.draw_all()
	lute.print_measurements()

def main():
	testing_all = 0
	if testing_all == 1:
		test_all_lutes()
	else:
		test_single_lute()

if __name__ == '__main__':
    main()
