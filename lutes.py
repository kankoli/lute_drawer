from abc import ABC, abstractmethod
from typing import final, override
from GeoDSL import GeoDSL, Point, GeoArc
import os.path

# Create an instance of GeoDSL
geo = GeoDSL()

class TopArc(ABC):
	@abstractmethod
	def _get_top_arc_radius(self):
		pass

	def _make_top_arc_circle(self):
		self.top_arc_radius = self._get_top_arc_radius()
		self.top_arc_center = geo.translate_point_y(self.form_side, self.top_arc_radius)
		self.top_arc_circle = geo.circle_by_center_and_radius(self.top_arc_center, self.top_arc_radius)

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

		helper_line = geo.line(self.top_2, self.top_arc_center)
		helper_point = geo.pick_point_closest_to(self.form_top, geo.intersection(helper_line, self.top_arc_circle))
		helper_circle = geo.circle_by_center_and_point(self.top_2, helper_point)
		self.point_neck_joint = geo.pick_point_closest_to(self.form_top, geo.intersection(helper_circle, self.spine))

class Neck_DoubleGolden(Neck):
	@abstractmethod
	def _make_top_2_point():
		pass

	@override
	def _make_neck_joint_fret(self):
		self._make_top_2_point()

		first_golden_point = geo.golden_ratio_divider(self.top_2, self.form_top)
		self.point_neck_joint = geo.golden_ratio_divider(self.form_top, first_golden_point)

class Neck_Quartered(Neck):
	@override
	def _make_neck_joint_fret(self):
		self._make_top_2_point()

		self.point_neck_joint = geo.translate_point_x(self.form_top, self.quarter_unit)

class SoundholeAt(ABC):
	@abstractmethod
	def _get_soundhole_center(self):
		pass

class SoundholeAt_NeckBridgeMidpoint(SoundholeAt):
	@override
	def _get_soundhole_center(self):
		return geo.midpoint(self.point_neck_joint, self.bridge)


class Soundhole(ABC):
	@abstractmethod
	def _get_soundhole_radius(self):
		pass

class Soundhole_OneThirdOfSegment(Soundhole):
	@override
	def _get_soundhole_radius(self):
		self.soundhole_perpendicular = geo.perpendicular_line(self.spine, self._get_soundhole_center())
		soundhole_perpendicular_intersection = geo.pick_point_closest_to(self.spine, geo.intersection(self.top_arc_circle, self.soundhole_perpendicular))

		# Let the whole line segment between the two top-arcs be 6 units, then the radius should be 1 unit
		# We worked with half of that line segment, only measuring towards one side. Hence division by 3
		return self._get_soundhole_center().distance(soundhole_perpendicular_intersection) / 3

class Soundhole_HalfUnit(Soundhole):
	@override
	def _get_soundhole_radius(self):
		return self.half_unit

class Soundhole_ThreeQuarters(Soundhole):
	@override
	def _get_soundhole_radius(self):
		return 3 * self.unit / 4


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
		soundholes_axis_point = geo.midpoint(self._get_soundhole_center(), self.form_center)
		soundholes_line = geo.perpendicular_line(self.spine, soundholes_axis_point)
		self.small_soundhole_locator = geo.circle_by_center_and_point(soundholes_axis_point, self.form_center)
		self.small_soundhole_centers = geo.intersection(self.small_soundhole_locator, soundholes_line)
		self.small_soundhole_circles = [geo.circle_by_center_and_radius(x, self._get_soundhole_radius()/2) for x in self.small_soundhole_centers]

class SmallSoundhole_Brussels0164(SmallSoundhole):
	@override
	def _make_small_soundholes(self):
		triangle_unit = self._get_soundhole_radius() / 3
		# travel 4 units towards the bridge
		soundholes_axis_point = geo.translate_point_x(self._get_soundhole_center(), 4 * triangle_unit)
		soundholes_line = geo.perpendicular_line(self.spine, soundholes_axis_point)
		self.small_soundhole_locator = geo.circle_by_center_and_radius(soundholes_axis_point, 3 * triangle_unit)
		self.small_soundhole_centers = geo.intersection(self.small_soundhole_locator, soundholes_line)
		self.small_soundhole_circles = [geo.circle_by_center_and_radius(x, self._get_soundhole_radius() / 4) for x in self.small_soundhole_centers]


class CircleBlender(ABC):
	@abstractmethod
	def _make_blender_side_circle(self):
		pass

	@abstractmethod
	def _make_arcs(self):
		pass

class Blend_Classic(CircleBlender):
	@override
	def _make_blender_side_circle(self):
		self.side_circle = self.top_arc_circle

	@override
	def _make_arcs(self):
		self.arc_params = [ \
			[self.top_arc_center, self.blender_intersection_1, self.form_top], \
			[self.blender_circle.center, self.blender_intersection_2, self.blender_intersection_1], \
			[self.form_top, self.form_bottom, self.blender_intersection_2] \
		]

class Blend_WithCircle(CircleBlender):
	@override
	def _make_arcs(self):
		self.arc_params = [ \
			[self.top_arc_center, self.top_arc_finish, self.form_top], \
			[self.side_circle.center, self.blender_intersection_1, self.top_arc_finish], \
			[self.blender_circle.center, self.blender_intersection_2, self.blender_intersection_1], \
			[self.form_top, self.form_bottom, self.blender_intersection_2] \
		]

class Blend_SideCircle(Blend_WithCircle):
	@abstractmethod
	def _get_side_circle_radius(self):
		pass

	@override
	def _make_blender_side_circle(self):
		second_arc_radius = self._get_side_circle_radius()
		second_arc_center = geo.translate_point_y(self.form_side, second_arc_radius)
		self.side_circle = geo.circle_by_center_and_radius(second_arc_center, second_arc_radius) # Readily blended with top_arc_circle
		self.top_arc_finish = self.form_side # Shortcut to intersection of the top circle and side circle

class Blend_StepCircle(Blend_WithCircle):
	@abstractmethod
	def _get_step_circle_radius(self):
		pass

	@override
	def _make_blender_side_circle(self):
		self.step_circle = geo.circle_by_center_and_radius(self.form_side, self._get_step_circle_radius())

		step_intersections = geo.intersection(self.step_circle, self.top_arc_circle)
		self.top_arc_finish = geo.pick_point_furthest_from(self.form_top, step_intersections)

		self.connector_1 = geo.line(self.top_arc_finish, self.top_arc_center)
		self.connector_intersections = geo.intersection(self.connector_1, self.spine)
		second_arc_center = self.connector_intersections[0] # single intersection

		second_arc_radius = second_arc_center.distance(self.top_arc_finish)
		self.side_circle = geo.circle_by_center_and_radius(second_arc_center, second_arc_radius)

	@override
	def _make_arcs(self):
		self.arc_params = [ \
			[self.top_arc_center, self.top_arc_finish, self.form_top], \
			[self.side_circle.center, self.blender_intersection_1, self.top_arc_finish], \
			[self.blender_circle.center, self.blender_intersection_2, self.blender_intersection_1], \
			[self.form_top, self.form_bottom, self.blender_intersection_2] \
		]


class BlendWith(ABC):
	@abstractmethod
	def _get_blender_radius(self):
		pass

class BlendWith_Unit(BlendWith):
	@override
	def _get_blender_radius(self):
		return self.unit

class BlendWith_DoubleUnit(BlendWith):
	@override
	def _get_blender_radius(self):
		return 2 * self.unit


class Lute(ABC):
	@staticmethod
	def print_meaurement(measurement_name, value):
		print(f"{measurement_name:<30} {value:.2f}")

	def __init__(self):
		self._base_construction()

		self._make_spine_points()
		self.__make_bottom_arc_circle()
		self._make_soundhole()
		self._get_blender_radius()
		self.__blend()
		self._make_arcs()
		self.__generate_arcs()

	def _get_unit_length(self):
		""" In mm's.
		It will be used to when printing measurements.
		"""
		return 100 # dummy value

	def _base_construction(self):
		self.double_unit = 300
		self.unit = self.double_unit / 2 #  1/4th of the belly
		self.half_unit = self.unit / 2
		self.quarter_unit = self.half_unit / 2

		self.A = geo.point(150, 600)
		self.B = geo.point(self.A.x + self.unit, self.A.y)

		self.form_center = geo.point(700, 500)
		self.waist_2 = geo.translate_point_y(self.form_center, -self.unit)
		self.form_side = geo.translate_point_y(self.form_center, -2 * self.unit)

		self._make_top_arc_circle()

		self.centerline = geo.line(self.top_arc_center, self.form_side)
		self.spine = geo.perpendicular_line(self.centerline, self.form_center)

		# Finding the form_top
		intersections = geo.intersection(self.top_arc_circle, self.spine)
		self.form_top = geo.pick_west_point(*intersections)


	@abstractmethod
	def _make_top_arc_circle(self):
		pass

	@abstractmethod
	def _make_spine_points(self):
		pass

	def __make_bottom_arc_circle(self):
		self.bottom_arc_circle = geo.circle_by_center_and_point(self.form_top, self.form_bottom)

	@abstractmethod
	def _get_soundhole_center(self):
		pass

	@abstractmethod
	def _get_soundhole_radius(self):
		pass

	def _make_soundhole(self):
		self.soundhole_circle = geo.circle_by_center_and_radius(self._get_soundhole_center(), self._get_soundhole_radius())

		self._make_small_soundholes()

	# Will be overridden by SmallSoundhole classes, or not
	def _make_small_soundholes(self):
		return None

	@abstractmethod
	def _get_blender_radius(self):
		pass

	@abstractmethod
	def _make_blender_side_circle(self):
		pass

	def __blend(self):
		self._make_blender_side_circle()
		self.blender_circle, self.blender_intersection_1, self.blender_intersection_2 = geo.blend_two_circles(self._get_blender_radius(), self.side_circle, self.bottom_arc_circle)

	def __generate_arcs(self):
		self.final_arcs = [geo.arc_by_center_and_two_points(*params) for params in self.arc_params]
		self.final_reflected_arcs = [geo.reflect(x, self.spine) for x in self.final_arcs]

	def draw(self):
		self._make_template_objects()
		self.__dump_template()

		self._make_helper_objects()
		self.__dump_helper()

		self._make_full_view_objects()
		self.__dump_full_view()

	def print_measurements(self):
		print(37 * "=")
		print(f"{type(self).__name__:<30} mm")

		# Top width is 4 * unit, unless the blending narrows it by falling towards the top
		# TODO: Could there a larger blender towards the bottom?
		if (self.blender_intersection_1.x < self.form_side.x):
			form_width = 2 * self.blender_intersection_1.distance(self.spine)
		else:
			form_width = 2 * self.form_side.distance(self.spine)

		measurements = [
			("Unit:", self.unit),
			("Form Width:", form_width),
			("Form Length:", self.form_bottom.distance(self.form_top)),
			("Neck to Bottom:", self.form_bottom.distance(self.point_neck_joint)),
			("(1/3-Neck) Scale:", (3 / 2) * self.point_neck_joint.distance(self.bridge)),
			("(1/3-Neck) Neck length:", (1 / 2) * self.point_neck_joint.distance(self.bridge)),
			("(Half-Neck) Scale:", 2 * self.point_neck_joint.distance(self.bridge)),
			("(Half-Neck) Neck:", self.point_neck_joint.distance(self.bridge)),
			("Neck-joint width:", self.__get_neck_joint_width())
		]

		convert = self._get_unit_length() / self.unit

		[Lute.print_meaurement(measurement_name, convert * measurement_value) \
			for (measurement_name, measurement_value) in measurements]

		print(37 * "=")

	def __get_neck_joint_width(self):
		return self.get_form_width_at_point(self.point_neck_joint)

	def get_form_width_at_point(self, point):
		perpendicular_line = geo.perpendicular_line(self.spine, point)

		arc_intersection = geo.pick_point_closest_to(self.spine,geo.intersection(self.top_arc_circle, perpendicular_line))

		width_at_point = 2 * point.distance(arc_intersection)

		return width_at_point

	def _make_template_points(self):
		self.template_bottom_halving_point = geo.midpoint(self.bridge, self.form_bottom)
		self.template_top = geo.translate_point_x(self.form_top, -self.quarter_unit)
		self.template_bottom = geo.translate_point_x(self.form_bottom, self.quarter_unit)
		self.template_spine = geo.line(self.template_top, self.template_bottom)

		self.template_points = [
			self.template_top, \
			self.form_top, \
			self.point_neck_joint, \
			self.form_center, \
			self.bridge, \
			self.form_bottom, self.template_bottom_halving_point, \
			self.template_bottom, \
		]

		soundhole_center = self._get_soundhole_center()
		if soundhole_center is not None:
			self.template_points.append(soundhole_center)

	def _make_template_lines(self):
		self._make_template_points()
		self.template_lines = [geo.perpendicular_segment(self.spine, geo.point(p.x, p.y - 3 * self.unit)) for p in self.template_points]

	def _make_template_objects(self):
		self._make_template_lines()

		self.template_objects = [
			self.A, self.B, \
			self.template_spine, \
			*self.final_arcs
		]

		if self.soundhole_circle is not None:
			self.template_objects.append(self.soundhole_circle)

		self.template_objects.extend(self.template_lines)

	@final
	def __get_file_name_prefix(self):
		output_dir = "output"
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		return output_dir + "/" + type(self).__name__

	@final
	def __dump_template(self):
		GeoDSL.draw_svg(self.template_objects, self.__get_file_name_prefix() + '_template.svg')

	def _make_helper_objects(self):
		self.helper_objects = [
		    self.A, self.B, \
		    self.form_top, self.form_center, self.form_bottom, self.form_side, self.spine, self.centerline, \
		    self.point_neck_joint, \
		    self.top_arc_circle, self.bottom_arc_circle, self.side_circle, \
		    self.blender_intersection_1, self.blender_intersection_2, self.blender_circle, \
		    self.bridge
	    ]

		if self.soundhole_circle is not None:
			self.helper_objects.append(self.soundhole_circle)

		soundhole_center = self._get_soundhole_center()
		if soundhole_center is not None:
			self.helper_objects.append(soundhole_center)

	@final
	def __dump_helper(self):
		GeoDSL.draw_svg(self.helper_objects, self.__get_file_name_prefix() + '_helpers.svg')

	def _make_full_view_objects(self):
		self.full_view_objects = [
		    self.A, self.B, \
		    self.form_top, self.form_center, self.form_bottom, self.form_side, \
		    self.point_neck_joint, \
		    self.bridge, \
		    *self.final_arcs, \
		    *self.final_reflected_arcs
	    ]

		if self.soundhole_circle is not None:
			self.full_view_objects.append(self.soundhole_circle)

		soundhole_center = self._get_soundhole_center()
		if soundhole_center is not None:
			self.full_view_objects.append(soundhole_center)

	@final
	def __dump_full_view(self):
		GeoDSL.draw_svg(self.full_view_objects, self.__get_file_name_prefix() + '_full_view.svg')

class LuteType3(TopArc_Type3, Lute):
	pass

class Brussels0404(BlendWith_DoubleUnit, Blend_Classic, Soundhole_HalfUnit, LuteType3):
	override
	def _make_spine_points(self):
		half_vesica_piscis_circle = geo.circle_by_center_and_radius(self.waist_2, 2*self.unit)
		self.vesica_piscis_intersections = geo.intersection(self.spine, half_vesica_piscis_circle)
		self.form_bottom = geo.pick_point_furthest_from(self.form_top, self.vesica_piscis_intersections)
		self.bridge = geo.translate_point_x(self.form_bottom, -self.unit)

	@override
	def _get_soundhole_center(self):
		return geo.pick_point_closest_to(self.form_top, self.vesica_piscis_intersections)

	@override
	def _make_soundhole(self):
		super()._make_soundhole()
		self._make_neck_joint_fret()

	@override
	def _make_neck_joint_fret(self):
		self.point_neck_joint = geo.reflect(self.bridge, self._get_soundhole_center())


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
		self.top_2 = geo.translate_point_x(self.form_top, self.unit)
		self.top_3 = geo.translate_point_x(self.form_top, 2 * self.unit)
		self.top_4 = geo.translate_point_x(self.form_top, 3 * self.unit)


class VesicaPiscesOud(BlendWith_Unit, Blend_Classic, SmallSoundhole_Turkish, Soundhole_HalfUnit, TopArc_Type2, Neck_DoubleGolden, Lute):
	@override
	def _get_unit_length(self):
		return 366 / 4

	@override
	def _make_top_2_point(self):
		# Dividing the distance by 4 and getting the top_3 is a shortcut to
		# creating a half-sized vesica pisces
		self.top_2, self.top_3, self.top_4 = geo.divide_distance(self.form_top, self.form_center, 4)


	@override
	def _make_spine_points(self):
		self._make_top_2_point() # We need those points already

		self.form_bottom = geo.reflect(self.top_3, self.form_center)
		self.bridge = geo.reflect(self.top_4, self.form_center)

		self._make_neck_joint_fret()

	@override
	def _get_soundhole_center(self):
		return geo.midpoint(self.top_3, self.top_4)

	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.top_2, self.top_3, self.top_4])

class TurkishOud2(Blend_Classic, SmallSoundhole_Turkish, Soundhole_HalfUnit, LuteType2, Neck_Quartered):
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
	def _get_unit_length(self):
		return 366 / 4 # Form width in mm / 4

	@override
	def _get_soundhole_center(self):
		# geo.midpoint(self.top_3, self.top_4)
		# geo.golden_ratio_divider(self.form_top, self.form_center)
		# geo.midpoint(self.top_2, self.form_center)
		return geo.golden_ratio_divider(self.top_4, self.top_3)

	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		vertical_X = self.point_neck_joint.distance(self._get_soundhole_center())
		# Neck to form bottom is said to be X + X + 0.5 X
		self.bridge = geo.translate_point_x(self._get_soundhole_center(), vertical_X)
		self.form_bottom = geo.translate_point_x(self.bridge, vertical_X / 2)

	@override
	def _get_blender_radius(self):
		return self.small_soundhole_centers[0].distance(self._get_soundhole_center())

	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.top_2, self.top_3, self.top_4])


class TurkishOud(LuteType2, Neck_DoubleGolden):
	@override
	def _get_unit_length(self):
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

		return geo.translate_point_x(self.point_neck_joint, 2 * self.vertical_unit)

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
		self.bridge = geo.translate_point_x(soundhole_center, 2 * self.vertical_unit)
		self.form_bottom = geo.translate_point_x(self.bridge, self.vertical_unit)

class TurkishOudSingleMiddleArc(Blend_Classic, SmallSoundhole_Turkish, Soundhole_HalfUnit, TurkishOud):
	@override
	def _get_blender_radius(self):
		return 3 * self.small_soundhole_centers[0].distance(self.small_soundhole_centers[1]) / 4

class TurkishOudDoubleMiddleArcs(BlendWith_Unit, Blend_SideCircle, SmallSoundhole_Turkish, Soundhole_HalfUnit, TurkishOud):
	@override
	def _get_side_circle_radius(self):
		return 2*self.unit

class TurkishOudComplexLowerBout(BlendWith_Unit, Blend_StepCircle, SmallSoundhole_Turkish, Soundhole_HalfUnit, TurkishOud):
	@override
	def _get_step_circle_radius(self):
		return self.unit / 4

	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.connector_1, self.step_circle])
		self.helper_objects.append(self.top_arc_center)
		self.helper_objects.append(self.top_arc_finish)
		self.helper_objects.append(self.connector_intersections[0])

		self.helper_objects.extend([self.top_2, self.top_3, self.top_4])

class TurkishOudSoundholeThird(Blend_Classic, SmallSoundhole_Turkish, Soundhole_OneThirdOfSegment, TurkishOud):
	@override
	def _get_blender_radius(self):
		return 3 * self.small_soundhole_centers[0].distance(self.small_soundhole_centers[1]) / 4

class IstanbulLavta(Blend_StepCircle, Soundhole_OneThirdOfSegment, SoundholeAt_NeckBridgeMidpoint, LuteType2, Neck_ThruTop2):
	@override
	def _get_unit_length(self):
		return 300 / 4

	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.top_6 = geo.translate_point_x(self.form_top, 5 * self.unit)
		self.form_bottom = geo.translate_point_x(self.form_top, 6 * self.unit)

		self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 4

		self.bridge = geo.translate_point_x(self.form_bottom, -self.vertical_unit) # negation is important

	@override
	def _get_step_circle_radius(self):
		return self.unit / 2

	@override
	def _get_blender_radius(self):
		return 5*self.unit/4

class IkwanAlSafaOud(BlendWith_DoubleUnit, Blend_Classic, Soundhole_HalfUnit, SoundholeAt_NeckBridgeMidpoint, LuteType2, Neck_Quartered):
	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.bridge = geo.translate_point_x(self.form_center, 3 * self.unit / 4)
		self.form_bottom = geo.translate_point_x(self.form_center, 2*self.unit)

class HannaNahatOud(Blend_SideCircle, Soundhole_ThreeQuarters, SoundholeAt_NeckBridgeMidpoint, LuteType2, Neck_Quartered):
	@override
	def _get_unit_length(self):
		return 365/4

	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.bridge = geo.translate_point_x(self.form_center, 3 * self.unit / 4)
		self.form_bottom = geo.translate_point_x(self.bridge, self.unit)

	@override
	def _get_side_circle_radius(self):
		return 2*self.unit

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


	form_size		<2 units>		form_center 	<2 units> arc_center (radius = 4 units)


									bridge (sometimes)


									form_bottom
	"""
	@override
	def _make_spine_points(self):
		# Dividing the distance by 4 and getting the top_3 is a shortcut to
		# creating a second, half-sized vesica pisces
		self.top_2, self.top_3, self.top_4 = geo.divide_distance(self.form_top, self.form_center, 4)

		# Following the double vesica pisces construction...
		self.form_bottom = geo.reflect(self.top_3, self.form_center)
		self.bridge = geo.reflect(self.top_4, self.form_center)

class HochLavta(Blend_Classic, LuteType1, Neck_ThruTop2):
	@override
	def _make_top_2_point(self):
		# top_2 is already made within base _make_spine_points
		pass

	@override
	def _get_soundhole_center(self):
		return geo.midpoint(self.top_3, geo.midpoint(self.top_4, self.form_center))

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
		experimental_neck_joint = geo.reflect(self.bridge, self._get_soundhole_center())
		self.helper_objects.extend([experimental_neck_joint])

		converter = self._get_unit_length() / self.unit
		Lute.print_meaurement("Neck at experimental point", converter * self.get_form_width_at_point(experimental_neck_joint))

class LavtaSmallThreeCourse(BlendWith_Unit, Blend_Classic, Soundhole_OneThirdOfSegment, SoundholeAt_NeckBridgeMidpoint, LuteType1, Neck_ThruTop2):
	@override
	def _make_top_2_point(self):
		# top_2 is already made within base _make_spine_points
		pass

	@override
	def _make_soundhole(self):
		self._make_neck_joint_fret()
		super()._make_soundhole()

class Brussels0164(Blend_Classic, SmallSoundhole_Brussels0164, LuteType1):
	@override
	def _make_spine_points(self):
		super()._make_spine_points()

		self.bridge = geo.translate_point_x(self.form_bottom, -self.unit)

		self._make_neck_joint_fret()

	@override
	def _get_soundhole_center(self):
		return geo.divide_distance(self.top_3, self.form_center, 3)[0]

	@override
	def _get_soundhole_radius(self):
		return self.top_3.distance(self._get_soundhole_center())

	@override
	def _make_neck_joint_fret(self):
		self.point_neck_joint = geo.reflect(self.bridge, self._get_soundhole_center())

	@override
	def _get_blender_radius(self):
		return float(self._get_soundhole_center().distance(self.form_center))


class LuteType10(TopArc_Type10, Lute):
	@override
	def _make_top_2_point(self):
		self.top_2 = geo.translate_point_x(self.form_top, self.unit)

	@override
	def _make_spine_points(self):
		self.bridge = geo.translate_point_x(self.form_center, self.unit)
		self.form_bottom = geo.reflect(self.form_center, self.bridge)

		self._make_neck_joint_fret()

class BaltaSaz(BlendWith_Unit, Blend_Classic, LuteType10, Neck_ThruTop2):
	@override
	def _get_unit_length(self):
		return 200/4


def test_all_lutes():
	lutes = []
	lutes.extend([ lute() for lute in TurkishOud.__subclasses__() ])
	lutes.extend([ IstanbulLavta(),IkwanAlSafaOud(), HannaNahatOud() ])
	lutes.extend([ lute() for lute in LuteType3.__subclasses__() ])
	lutes.extend([ lute() for lute in LuteType1.__subclasses__() ])
	lutes.extend([ TurkishOud2() ])

	print("\n\n\n\n\n")
	[lute.print_measurements() for lute in lutes]
	[lute.draw() for lute in lutes]

def test_single_lute():
	lute = TurkishOud2()
	lute.draw()
	lute.print_measurements()

def main():
	testing_all = 1
	if testing_all == 1:
		test_all_lutes()
	else:
		test_single_lute()

if __name__ == '__main__':
    main()


