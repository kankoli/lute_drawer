from abc import ABC, abstractmethod
from typing import final, override
from GeoDSL import GeoDSL, Point, GeoArc
import os.path

# Create an instance of GeoDSL
geo = GeoDSL()

class TopArc(ABC):
	@abstractmethod
	def _make_top_arc_circle(self):
		pass

class TopArc_Type1(TopArc):
	@override
	def _make_top_arc_circle(self):
		self.top_arc_radius = 4 * self.unit # Type 1
		self.top_arc_circle = geo.circle_by_center_and_radius(self.top_arc_center, self.top_arc_radius)
		intersections = geo.intersection(self.top_arc_circle, self.spine)
		self.form_top = geo.pick_west_point(*intersections)

class TopArc_Type2(TopArc):
	@override
	def _make_top_arc_circle(self):
		self.top_arc_center = geo.reflect(self.waist_4, self.waist_5)
		self.top_arc_radius = 5 * self.unit # Type 2
		self.top_arc_circle = geo.circle_by_center_and_radius(self.top_arc_center, self.top_arc_radius)


class Neck(ABC):
	@abstractmethod
	def _make_neck_joint_fret(self):
		pass

class Neck_ThruTop2(Neck):
	@override
	def _make_neck_joint_fret(self):
		helper_line = geo.line(self.top_2, self.top_arc_center)
		helper_point = geo.intersection(helper_line, self.top_arc_circle)[0] # top intersection
		helper_circle = geo.circle_by_center_and_point(self.top_2, helper_point)
		self.point_neck_joint = geo.intersection(helper_circle, self.spine)[0] # top intersection

class Neck_DoubleGolden(Neck):
	@override
	def _make_neck_joint_fret(self):
		# 7th fret location for ouds
		first_golden_point = geo.golden_ratio_divider(self.top_2, self.form_top)
		self.point_neck_joint = geo.golden_ratio_divider(self.form_top, first_golden_point)

class Neck_Quartered(Neck):
	@override
	def _make_neck_joint_fret(self):
		# 7th fret location for ouds
		self.point_neck_joint = geo.translate_point_x(self.form_top, self.quarter_unit)


class Soundhole(ABC):
	@abstractmethod
	def _make_soundhole(self):
		pass

class Soundhole_OneThirdOfSegment(Soundhole):
	@override
	def _make_soundhole(self):
		opposite_top_arc_center = geo.reflect(self.top_arc_center, self.form_center) # R2 opposite
		opposite_top_arc_circle = geo.circle_by_center_and_radius(opposite_top_arc_center, self.top_arc_radius)

		self.soundhole_perpendicular = geo.perpendicular_line(self.spine, self.soundhole_center)
		soundhole_perpendicular_left = geo.intersection(self.top_arc_circle, self.soundhole_perpendicular)[0] # Trial and error to find the right point of intersection
		soundhole_perpendicular_right = geo.intersection(opposite_top_arc_circle, self.soundhole_perpendicular)[0] # Trial and error to find the right point of intersection
		self.soundhole_segment_divisions = geo.divide_distance(soundhole_perpendicular_left, soundhole_perpendicular_right, 3) # Mark soundhole diameter

		soundhole_radius = self.soundhole_segment_divisions[0].distance(self.soundhole_segment_divisions[1]) / 2
		self.soundhole_circle = geo.circle_by_center_and_radius(self.soundhole_center, soundhole_radius)
		super()._make_soundhole()

class Soundhole_HalfUnit(Soundhole):
	@override
	def _make_soundhole(self):
		self.soundhole_circle = geo.circle_by_center_and_radius(self.soundhole_center, self.half_unit)


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
		soundholes_axis_point = bridge_soundhole_midpoint = geo.midpoint(self.soundhole_center, self.form_center)
		soundholes_line = geo.perpendicular_line(self.spine, soundholes_axis_point)
		self.small_soundhole_locator = geo.circle_by_center_and_point(soundholes_axis_point, self.form_center)
		self.small_soundhole_centers = geo.intersection(self.small_soundhole_locator, soundholes_line)
		self.small_soundhole_circles = [geo.circle_by_center_and_radius(x, self.quarter_unit) for x in self.small_soundhole_centers]

class SmallSoundhole_Brussels0164(SmallSoundhole):
	@override
	def _make_small_soundholes(self):
		triangle_unit = self.soundhole_radius / 3
		# travel 4 units towards the bridge
		soundholes_axis_point = geo.translate_point_x(self.soundhole_center, 4 * triangle_unit)
		soundholes_line = geo.perpendicular_line(self.spine, soundholes_axis_point)
		self.small_soundhole_locator = geo.circle_by_center_and_radius(soundholes_axis_point, 3 * triangle_unit)
		self.small_soundhole_centers = geo.intersection(self.small_soundhole_locator, soundholes_line)
		self.small_soundhole_circles = [geo.circle_by_center_and_radius(x, self.soundhole_radius / 4) for x in self.small_soundhole_centers]


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

		self.side_circle = geo.circle_by_center_and_radius(second_arc_center, second_arc_radius) # Readily blended with top_arc_circle
		self.top_arc_finish = self.form_side # Shortcut to intersection of the top circle and side circle

class Blend_StepCircle(Blend_WithCircle):
	@abstractmethod
	def _get_step_circle_radius(self):
		pass

	@abstractmethod
	def _get_step_circle_function(self):
		pass

	def _get_step_circle_intersection(self, intersections):
		func = self._get_step_circle_function()
		return func(*intersections)

	@override
	def _make_blender_side_circle(self):
		self.step_circle = geo.circle_by_center_and_radius(self.form_side, self._get_step_circle_radius())

		step_intersections = geo.intersection(self.step_circle, self.top_arc_circle)
		self.top_arc_finish = self._get_step_circle_intersection(step_intersections)

		self.connector_1 = geo.line(self.top_arc_finish, self.top_arc_center)
		self.connector_intersections = geo.intersection(self.connector_1, self.spine)
		second_arc_center = self.connector_intersections[0] # single intersection
		"""
		print ("two points", self.second_arc_center, self.top_arc_finish)
		self.divisions = geo.divide_distance(self.second_arc_center, self.top_arc_finish, 4)
		print ("divisions", self.divisions)
		self.second_arc_center = self.divisions[0] # 3 /4 th
		"""

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


class Lute(ABC):
	def __init__(self):
		self._base_construction()

		self._make_top_arc_circle()
		self._make_spine_points()
		self.__make_bottom_arc_circle()
		self._make_soundhole()
		self._get_blender_radius()
		self.__blend()
		self._make_arcs()
		self.__generate_arcs()

	@abstractmethod
	def _base_construction(self):
		pass

	@abstractmethod
	def _make_top_arc_circle(self):
		pass

	@abstractmethod
	def _make_spine_points(self):
		pass

	@abstractmethod
	def _make_neck_joint_fret(self):
		pass # NOT called directly within the flow in __init__. Expected to be called somewhere inside _make_spine_points implementation

	def __make_bottom_arc_circle(self):
		self.bottom_arc_circle = geo.circle_by_center_and_point(self.form_top, self.form_bottom)

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

	def _make_template_points(self):
		self.template_bottom_halving_point = geo.midpoint(self.bridge, self.form_bottom)
		self.template_top = geo.translate_point_x(self.form_top, -self.quarter_unit)
		self.template_bottom = geo.translate_point_x(self.form_bottom, self.quarter_unit)
		self.template_spine = geo.line(self.template_top, self.template_bottom)

		self.template_points = [
			self.template_top, \
			self.form_top, \
			self.point_neck_joint, \
			self.top_2, self.top_4, self.form_center, \
			self.bridge, \
			self.soundhole_center, \
			self.form_bottom, self.template_bottom_halving_point, \
			self.template_bottom
		]

	def _make_template_lines(self):
		self._make_template_points()
		self.template_lines = [geo.perpendicular_segment(self.spine, geo.point(p.x, p.y - 3 * self.unit)) for p in self.template_points]

	def _make_template_objects(self):
		self._make_template_lines()

		self.template_objects = [
			self.A, self.B, \
			self.soundhole_circle, \
			self.template_spine, \
			*self.final_arcs
		]

		self.template_objects.extend(self.template_lines)

	@final
	def __get_file_name_prefix(self):
		return os.path.basename(type(self).__name__)

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
			self.soundhole_center, self.soundhole_circle, \
		    self.bridge
	    ]

	@final
	def __dump_helper(self):
		GeoDSL.draw_svg(self.helper_objects, self.__get_file_name_prefix() + '_helpers.svg')

	def _make_full_view_objects(self):
		self.full_view_objects = [
		    self.A, self.B, \
		    self.form_top, self.form_center, self.form_bottom, self.form_side, \
		    self.point_neck_joint, \
		    self.soundhole_circle, self.soundhole_center, \
		    self.bridge, \
		    *self.final_arcs, \
		    *self.final_reflected_arcs
	    ]

	@final
	def __dump_full_view(self):
		GeoDSL.draw_svg(self.full_view_objects, self.__get_file_name_prefix() + '_full_view.svg')


class LuteType2(TopArc_Type2, Lute):

	"""
																 waist_5

																 =======

																 waist_4

																 =======

	form_top ======= top_2 ======= top_3  ======= top_4 ======= form_center

																 =======

																 waist_2

																 =======

																 form_side

	"""

	def _base_construction(self):
		self.double_unit = 400 # Neck length, for Turkish Ouds
		self.unit = self.double_unit / 2
		self.half_unit = self.unit / 2
		self.quarter_unit = self.half_unit / 2

		self.A = geo.point(75, 600)
		self.B = geo.point(self.A.x + self.double_unit, self.A.y) # A - B = neck length
		self.form_top = geo.point(75, 500)

		self.top_2 = geo.translate_point_x(self.form_top, self.unit)
		self.top_3 = geo.translate_point_x(self.form_top, 2 * self.unit)
		self.top_4 = geo.translate_point_x(self.form_top, 3 * self.unit)
		self.form_center = geo.translate_point_x(self.form_top, 4 * self.unit)

		self.spine = geo.line(self.form_top, self.form_center)
		self.centerline = geo.perpendicular_line(self.spine, self.form_center)

		self.form_side = geo.translate_point_y(self.form_center, -2 * self.unit)
		self.waist_2 = geo.translate_point_y(self.form_center, - self.unit)
		# self.form_center
		self.waist_4 = geo.translate_point_y(self.form_center, self.unit)
		self.waist_5 = geo.translate_point_y(self.form_center, 2 * self.unit)

class TurkishOud(Neck_DoubleGolden, SmallSoundhole_Turkish, LuteType2):
	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.soundhole_center = geo.translate_point_x(self.point_neck_joint, 2 * self.unit)
		self.bridge = geo.translate_point_x(self.soundhole_center, 2 * self.unit)
		self.form_bottom = geo.translate_point_x(self.bridge, self.unit)

class TurkishOudSingleMiddleArc(Blend_Classic, TurkishOud, Soundhole_HalfUnit):
	@override
	def _get_blender_radius(self):
		return 3 * self.small_soundhole_centers[0].distance(self.small_soundhole_centers[1]) / 4

class TurkishOudDoubleMiddleArcs(Blend_SideCircle, TurkishOud, Soundhole_HalfUnit):
	@override
	def _get_blender_radius(self):
		return self.unit

	@override
	def _get_side_circle_radius(self):
		return 2*self.unit

class TurkishOudComplexLowerBout(Blend_StepCircle, TurkishOud, Soundhole_HalfUnit):
	@override
	def _get_step_circle_radius(self):
		return 3 * self.unit / 4

	@override
	def _get_step_circle_function(self):
		return geo.pick_west_point

	@override
	def _get_blender_radius(self):
		return self.unit + self.half_unit



	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.connector_1, self.step_circle])
		self.helper_objects.append(self.top_arc_center)
		self.helper_objects.append(self.top_arc_finish)
		self.helper_objects.append(self.connector_intersections[0])

class TurkishOudSoundholeThird(Blend_Classic, TurkishOud, Soundhole_OneThirdOfSegment):
	@override
	def _get_blender_radius(self):
		return 3 * self.small_soundhole_centers[0].distance(self.small_soundhole_centers[1]) / 4

class IstanbulLavta(Neck_ThruTop2, Blend_StepCircle, Soundhole_OneThirdOfSegment, LuteType2):

	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.top_6 = geo.translate_point_x(self.form_top, 5 * self.unit)
		self.form_bottom = geo.translate_point_x(self.form_top, 6 * self.unit)

		self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 4

		self.bridge = geo.translate_point_x(self.form_bottom, -self.vertical_unit) # negation is important
		self.soundhole_center = geo.midpoint(self.point_neck_joint, self.bridge)

	@override
	def _get_step_circle_radius(self):
		return self.unit / 2

	@override
	def _get_step_circle_function(self):
		return geo.pick_east_point

	@override
	def _get_blender_radius(self):
		return 5*self.unit/4

	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.connector_1, self.step_circle])
		self.helper_objects.extend([self.soundhole_perpendicular, *self.soundhole_segment_divisions])

class HannaNahatOud(Neck_DoubleGolden, Blend_SideCircle, LuteType2):
	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.bridge = geo.translate_point_x(self.form_center, 3 * self.unit / 4)
		self.form_bottom = geo.translate_point_x(self.bridge, self.unit)
		self.soundhole_center = geo.midpoint(self.bridge, self.point_neck_joint)

	@override
	def _make_soundhole(self):
		bridge_reflected = geo.reflect(self.bridge, self.form_center)
		self.soundhole_radius = bridge_reflected.distance(self.soundhole_center)
		self.soundhole_circle = geo.circle_by_center_and_radius(self.soundhole_center, self.soundhole_radius)

	@override
	def _get_side_circle_radius(self):
		return 2*self.unit

	@override
	def _get_blender_radius(self):
		return (2 - 3 /4 ) * self.unit # difference of the bridge-form_center difference from the form half (2 units)


class LuteType1(TopArc_Type1, Lute):

	@override
	def _base_construction(self):
		self.unit = 100 #  1/4th of the belly
		self.half_unit = self.unit / 2
		self.quarter_unit = self.half_unit / 2
		self.eighth_unit = self.quarter_unit / 2

		self.A = geo.point(75, 600)
		self.B = geo.point(self.A.x+self.unit, self.A.y)

		self.top_arc_center = geo.point(400, 700)
		self.waist_4 = geo.translate_point_y(self.top_arc_center, -self.unit)
		self.form_center = geo.translate_point_y(self.top_arc_center, -2*self.unit)
		self.waist_2 = geo.translate_point_y(self.top_arc_center, -3*self.unit)
		self.form_side = geo.translate_point_y(self.top_arc_center, -4*self.unit)

		self.centerline = geo.line(self.top_arc_center, self.form_side)
		self.spine = geo.perpendicular_line(self.centerline, self.form_center)

	@override
	def _make_spine_points(self):
		self.top_2, self.top_3, self.top_4 = geo.divide_distance(self.form_top, self.form_center, 4)
		self.bridge = geo.reflect(self.top_4, self.form_center)
		self.form_bottom = geo.reflect(self.form_center, self.bridge)

		self._make_neck_joint_fret()

class HochLavta(Neck_ThruTop2, Blend_Classic, LuteType1):
	@override
	def _make_soundhole(self):
		soundhole_helper_point = geo.midpoint(self.top_4, self.form_center)
		self.soundhole_center = geo.midpoint(self.top_3, soundhole_helper_point)
		soundhole_radius = float(self.soundhole_center.distance(soundhole_helper_point))
		self.soundhole_circle = geo.circle_by_center_and_radius(self.soundhole_center, soundhole_radius)

		self.second_neck_joint = geo.reflect(self.bridge, self.soundhole_center)

	@override
	def _get_blender_radius(self):
		return float(self.soundhole_center.distance(self.waist_2))

	@override
	def _make_full_view_objects(self):
		super()._make_full_view_objects()
		self.full_view_objects.append(self.second_neck_joint)

class LavtaSmallThreeCourse(Neck_ThruTop2, Blend_Classic, Soundhole_OneThirdOfSegment, LuteType1):
	@override
	def _make_soundhole(self):
		self.soundhole_center = geo.midpoint(self.point_neck_joint, self.bridge)
		super()._make_soundhole()

	@override
	def _get_blender_radius(self):
		return float(self.soundhole_center.distance(self.waist_2))

	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.soundhole_perpendicular, *self.soundhole_segment_divisions])

class Brussels0164(Blend_Classic, SmallSoundhole_Brussels0164, LuteType1):
	@override
	def _make_spine_points(self):
		smaller_vesica_piscis_circle = geo.circle_by_center_and_radius(self.waist_4, 2*self.unit)
		vesica_piscis_intersections = geo.intersection(self.spine, smaller_vesica_piscis_circle)
		self.form_bottom = geo.pick_east_point(*vesica_piscis_intersections)
		self.bridge = geo.translate_point_x(self.form_bottom, -self.unit)
		self.soundhole_top = geo.pick_west_point(*vesica_piscis_intersections)

		# bogus to keep _make_template_points happy
		self.top_2, self.top_4 = self.soundhole_top, geo.midpoint(self.form_bottom, self.soundhole_top)

	@override
	def _make_soundhole(self):
		self.soundhole_center = geo.divide_distance(self.soundhole_top, self.form_center, 3)[0]
		self.soundhole_radius = self.soundhole_top.distance(self.soundhole_center)
		self.soundhole_circle = geo.circle_by_center_and_radius(self.soundhole_center, self.soundhole_radius)

		self._make_neck_joint_fret()
		self._make_small_soundholes()

	@override
	def _make_neck_joint_fret(self):
		self.point_neck_joint = geo.reflect(self.bridge, self.soundhole_center)

	@override
	def _get_blender_radius(self):
		return float(self.soundhole_center.distance(self.form_center))

def test_all_lutes():
	lutes = [ \
		TurkishOudSingleMiddleArc(), \
		TurkishOudDoubleMiddleArcs(), \
		TurkishOudComplexLowerBout(), \
		TurkishOudSoundholeThird(), \
		IstanbulLavta(), \
		HochLavta(), \
		LavtaSmallThreeCourse(), \
		HannaNahatOud(), \
		Brussels0164()
		]

	[lute.draw() for lute in lutes]

def main():
	test_all_lutes()

	# lute = Brussels0164()
	# lute.draw()

if __name__ == '__main__':
    main()


