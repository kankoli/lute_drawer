from abc import ABC, abstractmethod
from typing import final, override
from GeoDSL import GeoDSL, Point, GeoArc
import os.path

# Create an instance of GeoDSL
geo = GeoDSL()


class Lute(ABC):
	def __init__(self):
		self._base_construction()

		self._make_top_arc_circle()
		self._make_spine_points()
		self._make_bottom_arc_circle()
		self._make_soundhole()
		self._get_blender_radius()
		self._blend()
		self._make_arcs()

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

	def _make_bottom_arc_circle(self):
		self.bottom_arc_circle = geo.circle_by_center_and_point(self.form_top, self.form_bottom)

	@abstractmethod
	def _get_blender_radius(self):
		pass

	@abstractmethod
	def _make_soundhole(self):
		pass

	# Overwrite for intermediate circles to blend, for complex lower-bouts
	def _make_blender_side_circle(self):
		self.side_circle = self.top_arc_circle

	def _blend(self):
		self._make_blender_side_circle()
		self.blender_circle, self.blender_intersection_1, self.blender_intersection_2 = geo.blend_two_circles(self._get_blender_radius(), self.side_circle, self.bottom_arc_circle)

	@override
	def _make_arcs(self):
		top_arc = geo.arc_by_center_and_two_points(self.top_arc_center, self.blender_intersection_1, self.form_top)
		blender_arc = geo.arc_by_center_and_two_points(self.blender_circle.center, self.blender_intersection_2, self.blender_intersection_1)
		bottom_arc = geo.arc_by_center_and_two_points(self.form_top, self.form_bottom, self.blender_intersection_2)
		self.final_arcs = [top_arc, bottom_arc, blender_arc]
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
		    self.form_top, self.form_center, self.form_bottom, self.form_side, self.spine, self.centerline, \
		    self.point_neck_joint, \
		    self.soundhole_circle, \
		    self.bridge, \
		    *self.final_arcs, \
		    *self.final_reflected_arcs
	    ]

	@final
	def __dump_full_view(self):
		GeoDSL.draw_svg(self.full_view_objects, self.__get_file_name_prefix() + '_full_view.svg')

class Neck_ThruTop2(ABC):
	def _make_neck_joint_fret(self):
		helper_line = geo.line(self.top_2, self.top_arc_center)
		helper_point = geo.intersection(helper_line, self.top_arc_circle)[0] # top intersection
		helper_circle = geo.circle_by_center_and_point(self.top_2, helper_point)
		self.point_neck_joint = geo.intersection(helper_circle, self.spine)[0] # top intersection

class Neck_DoubleGolden(ABC):
	def _make_neck_joint_fret(self):
		# 7th fret location for ouds
		first_golden_point = geo.golden_ratio_divider(self.top_2, self.form_top)
		self.point_neck_joint = geo.golden_ratio_divider(self.form_top, first_golden_point)

class Soundhole_OneThirdOfSegment(ABC):
	def _make_soundhole(self):
		opposite_top_arc_center = geo.reflect(self.waist_2, self.form_side) # R2 opposite
		opposite_top_arc_circle = geo.circle_by_center_and_radius(opposite_top_arc_center, self.top_arc_radius)

		self.soundhole_perpendicular = geo.perpendicular_line(self.spine, self.soundhole_center)
		soundhole_perpendicular_left = geo.intersection(self.top_arc_circle, self.soundhole_perpendicular)[0] # Trial and error to find the right point of intersection
		soundhole_perpendicular_right = geo.intersection(opposite_top_arc_circle, self.soundhole_perpendicular)[0] # Trial and error to find the right point of intersection
		self.soundhole_segment_divisions = geo.divide_distance(soundhole_perpendicular_left, soundhole_perpendicular_right, 3) # Mark soundhole diameter

		soundhole_radius = self.soundhole_segment_divisions[0].distance(self.soundhole_segment_divisions[1]) / 2
		self.soundhole_circle = geo.circle_by_center_and_radius(self.soundhole_center, soundhole_radius)

class LuteType2(Lute):

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

	def _make_top_arc_circle(self):
		self.top_arc_center = geo.reflect(self.waist_4, self.waist_5)
		self.top_arc_radius = 5 * self.unit # Type 2
		self.top_arc_circle = geo.circle_by_center_and_radius(self.top_arc_center, self.top_arc_radius)

		return self.top_arc_circle	

class TurkishOud(Neck_DoubleGolden, LuteType2):
	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.soundhole_center = geo.translate_point_x(self.point_neck_joint, 2 * self.unit)
		self.bridge = geo.translate_point_x(self.soundhole_center, 2 * self.unit)
		self.form_bottom = geo.translate_point_x(self.bridge, self.unit)
	
	@override
	def _make_soundhole(self):
		self.soundhole_circle = geo.circle_by_center_and_radius(self.soundhole_center, self.half_unit)
		self._make_small_soundholes()

	def _make_small_soundholes(self):
		soundholes_axis_point = bridge_soundhole_midpoint = geo.midpoint(self.soundhole_center, self.bridge)
		soundholes_line = geo.perpendicular_line(self.spine, soundholes_axis_point)
		self.small_soundhole_locator = geo.circle_by_center_and_point(soundholes_axis_point, self.form_center)
		self.small_soundhole_centers = geo.intersection(self.small_soundhole_locator, soundholes_line)
		self.small_soundhole_circles = [geo.circle_by_center_and_radius(x, self.quarter_unit) for x in self.small_soundhole_centers]

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

class TurkishOudSingleMiddleArc(TurkishOud):
	@override
	def _get_blender_radius(self):
		return 3 * self.small_soundhole_centers[0].distance(self.small_soundhole_centers[1]) / 4

class TurkishOudDoubleMiddleArcs(TurkishOud):
	
	@override
	def _get_blender_radius(self):
		return self.unit

	@override
	def _make_blender_side_circle(self):
		self.second_arc_radius = 2 * self.unit
		self.second_arc_center = self.form_center
		self.second_arc_circle = geo.circle_by_center_and_radius(self.second_arc_center, self.second_arc_radius) # Readily blended with top_arc_circle

		self.side_circle = self.second_arc_circle

	@override
	def _make_arcs(self):
		top_arc = geo.arc_by_center_and_two_points(self.top_arc_center, self.form_side, self.form_top)
		middle_1_arc = geo.arc_by_center_and_two_points(self.second_arc_center, self.blender_intersection_1, self.form_side)
		blender_arc = geo.arc_by_center_and_two_points(self.blender_circle.center, self.blender_intersection_2, self.blender_intersection_1)
		bottom_arc = geo.arc_by_center_and_two_points(self.form_top, self.form_bottom, self.blender_intersection_2)
		self.final_arcs = [top_arc, bottom_arc, middle_1_arc, blender_arc]
		self.final_reflected_arcs = [geo.reflect(x, self.spine) for x in self.final_arcs]

class TurkishOudComplexLowerBout(TurkishOud):
	"""
	blender_radius, 			step_circle_radius, 		second_arc_center
	self.unit, 			self.half_unit, 			connector x spine intersection (rounder)
	3 * self.unit / 4, 	3 * self.unit / 4, 	connector x spine intersection
	"""
	
	@override
	def _get_blender_radius(self):
		return self.unit + self.half_unit

	@override
	def _make_blender_side_circle(self):
		self.step_circle = geo.circle_by_center_and_radius(self.form_side, 3 * self.unit / 4)
		step_intersections = geo.intersection(self.step_circle, self.top_arc_circle)
		self.top_arc_finish = geo.pick_west_point(*step_intersections)

		self.connector_1 = geo.line(self.top_arc_finish, self.top_arc_center)
		self.connector_intersections = geo.intersection(self.connector_1, self.spine)
		self.second_arc_center = self.connector_intersections[0] # single intersection
		"""
		print ("two points", self.second_arc_center, self.top_arc_finish)
		self.divisions = geo.divide_distance(self.second_arc_center, self.top_arc_finish, 4)
		print ("divisions", self.divisions)
		self.second_arc_center = self.divisions[0] # 3 /4 th
		"""
	
		second_arc_radius = self.second_arc_center.distance(self.top_arc_finish)
		self.second_arc_circle = geo.circle_by_center_and_radius(self.second_arc_center, second_arc_radius)

		self.side_circle = self.second_arc_circle

	@override
	def _make_arcs(self):
		top_arc = geo.arc_by_center_and_two_points(self.top_arc_center, self.top_arc_finish, self.form_top)
		second_arc = geo.arc_by_center_and_two_points(self.second_arc_center, self.blender_intersection_1, self.top_arc_finish)
		blender_arc = geo.arc_by_center_and_two_points(self.blender_circle.center, self.blender_intersection_2, self.blender_intersection_1)
		bottom_arc = geo.arc_by_center_and_two_points(self.form_top, self.form_bottom, self.blender_intersection_2)
		self.final_arcs = [top_arc, bottom_arc, second_arc, blender_arc]
		self.final_reflected_arcs = [geo.reflect(x, self.spine) for x in self.final_arcs]

	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.connector_1, self.step_circle])
		self.helper_objects.append(self.top_arc_center)
		self.helper_objects.append(self.top_arc_finish)
		self.helper_objects.append(self.connector_intersections[0])

class IstanbulLavta(Neck_ThruTop2, Soundhole_OneThirdOfSegment, LuteType2):

	@override
	def _make_spine_points(self):
		self._make_neck_joint_fret()

		self.top_6 = geo.translate_point_x(self.form_top, 5 * self.unit)
		self.form_bottom = geo.translate_point_x(self.form_top, 6 * self.unit)

		self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 4

		self.bridge = geo.translate_point_x(self.form_bottom, -self.vertical_unit) # negation is important
		self.soundhole_center = geo.midpoint(self.point_neck_joint, self.bridge)

	@override
	def _get_blender_radius(self):
		return 5*self.unit/4

	@override
	def _make_blender_side_circle(self):
		self.step_circle = geo.circle_by_center_and_radius(self.form_side, self.half_unit)
		step_intersections = geo.intersection(self.step_circle, self.top_arc_circle)
		self.top_arc_finish = geo.pick_south_point(*step_intersections)

		self.connector_1 = geo.line(self.top_arc_finish, self.top_arc_center)
		connector_intersections = geo.intersection(self.connector_1, self.spine)
		self.second_arc_center = connector_intersections[0] # single intersection

		second_arc_radius = self.second_arc_center.distance(self.top_arc_finish)
		self.second_arc_circle = geo.circle_by_center_and_radius(self.second_arc_center, second_arc_radius)

		self.side_circle = self.second_arc_circle

	@override
	def _make_arcs(self):
		top_arc = geo.arc_by_center_and_two_points(self.top_arc_center, self.top_arc_finish, self.form_top)
		second_arc = geo.arc_by_center_and_two_points(self.second_arc_center, self.blender_intersection_1, self.top_arc_finish)
		blender_arc = geo.arc_by_center_and_two_points(self.blender_circle.center, self.blender_intersection_2, self.blender_intersection_1)
		bottom_arc = geo.arc_by_center_and_two_points(self.form_top, self.form_bottom, self.blender_intersection_2)
		self.final_arcs = [top_arc, bottom_arc, second_arc, blender_arc]
		self.final_reflected_arcs = [geo.reflect(x, self.spine) for x in self.final_arcs]

	@override
	def _make_helper_objects(self):
		super()._make_helper_objects()
		self.helper_objects.extend([self.connector_1, self.step_circle])
		self.helper_objects.extend([self.soundhole_perpendicular, *self.soundhole_segment_divisions])

class LuteType1(Lute):

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
	def _make_top_arc_circle(self):
		self.top_arc_radius = 4 * self.unit # Type 1
		self.top_arc_circle = geo.circle_by_center_and_radius(self.top_arc_center, self.top_arc_radius)
		intersections = geo.intersection(self.top_arc_circle, self.spine)
		self.form_top = geo.pick_west_point(*intersections)

	@override
	def _make_spine_points(self):
		self.top_2, self.top_3, self.top_4 = geo.divide_distance(self.form_top, self.form_center, 4)
		self.bridge = geo.reflect(self.top_4, self.form_center)
		self.form_bottom = geo.reflect(self.form_center, self.bridge)

		self._make_neck_joint_fret()

class HochLavta(Neck_ThruTop2, LuteType1):
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

class LavtaSmallThreeCourse(Soundhole_OneThirdOfSegment, Neck_ThruTop2, LuteType1):
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

def main():
	
	oud = TurkishOudSingleMiddleArc()
	oud = TurkishOudDoubleMiddleArcs()
	oud = TurkishOudComplexLowerBout()
	lavta = IstanbulLavta()
	kucuk_lavta = LavtaSmallThreeCourse()
	hoch_lavta = HochLavta()
	

	lute = TurkishOudSingleMiddleArc()
	lute.draw()

if __name__ == '__main__':
    main()


