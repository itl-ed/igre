import unittest, re

from igre.lang import RefExpParser

testsuite = [
            ("a square .", "<_a_q x4. _square_n_of(x4)>"),
            ("not a square above a circle .", "neg(<_a_q x4. _a_q x11.(_circle_n_of(x11),_square_n_of(x4) ^ _above_p(x4,x11))>)"),
            ("the two squares .", "<_the_2_q x4. _square_n_of(x4)>"),
            ("a circle to the left of exactly two green ellipses .","<_a_q x4. _exactly_2_q x16.(_green_a_2(x16) ^ _ellipsis_n_1(x16),_circle_n_of(x4) ^ _left_n_of(x4,x16))>"),
            ("a square behind the one square that is above a triangle .","<_a_q x4. _the_1_q x11.(_a_q x20.(_triangle_n_1(x20),_square_n_of(x11) ^ _above_p(x11,x20)),_square_n_of(x4) ^ _behind_p(x4,x11))>"),
            ("a square behind the one circle and a cross .","<_a_q x4. _the_1_q x11.(_circle_n_of(x11),_square_n_of(x4) ^ _behind_p(x4,x11))>  ^  <_a_q x4. _cross_n_1(x4)>"),
            ("exactly one circle behind a square .","<_exactly_1_q x4. _a_q x14.(_square_n_of(x14),_circle_n_of(x4) ^ _behind_p(x4,x14))>"),
            ("all but one circle behind the two squares .","<_all_but_1_q x4. _the_2_q x14.(_square_n_of(x14),_circle_n_of(x4) ^ _behind_p(x4,x14))>"),
            ("a yellow circle behind the two gray squares .", "<_a_q x4. _the_2_q x12.(_grey_a_1(x12) ^ _square_n_of(x12),_yellow_a_1(x4) ^ _circle_n_of(x4) ^ _behind_p(x4,x12))>"),
            ("exactly two triangles behind the two squares","<_exactly_2_q x4. _the_2_q x13.(_square_n_of(x13),_triangle_n_1(x4) ^ _behind_p(x4,x13))>"),
            ("a circle above the one square .","<_a_q x4. _the_1_q x11.(_square_n_of(x11),_circle_n_of(x4) ^ _above_p(x4,x11))>"),
            ("all but one star to the left of the two squares .","<_all_but_1_q x4. _the_2_q x18.(_square_n_of(x18),_star_n_1(x4) ^ _left_n_of(x4,x18))>"),
            ("at most one cyan ellipse above a square .","<_at_most_1_q x4. _a_q x14.(_square_n_of(x14),_cyan_a_1(x4) ^ _ellipse/nn_u_unknown(x4) ^ _above_p(x4,x14))>"),
            ("at least one pentagon behind a square .","<_at_least_1_q x4. _a_q x13.(_square_n_of(x13),_pentagon_n_1(x4) ^ _behind_p(x4,x13))>"),
            ("a circle behind at least one square .","<_a_q x4. _at_least_1_q x11.(_square_n_of(x11),_circle_n_of(x4) ^ _behind_p(x4,x11))>"),
            ("a circle behind at most one semicircle .", "<_a_q x4. _at_most_1_q x11.(_semicircle/nn_u_unknown(x11),_circle_n_of(x4) ^ _behind_p(x4,x11))>"),
            ("at least two blue squares behind at most one rectangle","<_at_least_2_q x4. _at_most_1_q x15.(_rectangle_n_1(x15),_blue_a_1(x4) ^ _square_n_of(x4) ^ _behind_p(x4,x15))>"),
        ]


class TestRefExpParser(unittest.TestCase):

    def test_testsuite(self):
        """Test parsing of different referential expressions"""
        nlu = RefExpParser()
        
        for surface, lf in testsuite:
            parsed_lf = nlu(surface) 
            self.assertEqual(lf, parsed_lf)


if __name__ == '__main__':
    unittest.main()

