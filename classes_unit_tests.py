import unittest
import math

# -----------------------------------------------------------------------------------
# --------------------- CLASSES UNIT TESTS ------------------------------------------
# -----------------------------------------------------------------------------------

class TestFunction41(BaseTestCase):
    triangle = imported_functions[40]
    
    def setUp(self):
        # Creating a triangle instance for testing
        self.tri = triangle(10, 5, 90, 'black')

    def test_get_edge_a(self):
        self.assertEqual(self.tri.get('a'), 10)

    def test_get_edge_b(self):
        self.assertEqual(self.tri.get('b'), 5)

    def test_get_angle_ab(self):
        self.assertEqual(self.tri.get('ab'), 90)

    def test_get_color(self):
        self.assertEqual(self.tri.get('color'), 'black')

    def test_get_edge_c(self):
        self.assertAlmostEqual(self.tri.get('c'), 11.180339887498949)

    def test_get_angle_ac(self):
        self.assertAlmostEqual(self.tri.get('ac'), 26.565051177077994)

    def test_get_angle_bc(self):
        self.assertAlmostEqual(self.tri.get('bc'), 63.43494882292201)

    def test_get_angle_ab(self):
        self.assertEqual(self.tri.get('ab'), 90)

    def test_invalid_attribute(self):
        with self.assertRaises(KeyError):
            self.tri.get('d')

    def test_attribute_sorting(self):
        # Testing if 'ba' gets sorted to 'ab'
        self.assertEqual(self.tri.get('ba'), self.tri.get('ab'))
        
class TestFunction42(BaseTestCase):
    worker = imported_functions[41]
    
    def setUp(self):
        # Creating a worker instance for testing
        self.worker = Worker('12345', 'Jon', 'Cohen', 'Salesman')
        self.worker_with_second_name = Worker('11111', 'David', 'Cohen', 'Math teacher', 'Julius')

    def test_get_full_name_basic(self):
        self.assertEqual(self.worker.getFullName(), "Jon Cohen")
    
    def test_get_full_name_with_second_name(self):
        self.assertEqual(self.worker_with_second_name.getFullName(), "David Julius Cohen")

    def test_get_job(self):
        self.assertEqual(self.worker.getJob(), "Salesman")

    def test_get_salary(self):
        self.assertEqual(self.worker.getSalary(), 5000)

    def test_update_job_and_salary(self):
        self.worker.update(job='Engineer', salary=7000)
        self.assertEqual(self.worker.getJob(), "Engineer")
        self.assertEqual(self.worker.getSalary(), 7000)

    def test_update_salary_only(self):
        self.worker.update(salary=9000)
        self.assertEqual(self.worker.getSalary(), 9000)
        self.assertEqual(self.worker.getJob(), "Salesman")  # Job should remain unchanged
        
    def test_update_job_only(self):
        self.worker.update(job='Manager')
        self.assertEqual(self.worker.getJob(), "Manager")
        self.assertEqual(self.worker.getSalary(), 5000)
        
class TestFunction43(BaseTestCase):
    binaric_arithmatic = imported_functions[42]
    
    def setUp(self):
        # Creating an instance for testing
        self.zero = binaric_arithmatic("0")
        self.one = binaric_arithmatic("1")
        self.seven = binaric_arithmatic("111")
        
    def test_inc_positive(self):
        self.assertEqual(self.seven.inc(), "1000")
        self.assertEqual(self.one.inc(), "10")
    
    def test_inc_zero(self):    
        self.assertEqual(self.zero.inc(), "1")

    def test_dec(self):
        self.assertEqual(seven.dec(), "110")
        self.assertEqual(one.dec(), "0")
    
    def test_get(self):
        self.assertEqual(self.seven.get(), "111")
        self.assertEqual(self.one.get(), "1")
        self.assertEqual(self.zero.get(), "0")
    
class TestFunction44(BaseTestCase):
    Point_2D = imported_functions[43]
    
    def setUp(self):
        # Creating Point_2D instances for testing
        self.a = Point_2D(1, 1)
        self.b = Point_2D(0, 1)
        self.c = Point_2D(-1, 1)
        self.d = Point_2D(1, 1)

    def test_repr(self):
        self.assertEqual(repr(self.a), "Point(1, 1)")

    def test_eq(self):
        self.assertTrue(self.a == self.d)
        self.assertFalse(self.a == self.b)

    def test_add(self):
        self.assertEqual(self.a + self.b, Point_2D(1, 2))

    def test_sub(self):
        self.assertEqual(self.a - self.b, Point_2D(1, 0))

    def test_distance(self):
        self.assertEqual(self.a.distance(self.c), 2)
        self.assertAlmostEqual(self.a.distance(self.b), math.sqrt(1))

    def test_angle_wrt_origin(self):
        self.assertAlmostEqual(self.b.angle_wrt_origin(self.c), math.pi / 4)
        self.assertAlmostEqual(self.c.angle_wrt_origin(self.b), 2 * math.pi - math.pi / 4)
    
    