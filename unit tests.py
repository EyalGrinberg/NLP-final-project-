import unittest

def string_to_function(func_str):
    """
    Converts a string representation of a function into a runnable function.
    
    Parameters:
    func_str (str): A string representation of the function to be converted.
    
    Returns:
    function: A runnable function object.
    """
    # Define a dictionary to act as the local namespace for exec
    local_namespace = {}
    
    # Execute the function string in the local namespace
    exec(func_str, globals(), local_namespace)
    
    # Extract the function name
    func_name = func_str.split('(')[0].split()[-1]
    
    # Return the function object from the local namespace
    return local_namespace[func_name]


class Test1(unittest.TestCase):

    def test_empty_list(self):
        """Test sum_even with an empty list."""
        self.assertEqual(sum_even([]), 0)

    def test_single_element(self):
        """Test sum_even with a single element list."""
        self.assertEqual(sum_even([2]), 2)
        self.assertEqual(sum_even([3]), 3)

    def test_single_list(self):
        """Test sum_even with a single list."""
        self.assertEqual(sum_even([1, 2, 3, 4, 5]), 9)

    def test_nested_list(self):
        """Test sum_even with a nested list."""
        self.assertEqual(sum_even([1, [2, 4], 3]), 6)
        self.assertEqual(sum_even([1, [3, 5], [2, 4]]), 6)

    def test_multiple_nested_levels(self):
        """Test sum_even with multiple levels of nesting."""
        self.assertEqual(sum_even([1, [2, 3, [4, 5]]]), 7)

class Test2(unittest.TestCase):
    def test_case_1(self):
        pass

if __name__ == "__main__":
    unittest.main()
