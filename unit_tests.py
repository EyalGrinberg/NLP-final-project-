import unittest

# List of string representations of functions
functions_list = [
"""
def sum_even(lst):
    total = 0
    for index, item in enumerate(lst):
        if isinstance(item, list):
            total += sum_even(item)
        elif index % 2 == 0:
            total += item
    return total
""",
"""
def find_num_changes(n, lst):
    if n == 0:
        return 1
    if len(lst) == 0:
        return 0
    if n < 0:
        return 0
    return find_num_changes(n - lst[0], lst) + find_num_changes(n, lst[1:])
""",
"""def sum_nested(lst):
    if len(lst) == 0:
        return 0.0
    if type(lst[0]) == str:
        return float(abs(sum_nested(lst[1:])))
    if type(lst[0]) == list:
        return float(abs(sum_nested(lst[0]))) + float(abs(sum_nested(lst[1:])))
    return float(abs(lst[0])) + float(abs(sum_nested(lst[1:])))
""",
"""def str_decomp(target, word_bank):
    if target == '':
        return 1
    total_count = 0
    for word in word_bank:
        if target.startswith(word):
            new_target = target[len(word):]
            total_count += str_decomp(new_target, word_bank)
    return total_count""",
"""def n_choose_k(n, k):
    if k < 0 or k > n:
        return 0
    if k == 1:
        return n
    if k == 0:
        return 1
    return n_choose_k(n-1, k-1) + n_choose_k(n-1, k)""",
"""def dfs_level_order(tree, index=0):
    if index >= len(tree) or tree[index] is None:
        return ""
    visited_str = str(tree[index])
    left_subtree = dfs_level_order(tree, 2 * index + 1)
    right_subtree = dfs_level_order(tree, 2 * index + 2)
    result = visited_str
    if left_subtree:
        result += "," + left_subtree
    if right_subtree:
        result += "," + right_subtree

    return result""",
"""def half_sum_subset(lst):
    total = sum(lst)
    if total % 2 != 0:
        return None
    target = total // 2
    def find_subset(idx, curr):
        if curr == target:
            return []
        if idx >= len(lst) or curr > target:
            return None
        w_curr = find_subset(idx + 1, curr + lst[idx])
        if w_curr is not None:
            return [lst[idx]] + w_curr
        wo_current = find_subset(idx + 1, curr)
        if wo_current is not None:
            return wo_current
        return None
    return find_subset(0, 0)""",
"""def str_dist(x, y): 
    if len(x) == 0 or len(y) == 0: 
        return max(len(x), len(y)) 
    if x[-1] == y[-1]: 
        return str_dist(x[: -1], y[: -1]) 
    return min(str_dist(x, y[: -1]), str_dist(x[: -1], y), str_dist(x[: -1], y[: -1])) + 1
    """,
"""def is_dag(graph):
    visited = set()
    exploring = set()

    def dfs(node):
        visited.add(node)
        exploring.add(node)
        for neighbor in graph[node]:
            if neighbor == node:
                continue
            if neighbor in exploring:
                return False
            if neighbor not in visited:
                return dfs(neighbor)
        exploring.remove(node)
        return True

    for node in range(len(graph)):
        if node not in visited and not dfs(node):
            return False
    return True

""",
"""def foo(num, x = 0):
    if num < 10:
        return num, x
    return foo(num * (2/3), x + 1)
    """,
"""def diff_sparse_matrices(lst):
    res_dict = lst[0]
    for dict in lst[1:]:
        for entry in dict:
            if entry in res_dict:
                res_dict[entry] -= dict[entry]
            else:
                res_dict[entry] = -dict[entry]
    return res_dict
""",
"""def longest_subsequence_length(lst):
    n = len(lst)
    if n == 0: return 0
    lis_lengths = [1] * n
    lds_lengths = [1] * n
    for i in range(1, n):
        for j in range(i):
            if lst[i] > lst[j]:
                lis_lengths[i] = max(lis_lengths[i], lis_lengths[j] + 1)
            if lst[i] < lst[j]:
                lds_lengths[i] = max(lds_lengths[i], lds_lengths[j] + 1)
    return max(max(lis_lengths), max(lds_lengths))
"""
    ]

# Function to convert string to a runnable function
def string_to_function(func_str):
    """
    Converts a string representation of a function into a runnable function.
    
    Parameters:
    func_str (str): A string representation of the function to be converted.
    
    Returns:
    function: A runnable function object.
    """
    # Extract the function name
    func_name = func_str.split('(')[0].split()[-1]
    
    # Define a placeholder in the local namespace
    local_namespace = {}
    exec(f"def {func_name}(): pass", globals(), local_namespace)
    
    # Compile and execute the function string in the local namespace
    code = compile(func_str, '<string>', 'exec')
    exec(code, globals(), local_namespace)

    globals()[func_name] = local_namespace[func_name]
    
    # Return the function object from the local namespace
    return local_namespace[func_name]

# Dictionary to map function indices to their respective test cases
test_cases = {i: f'TestFunction{i+1}' for i in range(len(functions_list))}

# Transform string representations to runnable functions
imported_functions = [string_to_function(func_str) for func_str in functions_list]

# Base class for test cases to inherit from
class BaseTestCase(unittest.TestCase):
    def setUp(self):
        pass  # Optional setup method to run before each test

    def tearDown(self):
        pass  # Optional teardown method to run after each test

class TestFunction1(BaseTestCase):
    sum_even = imported_functions[0]
    
    def test_empty_list(self):
        self.assertEqual(sum_even([]), 0)
    
    def test_single_item_list(self):
        self.assertEqual(sum_even([1]), 1)

    def test_flat_list(self):
        lst = [0, 1, 2, 3, 4, 5]
        self.assertEqual(sum_even(lst), 6)
    
    def test_nested_list(self):
        lst = [1, [2, 3, [4, 5]]]
        self.assertEqual(sum_even(lst), 7)
    
    def test_multiple_exclusive_nesting(self):
        lst = [1, [2, 3, [4, 5]], 6, [7, 8]]
        self.assertEqual(sum_even(lst), 20)

    def test_two_level_nesting(self):
        lst = [[[1, 2, 3], [4, 5, 6]]]
        self.assertEqual(sum_even(lst), 14)



class TestFunction2(BaseTestCase):
    find_num_changes = imported_functions[1]

    def test_empty_list(self):
        self.assertEqual(find_num_changes(5, []), 0)
    
    def test_zero_candidate(self):
        lst = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(find_num_changes(0, lst), 1)

    def test_negative_candidate(self):
        lst = [1,2,3,4,5]
        self.assertEqual(find_num_changes(-1, lst) , 0)

    def test_candidate_not_in_list(self):
        lst = [2, 5, 7]
        self.assertEqual(find_num_changes(1, lst), 0)
    
    def test_candidate_in_list(self):
        lst = [1, 2, 5, 6]
        self.assertEqual(find_num_changes(4, lst), 3)

class TestFunction3(BaseTestCase):
    sum_nested = imported_functions[2]

    def test_empty_list(self):
        self.assertEqual(sum_nested([]), 0.0)

    def test_flat_list_positives(self):
        lst = [1, 2, 3]
        self.assertEqual(sum_nested(lst), 6.0)
    
    def test_flat_list_nums(self):
        lst = [1, -2, 3]
        self.assertEqual(sum_nested(lst), 6.0)
    
    def test_flat_list_mixed(self):
        lst = [1, 2, "ab", -4]
        self.assertEqual(sum_nested(lst), 7.0)
    
    def test_flat_list_strings(self):
        lst = ['a', "abd", 'zzz', "hello world"]
        self.assertEqual(sum_nested(lst), 0.0)
    
    def test_nested_list_positives(self):
        lst = [0.5, 2.5, [3, 4], [5, [6, 7], 8], 9.4]
        self.assertEqual(sum_nested(lst), 45.4)

    def test_nested_list_nums(self):
        lst = [0.5, -2.5, [3, -4], [5, [-6, 7], 8], 9.4]
        self.assertEqual(sum_nested(lst), 45.4)

    def test_nested_list_mixed(self):
        lst = ["aa", [-3, -4.6], 'abc', [5, 'abc', [-4, 0.5]]]
        self.assertEqual(sum_nested(lst), 17.1)
    
    def test_nested_list_strings(self):
        lst = ["aa", "b", ["hello"]]
        self.assertEqual(sum_nested(lst), 0.0)

class TestFunction4(BaseTestCase):
    str_decomp = imported_functions[3]

    def test_empty_target(self):
        word_bank = ["aa", "bb", "cc"]
        self.assertEqual(str_decomp('', word_bank), 1)
    
    def test_empty_word_bank(self):
        target = 'abc'
        self.assertEqual(str_decomp(target, []), 0)

    def test_target_not_in_bank(self):
        target = "abc"
        word_bank = ["z", "x"]
        self.assertEqual(str_decomp(target, word_bank), 0)
    
    def test_target_in_bank(self):
        target = "abc"
        word_bank = ["z", "x", "y", "abc"]
        self.assertEqual(str_decomp(target, word_bank), 1)
    
    def test_overlaping_words_only(self):
        target = "abcdef"
        word_bank = ["ab", "cd", "def", "abcd"]
        self.assertEqual(str_decomp(target, word_bank), 0)

    def test_repeatidly_using_word(self):
        target = "purple"
        word_bank = ["p", "ur", 'le']
        self.assertEqual(str_decomp(target, word_bank), 1)
    
    def test_multiple_options(self):
        target = "purple"
        word_bank = ["purp", "e", "purpl", 'le']
        self.assertEqual(str_decomp(target, word_bank), 2)

    def test_multiple_options_with_reps(self):
        target = "aabbcc"
        word_bank = ["a", "ab", "b", "bc", "c", "abc", "abcd"]
        self.assertEqual(str_decomp(target, word_bank), 4)

class TestFunction5(BaseTestCase):
    n_choose_k = imported_functions[3]

    def test_n_negative(self):
        n, k = -1, -1 
        self.assertEqual(n_choose_k(n, k), 0)

    def test_k_larger(self):
        n, k = 1, 2
        self.assertEqual(n_choose_k(n, k), 0)
    
    def test_k_negative(self):
        n, k = 4, -1
        self.assertEqual(n_choose_k(n, k), 0)
    
    def test_n_equals_k(self):
        n, k = 3, 3
        self.assertEqual(n_choose_k(n, k), 1)
    
    def test_k_is_one(self):
        n, k = 8, 1
        self.assertEqual(n_choose_k(n, k), 8)
    
    def test_k_is_zero(self):
        n, k = 5, 0
        self.assertEqual(n_choose_k(n, k), 1)
    
    def test_k_one_smaller(self):
        n, k = 9, 8
        self.assertEqual(n_choose_k(n, k), 9)
    
    def test_regular_case(self):
        n, k = 10, 3
        self.assertEqual(n_choose_k(n, k), 120)

    
class TestFunction6(BaseTestCase):
    dfs_level_order = imported_functions[5]

    def test_empty_tree(self):
        self.assertEqual(dfs_level_order([]), "")

    def test_single_node_tree(self):
        self.assertEqual(dfs_level_order([1]), "1")

    def test_full_tree(self):
        tree = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(dfs_level_order(tree), "1,2,4,5,3,6,7")

    def test_incomplete_tree(self):
        tree = [1, 2, 3, None, 5, None, 7]
        self.assertEqual(dfs_level_order(tree), "1,2,5,3,7")

    def test_complex_tree(self):
        tree = [1, 2, 3, 4, None, None, 5, 6, None, None, None, None, None, 7]
        self.assertEqual(dfs_level_order(tree), "1,2,4,6,3,5,7")

    def test_all_none_tree(self):
        tree = [None, None, None]
        self.assertEqual(dfs_level_order(tree), "")

    def test_large_tree(self):
        tree = list(range(1, 16))
        self.assertEqual(dfs_level_order(tree), "1,2,4,8,9,5,10,11,3,6,12,13,7,14,15")

class TestFunction7(BaseTestCase):
    half_sum_subset = imported_functions[6]
        
    def test_empty_list(self):
        self.assertEqual(half_sum_subset([]), [])

    def test_single_item_list(self):
        self.assertIsNone(half_sum_subset([1]))
    
    def test_two_items_no_half_sum(self):
        self.assertIsNone(half_sum_subset([1, 2]))
    
    def test_two_items_with_half_sum(self):
        self.assertEqual(half_sum_subset([2, 2]), [2])
    
    def test_multiple_items_no_half_sum(self):
        self.assertIn(sorted(half_sum_subset([1, 2, 3])), [[1, 2], [3]])
    
    def test_multiple_items_with_half_sum(self):
        self.assertIn(sorted(half_sum_subset([1, 2, 3, 4])), [[1, 4], [2, 3]])
    
    def test_list_with_zero(self):
        self.assertIn(sorted(half_sum_subset([0, 1, 2, 3])), [[1, 2], [3], [0, 1, 2], [0, 3]])
    
    def test_with_negatives(self):
        self.assertIn(sorted(half_sum_subset([-1, -2, 3, 6])), [[-2, -1, 6], [3]])
        
    def test_with_even_total_sum_but_no_half_sum(self):
        self.assertIsNone(half_sum_subset([1, 3, 5, 13]))
    
    def test_large_list(self):
        self.assertIn(sorted(half_sum_subset([3, 1, 4, 2, 2])), [[1, 2, 3], [2, 4]])


class TestFunction8(BaseTestCase):
    str_dist = imported_functions[7]

    def test_empty_strings(self):
        self.assertEqual(str_dist("", ""), 0)

    def test_empty_first_string(self):
        self.assertEqual(str_dist("", "abc"), 3)

    def test_empty_second_string(self):
        self.assertEqual(str_dist("abc", ""), 3)

    def test_equal_strings(self):
        self.assertEqual(str_dist("abc", "abc"), 0)

    def test_one_char_insert(self):
        self.assertEqual(str_dist("a", "ab"), 1)

    def test_one_char_delete(self):
        self.assertEqual(str_dist("ab", "a"), 1)

    def test_one_char_replace(self):
        self.assertEqual(str_dist("a", "b"), 1)

    def test_multiple_operations(self):
        self.assertEqual(str_dist("kitten", "sitting"), 3)
        self.assertEqual(str_dist("flaw", "lawn"), 2)
        self.assertEqual(str_dist("intention", "execution"), 5)

class TestFunction9(BaseTestCase):
    is_dag = imported_functions[8]

    def test_empty_graph(self):
        self.assertTrue(is_dag([]))

    def test_single_node_graph(self):
        self.assertTrue(is_dag([[]]))

    def test_two_node_acyclic_graph(self):
        self.assertTrue(is_dag([[1], []]))

    def test_two_node_cyclic_graph(self):
        self.assertFalse(is_dag([[1], [0]]))

    def test_three_node_acyclic_graph(self):
        self.assertTrue(is_dag([[1], [2], []]))

    def test_three_node_cyclic_graph(self):
        self.assertFalse(is_dag([[1], [2], [0]]))

    def test_complex_acyclic_graph(self):
        self.assertTrue(is_dag([[1, 2], [3], [3], []]))

    def test_complex_cyclic_graph(self):
        self.assertFalse(is_dag([[1, 2], [3], [3], [1]]))

    def test_self_loop(self):
        self.assertTrue(is_dag([[0]]))

    def test_disconnected_graph(self):
        self.assertTrue(is_dag([[1], [], [3], []]))

class TestFunction10(BaseTestCase):
    foo = imported_functions[9]

    def test_initial_below_10(self):
        self.assertEqual(foo(5), (5, 0))

    def test_initial_exactly_10(self):
        self.assertEqual(foo(10), (10 * (2/3), 1))

    def test_initial_above_10(self):
        self.assertEqual(foo(20), (20 * (2/3) * (2/3), 2))

    def test_initial_below_10_with_counter(self):
        self.assertEqual(foo(5, 2), (5, 2))

    def test_initial_above_10_with_counter(self):
        self.assertEqual(foo(30, 2), (30 * (2/3) * (2/3) * (2/3), 5))

    def test_large_initial_value(self):
        self.assertEqual(foo(1000), (7.707346629258933, 12))

    def test_large_initial_value_with_counter(self):
        self.assertEqual(foo(1000, 3), (7.707346629258933, 15))

    def test_initial_zero(self):
        self.assertEqual(foo(0), (0, 0))

    def test_initial_negative(self):
        self.assertEqual(foo(-20), (-20, 0))

class TestFunction11(BaseTestCase):
    diff_sparse_matrices = imported_functions[10]

    def test_single_matrix(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}]), {(1, 3): 2, (2, 7): 1})

    def test_two_matrices(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}, {(1, 3): 6}]), {(1, 3): -4, (2, 7): 1})

    def test_two_matrices_zero_difference(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}, {(1, 3): 2}]), {(1, 3): 0, (2, 7): 1})

    def test_multiple_matrices(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}, {(1, 3): 6, (9, 10): 7}, {(2, 7): 0.5, (4, 2): 10}]), 
                         {(1, 3): -4, (2, 7): 0.5, (9, 10): -7, (4, 2): -10})

    def test_no_difference(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 5}, {(1, 1): 5}]), {(1, 1): 0})

    def test_negative_values(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 3}, {(1, 1): -4}]), {(1, 1): 7})

    def test_disjoint_matrices(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 3}, {(2, 2): 4}]), {(1, 1): 3, (2, 2): -4})

    def test_multiple_entries(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 1, (2, 2): 2, (3, 3): 3}, {(1, 1): 1, (2, 2): 2}, {(3, 3): 3}]), 
                         {(1, 1): 0, (2, 2): 0, (3, 3): 0})

    def test_zero_values(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 0}, {(1, 1): 3}]), {(1, 1): -3})

class TestFunction12(BaseTestCase):
    longest_subsequence_length = imported_functions[11]

    def test_empty_list(self):
        self.assertEqual(longest_subsequence_length([]), 0)

    def test_all_increasing(self):
        self.assertEqual(longest_subsequence_length([1, 2, 3, 4, 5]), 5)

    def test_all_decreasing(self):
        self.assertEqual(longest_subsequence_length([5, 4, 3, 2, 1]), 5)

    def test_mixed_increasing_decreasing(self):
        self.assertEqual(longest_subsequence_length([1, -4, 7, -5]), 3)

    def test_single_element(self):
        self.assertEqual(longest_subsequence_length([-4]), 1)

    def test_mixed_with_long_subsequence(self):
        self.assertEqual(longest_subsequence_length([1, -4, 2, 9, -8, 10, -6]), 4)

    def test_increasing_then_decreasing(self):
        self.assertEqual(longest_subsequence_length([1, 3, 5, 4, 2]), 3)

    def test_equal_elements(self):
        self.assertEqual(longest_subsequence_length([2, 2, 2, 2, 2]), 1)





# Custom TestResult class to count failures
class CustomTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_counts = [0] * len(functions_list)
        self.total_tests = [0] * len(functions_list)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_case_name = test.__class__.__name__
        for index, case_name in test_cases.items():
            if case_name == test_case_name:
                self.failure_counts[index] += 1
                break

    def startTest(self, test):
        super().startTest(test)
        test_case_name = test.__class__.__name__
        for index, case_name in test_cases.items():
            if case_name == test_case_name:
                self.total_tests[index] += 1
                break

    def failure_fractions(self):
        return [
            (self.failure_counts[i] / self.total_tests[i]) if self.total_tests[i] > 0 else 0
            for i in range(len(self.failure_counts))
        ]

# Main block to run the tests
if __name__ == '__main__':
    # Create an instance of CustomTestResult
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    for index, test_case_name in test_cases.items():
        # Get the test case class from globals()
        test_case_class = globals()[test_case_name]
        # Load tests from the test case class
        tests = test_loader.loadTestsFromTestCase(test_case_class)
        test_suite.addTests(tests)

    # Run the test suite with the custom result
    runner = unittest.TextTestRunner(resultclass=CustomTestResult)
    result = runner.run(test_suite)

    # Print the number of failures for each function
    print("Number of failures for each function:", result.failure_fractions())
