import unittest
import random
import itertools
import numpy as np
import pandas as pd
import math
from unittest.mock import patch


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
""",
"""def find_median(nums): 
    def select(lst, k):
        left, right = 0, len(lst) - 1
        while left <= right:
            pivot_index = random.randint(left, right)
            pivot_value = lst[pivot_index]
            lst[pivot_index], lst[right] = lst[right], lst[pivot_index]
            store_index = left
            for i in range(left, right):
                if lst[i] < pivot_value:
                    lst[store_index], lst[i] = lst[i], lst[store_index]
                    store_index += 1
            lst[store_index], lst[right] = lst[right], lst[store_index]
            if store_index == k:
                return lst[store_index]
            elif store_index < k:
                left = store_index + 1
            else:
                right = store_index - 1
    n = len(nums)
    if n % 2 == 1:
        return select(nums, n // 2)
    else:
        return 0.5 * (select(nums, n // 2 - 1) + select(nums, n // 2))
""",
"""def find_primary_factors(n): 
    factors = []
    k = 2
    while k * k <= n:
        if n % k:
            k += 1
        else:
            n //= k
            factors.append(k)
    if n > 1:
        factors.append(n)
    return factors
""",
"""def graphs_intersection(g1, g2): 
    res_dict = {}
    for node in g1:
        if node in g2:  
            for adj_node in g1[node]:
                if adj_node in g2[node]:  
                    if node in res_dict:
                        res_dict[node].append(adj_node)
                    else:
                        res_dict[node] = [adj_node]
    return res_dict
""",
"""def subset_sum(lst, target): 
    res = set()
    for i in range(len(lst) + 1):
        for subset in itertools.combinations(lst, i):
            if sum(subset) == target:
                res.add(subset)
    return res
""",
"""def sum_mult_str(expression): 
    lst = expression.split(sep = "'")
    lst.remove(lst[0])    
    lst.remove(lst[-1])
    text = lst[0]
    for i in range(1, len(lst), 2):
        if lst[i] == '+':
            text = text + lst[i+1]
        else:
            text = text * int(lst[i+1])
    return(text)
""",
"""def str_rep(s, k): #8
    lst = [s[:k]]
    for i in range(1, len(s) - k + 1):
        if lst.count(s[i:k+i]) != 0:
            return True
        else:
            lst.append(s[i:k+i])
    return False
""",
"""def sort_two_sorted_lists(lst): 
    if len(lst) == 0:
        return []
    new_lst = []
    n = len(lst)
    i_even = 0 
    i_odd = n-1 
    while i_even < n and i_odd > 0 :
        even = lst[i_even]
        odd = lst[i_odd]
        if even == odd:
            new_lst.append(even)
            new_lst.append(odd)
        elif even < odd:
            new_lst.append(even)
            if i_even == n-2:
                new_lst += lst[i_odd::-2]
                return new_lst
            else:
                i_even += 2
        else:
            new_lst.append(odd)
            if i_odd == 1:
               new_lst += lst[i_even::2]
               return new_lst
            else:
                i_odd -= 2
""",
"""def prefix_suffix_match(lst, k): 
    res_lst = []
    for i in range(len(lst)): 
        for j in range(len(lst)): 
            if i == j: 
                continue
            if k > len(lst[i]) or k > len(lst[j]):
                continue
            elif lst[i][:k] == lst[j][-k:]: 
                    res_lst.append((i,j))
    return res_lst

""",
"""def rotate_matrix_clockwise(mat): 
    n = len(mat)
    for i in range(n//2):
        for j in range(i, n-i-1):
            temp = mat[i][j]
            mat[i][j] = mat[n-j-1][i]
            mat[n-j-1][i] = mat[n-i-1][n-j-1]
            mat[n-i-1][n-j-1] = mat[j][n-i-1]
            mat[j][n-i-1] = temp
    return mat
""",
"""def cyclic_shift(lst, direction, steps): 
    if len(lst) == 0:
        return lst
    if (direction == 'L' and steps > 0) or (direction == 'R' and steps < 0):
        for i in range(max(steps, -steps) % len(lst)):
            lst.append(lst.pop(0))
    elif (direction == 'R' and steps > 0) or (direction == 'L' and steps < 0):
        for i in range(max(steps, -steps) % len(lst)):
            lst.insert(0, lst.pop())
    return lst
""",
"""def encode_string(s): 
    curr, count = None, 0
    res = ""
    for c in s:
        if c == curr:
            count += 1
        else:
            if count > 0:
                res += f"{str(count)}[{curr}]"
            curr = c
            count = 1
    if count > 0:
        res += f"{str(count)}[{curr}]"
    return res
""",
"""def list_sums(lst): 
    for i in range(1,len(lst)):
        lst[i] += lst[i-1]
""",
"""def convert_base(num, base): 
    if base > 9 or base < 1 or num < 0:
        return None 
    if num == 0:
        if base == 1:
            return ""
        return "0"
    res = ""
    if base == 1:
        return "1"*num
    while num > 0:
        remainder = num % base
        res = str(remainder) + res
        num //= base
    return res  
""",
"""def max_div_seq(n, k): 
    lst = []
    cnt = 0
    while n > 0:
        if (n % 10) % k == 0:
            cnt += 1
            if n < 10:
                lst.append(cnt)
        else:
            lst.append(cnt)
            cnt = 0
        n = n // 10
    return max(lst)
""",
"""def find_dup(lst): 
    ptr1 = ptr2 = lst[0]
    while True:
        ptr1 = lst[ptr1]
        ptr2 = lst[lst[ptr2]]
        if ptr1 == ptr2:
            break
    ptr1 = lst[0]
    while ptr1 != ptr2:
        ptr1 = lst[ptr1]
        ptr2 = lst[ptr2]
    return ptr1
""",
"""def lcm(a, b): 
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return a * b // gcd(a, b)
""",
"""def f19(): 
    result = None
    for number in range(1000, 10000):
        if number % 15 == 0:
            digits = [int(digit) for digit in str(number)]
            product_of_digits = 1
            for digit in digits:
                product_of_digits *= digit
            if 55 < product_of_digits < 65:
                result = number
                break
    return result
""",
"""def f20(): 
    num_str = str(14563743)
    for i in range(len(num_str)):
        for j in range(i+1, len(num_str)):
            for k in range(j+1, len(num_str)):
                new_number = int(num_str[:i] + num_str[i+1:j] + num_str[j+1:k] + num_str[k+1:])
                if new_number % 22 == 0:
                    return new_number
    return None
""",
"""def convolve_1d(signal, kernel): 
    signal_len = len(signal)
    kernel_len = len(kernel)
    result_len = signal_len + kernel_len - 1
    result = np.zeros(result_len)
    padded_signal = np.pad(signal, (kernel_len - 1, kernel_len - 1), mode='constant')
    flipped_kernel = np.flip(kernel)
    for i in range(result_len):
        result[i] = np.sum(padded_signal[i:i + kernel_len] * flipped_kernel)
    return result
""",
"""def mask_n(im, n, idx): 
    size = (im.max() - im.min()) / n 
    mask_greater = im >= (im.min() + size * idx) 
    mask_lower = im <= (im.min() + size * (idx + 1)) 
    return mask_greater * mask_lower 
""",
"""def entropy(mat):
    mat_values = mat.flatten() 
    bin_prob = np.bincount(mat_values) / (mat_values.shape[0]) 
    bin_prob = bin_prob[bin_prob != 0] 
    return (-bin_prob * np.log2(bin_prob)).sum() 
""",
"""def squeeze_vertical(im, factor):
    max_length = max(len(row) for row in im)
    padded_im = np.array([np.pad(row, (0, max_length - len(row)), 'constant') for row in im])
    h, w = padded_im.shape 
    new_h = h // factor 
    res = np.zeros((new_h, w), dtype=float) 
    for i in range(new_h): 
        res[i, :] = padded_im[i * factor: (i + 1) * factor, :].mean(axis=0) 
    return res
""", 
"""def denoise(im):
    def denoise_pixel(im, x, y, dx, dy):
        down = max(x - dx, 0)
        up = min(x + dx + 1, im.shape[0])
        left = max(y - dy, 0)
        right = min(y + dy + 1, im.shape[1])
        neighbors = im[down:up, left:right]
        good_nbrs = neighbors[neighbors > 0]
        if good_nbrs.size > 0:
            return np.median(good_nbrs)
        return im[x, y]
    new_im = np.zeros(im.shape)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            new_im[x, y] = denoise_pixel(im, x, y, 1, 1)
    return new_im
""",
"""def calculate_monthly_sales(data): 
    if data.empty:
      return pd.DataFrame(columns=['Product', 'YearMonth', 'Sales', 'AverageMonthlySales'])
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = pd.to_datetime(data['Date'])
    data['YearMonth'] = data['Date'].dt.to_period('M')
    monthly_sales = data.groupby(['Product', 'YearMonth'])['Sales'].sum().reset_index()
    monthly_average_sales = monthly_sales.groupby('Product')['Sales'].mean().reset_index()
    monthly_average_sales.rename(columns={'Sales': 'AverageMonthlySales'}, inplace=True)
    result = pd.merge(monthly_sales, monthly_average_sales, on='Product')
    return result
""",
"""def recommendations(movies, movies_genres, genres, search_title): 
    matching_title = movies[movies['title'] == search_title]
    if matching_title.empty:
        return pd.DataFrame(columns=['id', 'title', 'rate', 'runtime'])  
    matching_title = matching_title.iloc[0]
    matching_title_genres = movies_genres[movies_genres['movie_id'] == matching_title['id']]['genre_id'].tolist()
    genre_movie_ids = movies_genres[movies_genres['genre_id'].isin(matching_title_genres)]['movie_id'].tolist()
    filtered_movies = movies[
        (movies['id'].isin(genre_movie_ids)) &
        (movies['rate'].between(matching_title['rate'] - 1, matching_title['rate'] + 1)) &
        (movies['runtime'].between(matching_title['runtime'] - 15, matching_title['runtime'] + 15)) &
        (movies['id'] != matching_title['id'])
    ]
    return filtered_movies.head(3)
""",
"""def top_hours_worked_departments(employees, departments, works_on): 
    if employees.empty or departments.empty or works_on.empty:
        return pd.DataFrame(columns=['department_name', 'total_hours'])
    employees_project_hours = works_on.groupby('employee_id')['hours_worked'].sum().reset_index()
    employees_project_hours = employees_project_hours.merge(employees[['employee_id', 'name', 'department_id']], on='employee_id')
    employees_project_hours = employees_project_hours[['name', 'department_id', 'hours_worked']]
    employees_project_hours = employees_project_hours.rename(columns={'hours_worked': 'total_project_hours'})
    department_hours = employees_project_hours.groupby('department_id')['total_project_hours'].sum().reset_index()
    department_hours = department_hours.merge(departments, on='department_id')
    department_hours = department_hours[['name', 'total_project_hours']]
    department_hours = department_hours.rename(columns={'name': 'department_name', 'total_project_hours': 'total_hours'})
    return department_hours.sort_values(by='total_hours', ascending=False).head(3)
""",
"""def huge_population_countries(countries, borders): 
    if countries.empty or borders.empty:
        return pd.DataFrame(columns=['name', 'population', 'border_population_sum'])
    merged = borders.merge(countries, how='left', left_on='country2', right_on='name')
    merged = merged.rename(columns={'name': 'country_name_2', 'population': 'population_2'})
    merged = merged.merge(countries, how='left', left_on='country1', right_on='name')
    merged = merged.rename(columns={'name': 'country_name_1', 'population': 'population_1'})
    border_population_sum = merged.groupby('country1')['population_2'].sum().reset_index()
    border_population_sum = border_population_sum.rename(columns={'country1': 'name', 'population_2': 'border_population_sum'})
    result = countries.merge(border_population_sum, on='name', how='left')
    filtered_countries = result[result['population'] > result['border_population_sum']]
    return filtered_countries[['name', 'population', 'border_population_sum']]
""",
"""def countries_bordering_most_populated_in_asia(country_df, border_df): 
    asian_countries = country_df[country_df['continent'] == 'Asia']    
    max_population = asian_countries['population'].max()
    most_populated_countries = asian_countries[asian_countries['population'] == max_population]
    bordering_countries_set = set()
    for country in most_populated_countries['name']:
        borders = border_df[(border_df['country1'] == country) | (border_df['country2'] == country)]
        for _, row in borders.iterrows():
            bordering_countries_set.add(row['country1'])
            bordering_countries_set.add(row['country2'])
    bordering_countries_set -= set(most_populated_countries['name'])
    bordering_countries_list = sorted(bordering_countries_set)
    return bordering_countries_list
"""
    ]

classes_list = ["""class triangle: 
    def __init__(self, a, b, ab, color) -> None:
        self.d = {}
        self.d['a'] = a
        self.d['b'] = b
        self.d['ab'] = ab
        self.d['color'] = color
        c = math.sqrt(a**2 + b**2 - 2*b*a*math.cos(math.radians(ab)))
        self.d['c'] = c
        self.d['bc'] = math.degrees(math.acos((b**2 + c**2 - a**2)/(2*b*c)))
        self.d['ac'] = math.degrees(math.acos((a**2 + c**2 - b**2)/(2*a*c)))

    def get(self, name):
        if len(name) == 2:
            name = "".join(sorted(name))
        if name not in self.d:
            raise KeyError(f"ERROR: no triangale attribute with the name {name}.")
        return self.d[name]""",
"""class worker: 

    def __init__(self, id, first_name, last_name, job, salary = 5000, second_name = None):
        self.id = id
        if second_name:
            self.full_name = first_name + " " + second_name + " " + last_name
        else: 
            self.full_name = first_name + " " + last_name
        self.job = job
        self.salary = salary
    
    def getFullName(self):
        return self.full_name
    
    def getSalary(self):
        return self.salary
    
    def getJob(self):
        return self.job
    
    def update(self, job = None, salary = None):
        if job:
            self.job = job
        if salary:
            self.salary = salary
""",
"""class binaric_arithmatic: 

    def __init__(self, num):
        self.num = num
    
    def get(self):
        return self.num
    
    def inc(self):
        if self.num == "0":
            return "1"
        new_bin_rev = ""
        bin_rev = self.num[::-1]
        for i in range(len(self.num)):
            if bin_rev[i] == "1":
                new_bin_rev = new_bin_rev + "0"
            else:
                new_bin_rev = new_bin_rev + "1" + bin_rev[i+1:]
                return new_bin_rev[::-1]
        if "1" not in new_bin_rev:
            return "1" + new_bin_rev
    
    def dec(self):
        if self.num == "1":
            return "0"
        new_bin_rev = ""
        bin_rev = self.num[::-1]
        for i in range(len(self.num)):
            if bin_rev[i] == "0":
                new_bin_rev = new_bin_rev + "1"
            else:
                if i == (len(self.num) - 1):
                    new_bin_rev = new_bin_rev + "0"
                    break
                new_bin_rev = new_bin_rev + "0" + bin_rev[i + 1:]
                break
        if new_bin_rev[-1] == "0":
            return new_bin_rev[:-1][::-1]
        return new_bin_rev[::-1] 
""",
"""class Point_2D: 

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = math.sqrt(x**2 + y**2)
        self.theta = math.atan2(y, x)
     
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        return Point_2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point_2D(self.x - other.x, self.y - other.y)
    
    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_wrt_origin(self, other):
        dif_angle = other.theta - self.theta
        if dif_angle < 0:
            return dif_angle + 2 * math.pi
        return dif_angle  
""", 
"""class Roulette: 

    def __init__(self, initial_money):
        self.balance = initial_money
        self.reds = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
        self.blacks = [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]
    
    def get_balance(self):
        return self.balance
    
    def bet(self, amount, bet_type):
        if amount > self.balance:
            raise KeyError(f"ERROR: current balance = {self.balance}, can't bet {amount}.")
        roll = random.randint(0, 36)
        print("roll: ", roll)
        if bet_type == "red":
            self.balance -= amount
            if roll in self.reds:
                self.balance += amount * 2
        elif bet_type == "black":
            self.balance -= amount
            if roll in self.blacks:
                self.balance += amount * 2
        elif bet_type == "even":
            self.balance -= amount
            if roll > 0 and roll % 2 == 0:
                self.balance += amount * 2
        elif bet_type == "odd":
            self.balance -= amount
            if roll > 0 and roll % 2 == 1:
                self.balance += amount * 2
        elif bet_type == "1-12":
            self.balance -= amount
            if roll > 0 and roll < 13:
                self.balance += amount * 2
        elif bet_type == "13-24":
            self.balance -= amount
            if roll > 12 and roll < 25:
                self.balance += amount * 2
        elif bet_type == "25-36":
            self.balance -= amount
            if roll > 24 and roll < 37:
                self.balance += amount * 2      
        else:
            self.balance -= amount
            if roll == int(bet_type):
                self.balance += amount * 36
        return self.balance
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

def string_to_class(class_str):
    """
    Converts a string representation of a class into a runnable class.
    
    Parameters:
    class_str (str): A string representation of the class to be converted.
    
    Returns:
    class: A runnable class object.
    """
    local_namespace = {}
    code = compile(class_str, '<string>', 'exec')
    exec(code, globals(), local_namespace)
    for item in local_namespace.values():
        if isinstance(item, type):
            return item
    # Raise an error if the class definition is not found
    raise ValueError("Class definition not found in the provided string.")

# Dictionary to map function indices to their respective test cases
test_cases = {i: f'TestGeneratedSolution{i+1}' for i in range(len(functions_list) + len(classes_list))}

# Transform string representations to runnable functions
imported_functions = [string_to_function(func_str) for func_str in functions_list]

imported_classes = [string_to_class(class_str) for class_str in classes_list]

# Base class for test cases to inherit from
class BaseTestCase(unittest.TestCase):
    def setUp(self):
        pass  # Optional setup method to run before each test

    def tearDown(self):
        pass  # Optional teardown method to run after each test

class TestGeneratedSolution1(BaseTestCase):
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



class TestGeneratedSolution2(BaseTestCase):
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

class TestGeneratedSolution3(BaseTestCase):
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

class TestGeneratedSolution4(BaseTestCase):
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

class TestGeneratedSolution5(BaseTestCase):
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

    
class TestGeneratedSolution6(BaseTestCase):
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

class TestGeneratedSolution7(BaseTestCase):
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


class TestGeneratedSolution8(BaseTestCase):
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
        self.assertEqual(str_dist("flaw", "lawn"), 2)
        self.assertEqual(str_dist("intention", "execution"), 5)

class TestGeneratedSolution9(BaseTestCase):
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

class TestGeneratedSolution10(BaseTestCase):
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

class TestGeneratedSolution11(BaseTestCase):
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

class TestGeneratedSolution12(BaseTestCase):
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

class TestGeneratedSolution13(BaseTestCase):
    find_median = imported_functions[12]
    
    def test_odd_number_of_elements(self):
        self.assertEqual(find_median([1, 2, 3, 4, 5]), 3)
        self.assertEqual(find_median([5, 4, 3, 2, 1]), 3)
        self.assertEqual(find_median([3, 1, 2]), 2)

    def test_even_number_of_elements(self):
        self.assertAlmostEqual(find_median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(find_median([1, -4, 7, -5]), -1.5)
        self.assertAlmostEqual(find_median([1, 2, -4, -7]), -1.5)

    def test_single_element(self):
        self.assertEqual(find_median([7]), 7)
        self.assertEqual(find_median([-1]), -1)

    def test_negative_numbers(self):
        self.assertEqual(find_median([-1, -2, -3, -4, -5]), -3)
        self.assertEqual(find_median([-5, -3, -1, -2, -4]), -3)

    def test_mixed_positive_and_negative_numbers(self):
        self.assertEqual(find_median([1, -1, 2, -2, 0]), 0)
        self.assertEqual(find_median([-1, 0, 1]), 0)

    def test_duplicate_numbers(self):
        self.assertEqual(find_median([1, 2, 2]), 2)
        self.assertEqual(find_median([2, 2, 2, 2, 2]), 2)
        
    def test_large_list(self):
        self.assertEqual(find_median(list(range(1, 101))), 50.5)
        
    def test_float_list(self):
        self.assertEqual(find_median([1.5, 2.5, 3.5, 4.5]), 3.0)
    
    def test_mix_float_and_int(self):
        self.assertEqual(find_median([1.5, 2, 3.5, 4]), 2.75)

class TestGeneratedSolution14(BaseTestCase):
    find_primary_factors = imported_functions[13]

    def test_prime_number(self):
        self.assertEqual(find_primary_factors(7), [7])

    def test_composite_number(self):
        self.assertEqual(find_primary_factors(12), [2, 2, 3])
        self.assertEqual(find_primary_factors(105), [3, 5, 7])

    def test_large_prime_number(self):
        self.assertEqual(find_primary_factors(101), [101])
        self.assertEqual(find_primary_factors(1000000007), [1000000007])

    def test_large_composite_number(self):
        self.assertEqual(find_primary_factors(1001), [7, 11, 13])
        self.assertEqual(find_primary_factors(1524878), [2, 29, 61, 431])
        self.assertEqual(find_primary_factors(97*89), [89, 97])

    def test_large_composite_number_with_repeated_factors(self):
        self.assertEqual(find_primary_factors(1000), [2, 2, 2, 5, 5, 5])
        self.assertEqual(find_primary_factors(2**10), [2]*10)
        self.assertEqual(find_primary_factors(2**4 * 3**3), [2, 2, 2, 2, 3, 3, 3])

    def test_large_composite_number_with_large_prime_factors_and_repeated_factors(self):
        self.assertEqual(sorted(find_primary_factors(1524878*29)), [2, 29, 29, 61, 431])
    
    def test_no_factors(self):
        self.assertEqual(find_primary_factors(1), [])
    
    def test_smallest_prime(self):
        self.assertEqual(find_primary_factors(2), [2])
        
class TestGeneratedSolution15(BaseTestCase):
    graphs_intersection = imported_functions[14]

    def test_empty_graphs(self):
        self.assertEqual(graphs_intersection({}, {}), {})

    def test_single_node_graphs(self):
        self.assertEqual(graphs_intersection({1: []}, {1: []}), {})

    def test_single_edge_graphs(self):
        self.assertEqual(graphs_intersection({1: [2]}, {1: [2]}), {1: [2]})

    def test_single_edge_graphs_no_intersection(self):
        self.assertEqual(graphs_intersection({1: [2]}, {1: [3]}), {})

    def test_single_edge_graphs_different_nodes(self):
        self.assertEqual(graphs_intersection({1: [2]}, {3: [4]}), {})

    def test_single_edge_graphs_opposite_direction(self):
        self.assertEqual(graphs_intersection({1: [2]}, {2: [1]}), {})

    def test_single_edge_graphs_shared_node_no_intersection(self):
        self.assertEqual(graphs_intersection({1: [2]}, {2: [3]}), {})

    def test_graphs_form_a_directed_cycle_no_intersection(self):
        self.assertEqual(graphs_intersection({1: [2]}, {2: [3], 3: [1]}), {})

    def test_same_multiple_edges_graphs(self):
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [3]}, {1: [2, 3], 2: [3]}), {1: [2, 3], 2: [3]})

    def test_intersection_is_subset_of_the_graphs(self):
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [3]}, {1: [3], 2: [4]}), {1: [3]})     
    
    def test_intersection_lager_graphs(self):
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]} , {1: [3, 4], 2: [3, 5], 3: [1, 2], 4: [1], 5: [2]}), {1: [3], 2: [3], 3: [1, 2]})
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]} , {1: [2, 3, 5], 2: [1, 3], 3: [1, 2], 4: [5], 5: [1, 4]}), {1: [2, 3], 2: [1, 3], 3: [1, 2]})
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]} , {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3]}), {1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]})   

class TestGeneratedSolution16(BaseTestCase):
    subset_sum = imported_functions[15]

    def test_empty_list_zero_target(self):
        self.assertEqual(subset_sum([], 0), {()})
        
    def test_empty_list_non_zero_target(self):
        self.assertEqual(subset_sum([], 1), set())

    def test_single_item_list(self):
        self.assertEqual(subset_sum([1], 1), {(1,)})

    def test_single_item_list_no_sum(self):
        self.assertEqual(subset_sum([1], 2), set())

    def test_list_sums_to_target(self):
        self.assertEqual(subset_sum([1, 2], 3), {(1, 2)})

    def test_target_greater_than_sum_of_list(self):
        self.assertEqual(subset_sum([1, 2], 4), set())

    def test_no_subset_sum(self):
        self.assertEqual(subset_sum([1, 2, 6], 5), set())
    
    def test_multiple_subsets(self):
        self.assertEqual(subset_sum([1, 2, 3], 3), {(3,), (1, 2)})
    
    def test_multiple_subsets_with_duplicates_in_list(self):
        self.assertEqual(subset_sum([1, 2, 2], 3), {(1, 2)})
        self.assertEqual(subset_sum([1, 2, 2, 3], 3), {(3,), (1, 2)})
        
    def test_multiple_subsets_with_repeats_in_subsets(self):
        self.assertEqual(subset_sum([1, 2, 2], 5), {(1, 2, 2)})
        self.assertEqual(subset_sum([1, 2, 2, 3], 3), {(3,), (1, 2)})
    
    def test_negatives(self):
        self.assertEqual(subset_sum([-2, -1, 3], 2), {(-1, 3)})
    
    def test_negatives_with_zero_target(self):
        self.assertEqual(subset_sum([-2, -1, 3], 0), {(-2, -1, 3), ()})

    def test_negatives_with_negative_target(self):
        self.assertEqual(subset_sum([-2, -1, 3], -3), {(-2, -1)})
        
    def test_negatives_with_zero_element_and_zero_target(self):
        self.assertEqual(subset_sum([-2, -1, 0, 3], 0), {(0,), (), (-2, -1, 0, 3), (-2, -1, 3)})
    
    def test_list_is_singelton_zero_and_zero_target(self):
        self.assertEqual(subset_sum([0], 0), {(0,), ()})

class TestGeneratedSolution17(BaseTestCase):
    sum_mult_str = imported_functions[16]
    
    def test_single_string(self):
        self.assertEqual(sum_mult_str("'hello'"), "hello")
    
    def test_two_string_operands_addition_only(self):
        self.assertEqual(sum_mult_str("'a'+'b'"), "ab")
    
    def test_string_multiplication_only(self):
        self.assertEqual(sum_mult_str("'a'*'3'"), "aaa")
    
    def test_multiple_operands_addition_only(self):
        self.assertEqual(sum_mult_str("'a'+'b'+'c'"), "abc")
        
    def test_multiple_operands_multiplication_only(self):
        self.assertEqual(sum_mult_str("'a'*'3'*'2'"), "aaaaaa")
    
    def test_numbers_numltiplication_(self):
        self.assertEqual(sum_mult_str("'3'*'3'"), "333")
    
    def test_mixed_operands(self):
        self.assertEqual(sum_mult_str("'abc'*'3'+'def'"), "abcabcabcdef")
    
    def test_mixed_operands_unintuitive_order(self):
        self.assertEqual(sum_mult_str("'12'+'aa'*'2'"), "12aa12aa")
    
    def test_empty_string(self):
        self.assertEqual(sum_mult_str("''"), "")
    
    def test_add_empty_string(self):
        self.assertEqual(sum_mult_str("'a'+''"), "a")
    
    def test_multiply_by_zero(self):
        self.assertEqual(sum_mult_str("'a'*'0'"), "")
    
    def test_no_operations(self):
        self.assertEqual(sum_mult_str("'a'"), "a")
        self.assertEqual(sum_mult_str("'73'"), "73")
        
class TestGeneratedSolution18(BaseTestCase):
    str_rep = imported_functions[17]
    
    def test_with_repeats(self):
        self.assertTrue(str_rep("abcabc", 3))
        self.assertTrue(str_rep("aab2bab22", 3))
        
    def test_no_repeats(self):
        self.assertFalse(str_rep("abcabc", 4))
        self.assertFalse(str_rep("aab2bab22", 4))
    
    def test_single_char(self):
        self.assertFalse(str_rep("a", 1))
        self.assertFalse(str_rep("a", 2))
    
    def test_empty_string(self):
        self.assertFalse(str_rep("", 1))
        self.assertFalse(str_rep("", 2))
    
    def test_repeating_substring_of_length_1(self):
        self.assertTrue(str_rep("aba", 1))
        
    def test_long_string_with_repeating_substring(self):
        self.assertTrue(str_rep("abcdefghijklmnopabcdefghijklmnop", 16))
    
    def test_repeating_substring_with_overlap(self):
        self.assertTrue(str_rep("ababa", 3))

class TestGeneratedSolution19(BaseTestCase):
    sort_two_sorted_lists = imported_functions[18]
    
    def test_empty_list(self):
        self.assertEqual(sort_two_sorted_lists([]), [])
    
    def test_two_items_list(self):
        self.assertEqual(sort_two_sorted_lists([1, 2]), [1, 2])
        self.assertEqual(sort_two_sorted_lists([2, 1]), [1, 2])
        
    def test_with_negatives(self):
        self.assertEqual(sort_two_sorted_lists([-3, 1, -1, -2]), sorted([-3, 1, -1, -2]))
    
    def test_large_list(self):
        self.assertEqual(sort_two_sorted_lists([7, 6, 11, 4, 12, 0, 20, -10, 30, -30]), sorted([7, 6, 11, 4, 12, 0, 20, -10, 30, -30]))
    
class TestGeneratedSolution20(BaseTestCase):
    prefix_suffix_match = imported_functions[19]
    
    def test_empty_list(self):
        self.assertEqual(prefix_suffix_match([], 1), [])
        self.assertEqual(prefix_suffix_match([], 6), [])
        
    def test_single_item_list(self):
        self.assertEqual(prefix_suffix_match(["abc"], 1), [])
        self.assertEqual(prefix_suffix_match(["abc"], 3), [])
    
    def test_no_matches(self):
        self.assertEqual(prefix_suffix_match(["abc", "def"], 1), [])
        self.assertEqual(prefix_suffix_match(["abc", "def"], 3), [])
    
    def test_single_match(self):
        self.assertEqual(prefix_suffix_match(["abc", "cde"], 1), [(1, 0)])
    
    def test_symetric_match(self):
        self.assertEqual(prefix_suffix_match(["aa", "aa"], 1), [(0, 1), (1, 0)])
        self.assertEqual(prefix_suffix_match(["aa", "aa"], 2), [(0, 1), (1, 0)])
        
    def test_multiple_matches(self):
        self.assertEqual(prefix_suffix_match(["aaa", "cba", "baa"], 2), [(0, 2), (2, 1)])
        self.assertEqual(prefix_suffix_match(["abc", "bc", "c"], 1), [(2, 0), (2, 1)])

    def test_empty_string_no_match(self):
        self.assertEqual(prefix_suffix_match(["", "abc"], 1), [])
        
    def test_mix_empty_string_match(self):
        self.assertEqual(prefix_suffix_match(["", "abc", "", "cba"], 1), [(1, 3), (3, 1)])
        
    def test_k_greater_than_string_length(self):
        self.assertEqual(prefix_suffix_match(["abc", "abc"], 4), [])

    def test_special_characters(self):
        self.assertEqual(prefix_suffix_match(["ab#c", "#cde"], 2), [(1, 0)])
        self.assertEqual(prefix_suffix_match(["a!c", "b!c"], 2), [])

    def test_strings_with_spaces(self):
        self.assertEqual(prefix_suffix_match(["abc ", " cde"], 1), [(1, 0)])
        self.assertEqual(prefix_suffix_match(["abc ", "c de"], 2), [(1, 0)])
    
    def test_all_identical_strings(self):
        self.assertEqual(prefix_suffix_match(["aaa", "aaa", "aaa"], 2), [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
        
    def test_case_sensitivity(self): 
        self.assertEqual(prefix_suffix_match(["Ab", "Ba"], 1), [])
    
class TestGeneratedSolution21(BaseTestCase):
    rotate_matrix_clockwise = imported_functions[20]
    
    def test_empty_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([]), [])
    
    def test_single_item_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1]]), [[1]])
        
    def test_two_by_two_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1, 2], [3, 4]]), [[3, 1], [4, 2]])
    
    def test_three_by_three_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[7, 4, 1], [8, 5, 2], [9, 6, 3]])
    
    def test_four_by_four_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), 
                         [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]])
    
    def test_matrix_with_negative_numbers(self):
        self.assertEqual(rotate_matrix_clockwise([[1, -2, 3], [-4, 5, -6], [7, -8, 9]]), [[7, -4, 1], [-8, 5, -2], [9, -6, 3]])
    
class TestGeneratedSolution22(BaseTestCase):
    cyclic_shift = imported_functions[21]
    
    def test_empty_list(self):
        self.assertEqual(cyclic_shift([], 'R', 2), [])    
        
    def test_single_item_list(self):
        self.assertEqual(cyclic_shift([1], 'R', 1), [1])
        self.assertEqual(cyclic_shift([1], 'L', 1), [1])
        
    def test_two_item_list(self):
        self.assertEqual(cyclic_shift([1, 2], 'R', 1), [2, 1])
        self.assertEqual(cyclic_shift([1, 2], 'L', 1), [2, 1])
        self.assertEqual(cyclic_shift([1, 2], 'R', 2), [1, 2])
        self.assertEqual(cyclic_shift([1, 2], 'L', 2), [1, 2])
        
    def test_three_item_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 1), [3, 1, 2])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 1), [2, 3, 1])
    
    def test_shift_larger_than_length_of_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 4), [3, 1, 2])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 4), [2, 3, 1])
        
    def test_shift_negative(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', -1), [2, 3, 1])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', -1), [3, 1, 2])
          
    def test_shift_negative_larger_than_length_of_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', -4), [3, 1, 2])
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', -4), [2, 3, 1])
    
    def test_shift_zero(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 0), [1, 2, 3])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 0), [1, 2, 3])
    
    def test_shift_equal_to_length_of_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 3), [1, 2, 3])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 3), [1, 2, 3])
        
class TestGeneratedSolution23(BaseTestCase):
    encode_string = imported_functions[22]         
    
    def test_empty_string(self):
        self.assertEqual(encode_string(""), "")
        
    def test_string_no_repetitions(self):
        self.assertEqual(encode_string("abc"), "1[a]1[b]1[c]")
    
    def test_string_with_repetitions(self):
        self.assertEqual(encode_string("aabbcc"), "2[a]2[b]2[c]")
        self.assertEqual(encode_string("aaa"), "3[a]")
        self.assertEqual(encode_string("abbcdbaaa"), "1[a]2[b]1[c]1[d]1[b]3[a]")
    
    def test_string_with_numbers(self):
        self.assertEqual(encode_string("a222bb"), "1[a]3[2]2[b]")
    
    def test_string_with_special_characters(self):
        self.assertEqual(encode_string("a##b$"), "1[a]2[#]1[b]1[$]")
    
    def test_string_with_spaces(self):
        self.assertEqual(encode_string("a   b c"), "1[a]3[ ]1[b]1[ ]1[c]")
    
    def test_very_long_string(self):
        long_string = "a" * 1000 + "b" * 1000 + "c" * 1000
        expected_result = "1000[a]1000[b]1000[c]"
        self.assertEqual(encode_string(long_string), expected_result)
    
    def test_single_character_string(self):
        self.assertEqual(encode_string("a"), "1[a]")
        self.assertEqual(encode_string(" "), "1[ ]")
        self.assertEqual(encode_string("#"), "1[#]")    

class TestGeneratedSolution24(BaseTestCase):
    list_sums = imported_functions[23]
    
    def test_empty_list(self):
        self.assertIsNone(list_sums([]))
    
    def test_single_item_list(self):
        single_item_list = [1] 
        list_sums(single_item_list)
        self.assertEqual(single_item_list, [1])
        
    def test_multiple_item_list(self):
        multiple_item_list = [1, 2, 3, 4, 5]
        list_sums(multiple_item_list)
        self.assertEqual(multiple_item_list, [1, 3, 6, 10, 15])
    
    def test_list_with_negative_numbers(self):
        negative_numbers_list = [1, -2, 3, -4, 5]
        list_sums(negative_numbers_list)
        self.assertEqual(negative_numbers_list, [1, -1, 2, -2, 3])
    
    def test_list_with_zeros(self):
        zeros_list = [0, 0, 0, 0, 0]
        list_sums(zeros_list)
        self.assertEqual(zeros_list, [0, 0, 0, 0, 0])
        
    def test_list_with_repeated_numbers(self):
        repeated_numbers_list = [1, 1, 1, 1, 1]
        list_sums(repeated_numbers_list)
        self.assertEqual(repeated_numbers_list, [1, 2, 3, 4, 5])
    
    def test_list_with_floats(self):
        floats_list = [1.5, 2.5, 3.5, 4.5, 5.5]
        list_sums(floats_list)
        self.assertEqual(floats_list, [1.5, 4.0, 7.5, 12.0, 17.5])
        
    def test_multiple_calls(self):
        multiple_calls_list = [1, 2, 3, 4, 5]
        list_sums(multiple_calls_list)
        list_sums(multiple_calls_list)
        self.assertEqual(multiple_calls_list, [1, 4, 10, 20, 35])
    
class TestGeneratedSolution25(BaseTestCase):
    convert_base = imported_functions[24]
    
    def test_base_2(self):
        self.assertEqual(convert_base(10, 2), "1010")
        self.assertEqual(convert_base(15, 2), "1111")
        self.assertEqual(convert_base(255, 2), "11111111")

    def test_base_8(self):
        self.assertEqual(convert_base(10, 8), "12")
        self.assertEqual(convert_base(15, 8), "17")
        self.assertEqual(convert_base(255, 8), "377")    
    
    def test_unaric_base(self):
        self.assertEqual(convert_base(10, 1), "1" * 10)
        self.assertEqual(convert_base(15, 1), "1" * 15)
        
    def test_base_5(self):
        self.assertEqual(convert_base(80, 5), "310")
        
    def test_zero(self):
        self.assertEqual(convert_base(0, 2), "0")
        self.assertEqual(convert_base(0, 8), "0")
        self.assertEqual(convert_base(0, 5), "0")
    
    def test_zero_unaric_base(self):
        self.assertEqual(convert_base(0, 1), "")
        
    def test_negative_number(self):
        self.assertIsNone(convert_base(-10, 2))
    
    def test_negative_base(self):
        self.assertIsNone(convert_base(10, -2))
        
    def test_base_out_of_range(self):
        self.assertIsNone(convert_base(10, 37))
        self.assertIsNone(convert_base(10, 10))
        self.assertIsNone(convert_base(10, 0))
        
    def test_large_number(self):
        self.assertEqual(convert_base(1024, 2), "10000000000")
        self.assertEqual(convert_base(1024, 8), "2000")
        self.assertEqual(convert_base(1024, 5), "13044")
    
class TestGeneratedSolution26(BaseTestCase):
    max_div_seq = imported_functions[25]
    
    def test_div_seq_length_1(self):
        self.assertEqual(max_div_seq(123456, 3), 1)
        self.assertEqual(max_div_seq(123456, 5), 1)
    
    def test_n_is_single_digit_and_divisable_by_k(self):
        self.assertEqual(max_div_seq(6, 2), 1)
    
    def test_n_is_single_digit_and_not_divisable_by_k(self):
        self.assertEqual(max_div_seq(7, 2), 0)
    
    def test_dev_seq_greater_than_1(self):
        self.assertEqual(max_div_seq(124568633, 2), 3)
    
    def test_k_equals_1(self):
        self.assertEqual(max_div_seq(124568633, 1), 9)
        
    def test_no_digits_divisible_by_k(self):
        self.assertEqual(max_div_seq(123456, 7), 0)
    
class TestGeneratedSolution27(BaseTestCase):
    find_dup = imported_functions[26]
    
    def test_large_input(self):
        self.assertEqual(find_dup(list(range(1, 10001)) + [5000]), 5000)

    def test_duplicate_at_end(self):
        self.assertEqual(find_dup([1, 3, 4, 2, 5, 5]), 5)
    
    def test_duplicate_at_start(self):
        self.assertEqual(find_dup([1, 1, 2, 3, 4, 5]), 1)
    
    def test_duplicate_in_middle(self):
        self.assertEqual(find_dup([1, 2, 3, 4, 3, 5, 6]), 3)
    
    def test_two_elements(self):
        self.assertEqual(find_dup([1, 1]), 1)
    
class TestGeneratedSolution28(BaseTestCase):
    lcm = imported_functions[27]
    
    def test_basic_cases(self):
        self.assertEqual(lcm(3, 5), 15)
        self.assertEqual(lcm(4, 6), 12)

    def test_one_is_multiple_of_other(self):
        self.assertEqual(lcm(6, 3), 6)
        self.assertEqual(lcm(10, 5), 10)
    
    def test_prime_numbers(self):
        self.assertEqual(lcm(7, 11), 7 *11)
        self.assertEqual(lcm(13, 17), 13 * 17)

    def test_large_numbers_and_lcm_is_not_their_product(self):
        self.assertEqual(lcm(123456, 789012), 8117355456)

    def test_one_is_one(self):
        self.assertEqual(lcm(1, 99), 99)
        self.assertEqual(lcm(1, 1), 1)
    
    def test_equal_numbers(self):
        self.assertEqual(lcm(8, 8), 8)

    def test_coprime_numbers(self):
        self.assertEqual(lcm(9, 14), 9 * 14)
        self.assertEqual(lcm(15, 22), 15 * 22)
    
    def test_numbers_with_common_factors(self):
        self.assertEqual(lcm(18, 24), 72)
        self.assertEqual(lcm(40, 60), 120)
    
class TestGeneratedSolution29(BaseTestCase):
    f19 = imported_functions[28]
    
    def test_smallest_valid_number(self):
        self.assertEqual(f19(), 2235)
    
    def test_product_of_digits_within_range(self):
        result = f19()
        digits = [int(digit) for digit in str(result)]
        product_of_digits = 1
        for digit in digits:
            product_of_digits *= digit
        self.assertTrue(56 <= product_of_digits <= 64)
    
    def test_divisibility_by_15(self):
        result = f19()
        self.assertEqual(result % 15, 0)
    
    def test_is_four_digits(self):
        result = f19()
        self.assertTrue(1000 <= result <= 9999)    
    
class TestGeneratedSolution30(BaseTestCase):
    f20 = imported_functions[29]
    
    def test_expected_output(self):
        self.assertEqual(f20(), 14674)

    def test_divisibility_by_22(self):
        result = f20()
        self.assertIsNotNone(result)
        self.assertEqual(result % 22, 0)

    def test_length_of_result(self):
        result = f20()
        self.assertIsNotNone(result)
        self.assertEqual(len(str(result)), len(str(14563743)) - 3)

    def test_result_is_from_original_digits(self):
        result = f20()
        self.assertIsNotNone(result)
        result_str = str(result)
        num_str = str(14563743)
        for digit in result_str:
            self.assertIn(digit, num_str)
        self.assertEqual(len(result_str) + 3, len(num_str))
    
class TestGeneratedSolution31(BaseTestCase):
    convolve_1d = imported_functions[30]
    
    def test_basic_convolution(self):
        signal = np.array([1, 2, 3, 4, 5])
        kernel = np.array([0.2, 0.5, 0.2])
        expected_result = np.array([0.2, 0.9, 1.8, 2.7, 3.6, 3.3, 1.0])
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))

    def test_zero_kernel(self):
        signal = np.array([1, 2, 3, 4, 5])
        kernel = np.array([0])
        expected_result = np.zeros(len(signal))
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))

    def test_identity_kernel(self):
        signal = np.array([1, 2, 3, 4, 5])
        kernel = np.array([1])
        expected_result = np.array([1, 2, 3, 4, 5])
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))

    def test_unity_signal(self):
        signal = np.ones(5)
        kernel = np.array([0.2, 0.5, 0.2])
        expected_result = np.array([0.2, 0.7, 0.9, 0.9, 0.9, 0.7, 0.2])
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))    
    
class TestGeneratedSolution32(BaseTestCase):
    mask_n = imported_functions[31]
    
    def test_idx_zero(self):
        im = np.array([[3, 5, 9], [8, 1, 2], [7, 6, 4]])
        expected_mask = np.array([[True, False, False], [False, True, True], [False, False, False]])
        self.assertTrue(np.all(mask_n(im, 3, 0) == expected_mask))

    def test_idx_one(self):
        im = np.array([[3, 5, 9], [8, 1, 2], [7, 6, 4]])
        expected_mask = np.array([[False, True, False], [False, False, False], [False, True, True]])
        self.assertTrue(np.all(mask_n(im, 3, 1) == expected_mask))
        
    def test_n_equals_1(self):
        im = np.array([[3, 5, 9], [8, 1, 2], [7, 6, 4]])
        expected_mask = np.array([[False, False, True], [False, False, False], [False, False, False]])
        self.assertTrue(np.all(mask_n(im, 1, 1) == expected_mask))
    
    def test_same_pixels_image(self):
        im = np.ones((3, 3))
        expected_mask = np.array([[True, True, True], [True, True, True], [True, True, True]])
        self.assertTrue(np.all(mask_n(im, 3, 1) == expected_mask))
        
    def test_large_differece_between_pixels(self):
        im = np.array([[1, 100, 1], [100, 1, 100], [1, 100, 1]])
        expected_mask = np.array([[False, False, False], [False, False, False], [False, False, False]])
        self.assertTrue(np.all(mask_n(im, 3, 1) == expected_mask))
    
class TestGeneratedSolution33(BaseTestCase):
    entropy = imported_functions[32]

    def test_single_value_matrix(self):
        mat = np.array([[5]])
        self.assertAlmostEqual(entropy(mat), 0.0)

    def test_all_same_values(self):
        mat = np.array([[1, 1], [1, 1]])
        self.assertAlmostEqual(entropy(mat), 0.0)

    def test_uniform_distribution(self):
        mat = np.array([[1, 2], [2, 1]])
        self.assertAlmostEqual(entropy(mat), 1.0)

    def test_non_uniform_distribution(self):
        mat = np.array([[3, 5, 3], [8, 1, 1], [7, 7, 7]])
        self.assertAlmostEqual(entropy(mat), 2.197159723424149)

    def test_small_matrix(self):
        mat = np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(entropy(mat), 1.0)

    def test_unique_values(self):
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # Entropy for a matrix with all unique values should be log2(9)
        self.assertAlmostEqual(entropy(mat), np.log2(9))

    def test_matrix_with_zeros(self):
        mat = np.array([[0, 0, 0], [1, 1, 1]])
        # Probability: 3/6 for 0 and 3/6 for 1
        expected_entropy = -(0.5 * np.log2(0.5) + 0.5 * np.log2(0.5))
        self.assertAlmostEqual(entropy(mat), expected_entropy)

    def test_large_matrix(self):
        mat = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9]
        ])
        expected_entropy = -(3 * (0.16 * np.log2(0.16)) + 0.12 * np.log2(0.12)  + 2*(0.2 * np.log2(0.2)))
        self.assertAlmostEqual(entropy(mat), expected_entropy)

class TestGeneratedSolution34(unittest.TestCase):
    squeeze_vertical = imported_functions[33]

    def test_squeeze_vertical_basic(self):
        im = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        factor = 2
        expected = np.array([[2, 3], [6, 7]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    def test_squeeze_vertical_whole_image(self):
        im = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        factor = 4
        expected = np.array([[4, 5]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    def test_squeeze_vertical_no_squeeze(self):
        im = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        factor = 1
        expected = im
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    def test_squeeze_vertical_different_image(self):
        im = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]])
        factor = 2
        expected = np.array([[25, 35, 45], [85, 95, 105]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    def test_squeeze_vertical_5x5(self):
        im = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9]
        ])
        factor = 5
        expected = np.array([[4.2, 4.2, 4.2, 4.2, 4.2]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)

    def test_squeeze_vertical_with_padding(self):
        im = np.array([[1, 2, 5], [3, 4], [5, 6, 8], [7, 8]])
        factor = 2
        expected = np.array([[2.0, 3.0, 2.5], [6.0, 7.0, 4.0]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    def test_squeeze_vertical_single_row(self):
        im = np.array([[1, 2, 3]])
        factor = 1
        expected = np.array([[1, 2, 3]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    def test_squeeze_vertical_single_column(self):
        im = np.array([[1], [2], [3], [4]])
        factor = 2
        expected = np.array([[1.5], [3.5]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)

    def test_squeeze_vertical_single_element(self):
        im = np.array([[1]])
        factor = 1
        expected = np.array([[1]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)

class TestGeneratedSolution35(unittest.TestCase):
    denoise = imported_functions[34]
    
    def test_denoise_basic(self):
        im = np.array([[15, 110, 64, 150], 
                       [231, 150, 98, 160], 
                       [77, 230, 2, 0], 
                       [100, 81, 189, 91]])
        
        expected_result = np.array([[130, 104, 130, 124], 
                                    [130, 98, 130, 98], 
                                    [125, 100, 124, 98], 
                                    [90.5, 90.5, 91, 91]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result, decimal=1)

    def test_denoise_all_zeros(self):
        im = np.array([[0, 0, 0], 
                       [0, 0, 0], 
                       [0, 0, 0]])
        
        expected_result = np.array([[0, 0, 0], 
                                    [0, 0, 0], 
                                    [0, 0, 0]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    def test_denoise_single_value(self):
        im = np.array([[100]])
        
        expected_result = np.array([[100]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    def test_denoise_single_row(self):
        im = np.array([[1, 2, 3, 4, 5]])
        
        expected_result = np.array([[1.5, 2, 3, 4, 4.5]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    def test_denoise_single_column(self):
        im = np.array([[1], [2], [3], [4], [5]])
        
        expected_result = np.array([[1.5], [2], [3], [4], [4.5]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    def test_denoise_larger_matrix(self):
        im = np.array([[10, 20, 30, 40, 50], 
                       [60, 70, 80, 90, 100], 
                       [110, 120, 130, 140, 150], 
                       [160, 170, 180, 190, 200], 
                       [210, 220, 230, 240, 250]])
        
        expected_result = np.array([[ 40.,  45.,  55.,  65.,  70.],
                                    [65.,  70.,  80.,  90.,  95.],
                                    [115., 120., 130., 140., 145.,],
                                    [165., 170., 180., 190., 195.],
                                    [190., 195., 205., 215., 220.]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    def test_denoise_with_zeros(self):
        im = np.array([[0, 0, 0, 0], 
                       [0, 100, 200, 0], 
                       [0, 0, 0, 0], 
                       [0, 300, 400, 0]])
        
        expected_result = np.array([[100, 150, 150, 200], 
                                    [100, 150, 150, 200], 
                                    [200, 250, 250, 300], 
                                    [300, 350, 350, 400]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    def test_denoise_fail_1D_array(self):
        im = np.array([1, 2, 3, 4, 5])
        
        with self.assertRaises(Exception):
            denoise(im)
        

class TestGeneratedSolution36(BaseTestCase):
    calculate_monthly_sales = imported_functions[35]
    
    def test_multiple_products(self):
        data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-15', '2024-03-01', 
                     '2024-01-03', '2024-01-20', '2024-02-05', '2024-02-25', '2024-03-10'],
            'Product': ['A', 'A', 'A', 'A', 'A', 
                        'B', 'B', 'B', 'B', 'B'],
            'Sales': [100, 150, 200, 250, 300, 
                      120, 130, 140, 150, 160]
        })
        expected = pd.DataFrame({
            'Product': ['A', 'A', 'A', 'B', 'B', 'B'],
            'YearMonth': [pd.Period('2024-01', freq='M'), pd.Period('2024-02', freq='M'), pd.Period('2024-03', freq='M'),
                          pd.Period('2024-01', freq='M'), pd.Period('2024-02', freq='M'), pd.Period('2024-03', freq='M')],
            'Sales': [250, 450, 300, 250, 290, 160],
            'AverageMonthlySales': [333.333333, 333.333333, 333.333333, 233.333333, 233.333333, 233.333333]
        })
        result = calculate_monthly_sales(data)
        pd.testing.assert_frame_equal(result, expected)
    
    def test_single_product_single_month(self):
        data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-15'],
            'Product': ['A', 'A'],
            'Sales': [100, 150]
        })
        expected = pd.DataFrame({
            'Product': ['A'],
            'YearMonth': [pd.Period('2024-01', freq='M')],
            'Sales': [250],
            'AverageMonthlySales': [250.0]
        })
        result = calculate_monthly_sales(data)
        pd.testing.assert_frame_equal(result, expected)
    
    def test_no_sales(self):
        data = pd.DataFrame({
            'Date': [],
            'Product': [],
            'Sales': []
        })
        expected = pd.DataFrame(columns=['Product', 'YearMonth', 'Sales', 'AverageMonthlySales'])
        result = calculate_monthly_sales(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_different_date_formats(self):
        data = pd.DataFrame({
            'Date': ['01-01-2024', '15-01-2024', '01-02-2024', '15-02-2024', '01-03-2024', 
                     '03-01-2024', '20-01-2024', '05-02-2024', '25-02-2024', '10-03-2024'],
            'Product': ['A', 'A', 'A', 'A', 'A', 
                        'B', 'B', 'B', 'B', 'B'],
            'Sales': [100, 150, 200, 250, 300, 
                      120, 130, 140, 150, 160]
        })
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        expected = pd.DataFrame({
            'Product': ['A', 'A', 'A', 'B', 'B', 'B'],
            'YearMonth': [pd.Period('2024-01', freq='M'), pd.Period('2024-02', freq='M'), pd.Period('2024-03', freq='M'),
                          pd.Period('2024-01', freq='M'), pd.Period('2024-02', freq='M'), pd.Period('2024-03', freq='M')],
            'Sales': [250, 450, 300, 250, 290, 160],
            'AverageMonthlySales': [333.333333, 333.333333, 333.333333, 233.333333, 233.333333, 233.333333]
        })
        result = calculate_monthly_sales(data)
        pd.testing.assert_frame_equal(result, expected)

class TestGeneratedSolution37(BaseTestCase):
    recommendations = imported_functions[36]
    
    # Create dataframes for testing
    def setUp(self): 
        self.movies = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'title': ['Inception', 'The Matrix', 'Interstellar', 'Memento', 'Avatar'],
            'rate': [8.8, 8.7, 8.6, 8.4, 7.9],
            'runtime': [148, 136, 169, 113, 162]
        })
        self.movies_genres = pd.DataFrame({
            'movie_id': [1, 2, 3, 4, 5],
            'genre_id': [1, 1, 2, 3, 1]
        })
        self.genres = pd.DataFrame({
            'genre_id': [1, 2, 3],
            'genre_name': ['Sci-Fi', 'Adventure', 'Thriller']
        })
    
    def test_basic_functionality(self):
        search_title = 'Inception'
        expected = pd.DataFrame({
            'id': [2, 5],
            'title': ['The Matrix', 'Avatar'],
            'rate': [8.7, 7.9],
            'runtime': [136, 162]
        })
        result = recommendations(self.movies, self.movies_genres, self.genres, search_title)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_no_matching_titles(self):
        search_title = 'Nonexistent Movie'
        expected = pd.DataFrame(columns=['id', 'title', 'rate', 'runtime'])
        result = recommendations(self.movies, self.movies_genres, self.genres, search_title)
        pd.testing.assert_frame_equal(result, expected)
    
    def test_no_similar_movies(self):
        search_title = 'Interstellar'
        expected = pd.DataFrame(columns=['id', 'title', 'rate', 'runtime'])
        result = recommendations(self.movies, self.movies_genres, self.genres, search_title)
        # Check if the shapes match
        self.assertEqual(result.shape, expected.shape)
        # Check if both DataFrames are empty
        self.assertTrue(result.empty)
        self.assertTrue(expected.empty)

    def test_multiple_matching_movies(self):
        self.movies = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7, 8],
            'title': ['Inception', 'Inception 2', 'Inception 3', 'Inception 4', 'Inception 5', 'Inception 6', 'Inception 7', 'Inception 8'],
            'rate': [8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1],
            'runtime': [148, 148, 148, 148, 148, 148, 148, 148]
        })
        self.movies_genres = pd.DataFrame({
            'movie_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'genre_id': [1, 1, 2, 3, 1, 1, 1, 1]
        })
        search_title = 'Inception'
        result = recommendations(self.movies, self.movies_genres, self.genres, search_title)
        self.assertEqual(len(result), 3)

    def test_case_insensitive_search(self):
        search_title = 'inception'
        expected = pd.DataFrame(columns=['id', 'title', 'rate', 'runtime'])
        result = recommendations(self.movies, self.movies_genres, self.genres, search_title)
        pd.testing.assert_frame_equal(result, expected)

class TestGeneratedSolution38(BaseTestCase):
    top_hours_worked_departments = imported_functions[37]

    def setUp(self):
        self.employees = pd.DataFrame({
            'employee_id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'department_id': [101, 102, 101, 103],
            'salary': [50000, 60000, 55000, 70000]
        })
        self.departments = pd.DataFrame({
            'department_id': [101, 102, 103],
            'name': ['HR', 'Engineering', 'Sales']
        })
        self.works_on = pd.DataFrame({
            'employee_id': [1, 2, 2, 3, 4],
            'project_id': [1, 1, 2, 3, 2],
            'hours_worked': [120, 150, 200, 80, 100]
        })

    def test_basic(self):
        expected = pd.DataFrame({
            'department_name': ['Engineering', 'HR', 'Sales'],
            'total_hours': [350, 200, 100]
        })
        result = top_hours_worked_departments(self.employees, self.departments, self.works_on)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_no_employees(self):
        employees_empty = pd.DataFrame(columns=['employee_id', 'name', 'department_id', 'salary'])
        expected = pd.DataFrame(columns=['department_name', 'total_hours'])
        result = top_hours_worked_departments(employees_empty, self.departments, self.works_on)
        pd.testing.assert_frame_equal(result, expected)

    def test_no_projects(self):
        works_on_empty = pd.DataFrame(columns=['employee_id', 'project_id', 'hours_worked'])
        expected = pd.DataFrame(columns=['department_name', 'total_hours'])
        result = top_hours_worked_departments(self.employees, self.departments, works_on_empty)
        pd.testing.assert_frame_equal(result, expected)

    def test_no_departments(self):
        departments_empty = pd.DataFrame(columns=['department_id', 'name'])
        expected = pd.DataFrame(columns=['department_name', 'total_hours'])
        result = top_hours_worked_departments(self.employees, departments_empty, self.works_on)
        pd.testing.assert_frame_equal(result, expected)

    def test_less_than_three_departments(self):
        employees = pd.DataFrame({
            'employee_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'department_id': [101, 102, 101],
            'salary': [50000, 60000, 55000]
        })
        works_on = pd.DataFrame({
            'employee_id': [1, 2, 2],
            'project_id': [1, 1, 2],
            'hours_worked': [120, 150, 200]
        })
        expected = pd.DataFrame({
            'department_name': ['Engineering', 'HR'],
            'total_hours': [350, 120]
        })
        result = top_hours_worked_departments(employees, self.departments, works_on)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
        
    def test_more_than_three_departments(self):
        employees = pd.DataFrame({
            'employee_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'department_id': [101, 102, 101, 103, 104],
            'salary': [50000, 60000, 55000, 70000, 65000]
        })
        departments = pd.DataFrame({
            'department_id': [101, 102, 103, 104],
            'name': ['HR', 'Engineering', 'Sales', 'Marketing']
        })
        works_on = pd.DataFrame({
            'employee_id': [1, 2, 2, 3, 4, 5],
            'project_id': [1, 1, 2, 3, 2, 4],
            'hours_worked': [120, 150, 200, 80, 100, 90]
        })
        expected = pd.DataFrame({
            'department_name': ['Engineering', 'HR', 'Sales'],
            'total_hours': [350, 200, 100]
        })
        result = top_hours_worked_departments(employees, departments, works_on)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
    
class TestGeneratedSolution39(BaseTestCase):
    huge_population_countries = imported_functions[38]
    
    def setUp(self):
        self.countries = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'population': [1000, 2000, 500, 700, 300]
        })
        self.borders = pd.DataFrame({
            'country1': ['A', 'B', 'B', 'C', 'D', 'E'],
            'country2': ['B', 'A', 'C', 'B', 'E', 'D']
        })

    def test_basic(self):
        expected = pd.DataFrame({
            'name': ['B', 'D'],
            'population': [2000, 700],
            'border_population_sum': [1500, 300]
        })
        result = huge_population_countries(self.countries, self.borders)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_no_countries(self):
        countries = pd.DataFrame(columns=['name', 'population'])
        borders = pd.DataFrame({
            'country1': ['A', 'A', 'B'],
            'country2': ['B', 'C', 'C']
        })
        expected = pd.DataFrame(columns=['name', 'population', 'border_population_sum'])
        result = huge_population_countries(countries, borders)
        pd.testing.assert_frame_equal(result, expected)

    def test_no_borders(self):
        countries = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'population': [1000, 2000, 500]
        })
        borders = pd.DataFrame(columns=['country1', 'country2'])
        expected = pd.DataFrame(columns=['name', 'population', 'border_population_sum'])
        result = huge_population_countries(countries, borders)
        pd.testing.assert_frame_equal(result, expected)

    def test_tie_population(self):
        countries = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'population': [1000, 500, 500]
        })
        borders = pd.DataFrame({
            'country1': ['A', 'A', 'B', 'C'],
            'country2': ['B', 'C', 'A', 'A']
        })
        expected = pd.DataFrame(columns=['name', 'population', 'border_population_sum'])
        result = huge_population_countries(countries, borders)
        self.assertTrue(result.empty and expected.empty)
        self.assertEqual(list(result.columns), list(expected.columns))
    
    def test_no_huge_population_countries(self):
        countries = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'population': [1000, 1000, 1000]
        })
        borders = pd.DataFrame({
            'country1': ['A', 'A', 'B', 'B', 'C', 'C'],
            'country2': ['B', 'C', 'A', 'C', 'A', 'B']
        })
        expected = pd.DataFrame(columns=['name', 'population', 'border_population_sum'])
        result = huge_population_countries(countries, borders)
        self.assertTrue(result.empty and expected.empty)
        self.assertEqual(list(result.columns), list(expected.columns))

class TestGeneratedSolution40(BaseTestCase):
    countries_bordering_most_populated_in_asia = imported_functions[39]
    
    def setUp(self):
        self.country_data = {
            'name': ['China', 'India', 'Japan', 'Pakistan', 'Nepal', 'USA'],
            'capital': ['Beijing', 'New Delhi', 'Tokyo', 'Islamabad', 'Kathmandu', 'Washington D.C.'],
            'continent': ['Asia', 'Asia', 'Asia', 'Asia', 'Asia', 'North America'],
            'population': [1444216107, 1500000000, 126476461, 225199937, 29136808, 331002651]
        }
        self.border_data = {
            'country1': ['China', 'India', 'India', 'India', 'Pakistan', 'Nepal', 'USA', 'China'],
            'country2': ['India', 'Pakistan', 'Nepal', 'China', 'Nepal', 'Pakistan', 'Mexico', 'Mongolia']
        }
        self.country_df = pd.DataFrame(self.country_data)
        self.border_df = pd.DataFrame(self.border_data)

    def test_countries_bordering_most_populated_in_asia(self):
        expected_result = ['China', 'Nepal', 'Pakistan']
        result = countries_bordering_most_populated_in_asia(self.country_df, self.border_df)
        self.assertEqual(result, expected_result)

    def test_no_asian_countries(self):
        country_data = {
            'name': ['USA', 'Canada', 'Mexico'],
            'capital': ['Washington D.C.', 'Ottawa', 'Mexico City'],
            'continent': ['North America', 'North America', 'North America'],
            'population': [331002651, 37742154, 126190788]
        }
        border_data = {
            'country1': ['USA', 'USA', 'Canada'],
            'country2': ['Canada', 'Mexico', 'USA']
        }
        country_df = pd.DataFrame(country_data)
        border_df = pd.DataFrame(border_data)
        expected_result = []
        result = countries_bordering_most_populated_in_asia(country_df, border_df)
        self.assertEqual(result, expected_result)

    def test_single_asian_country(self):
        country_data = {
            'name': ['China'],
            'capital': ['Beijing'],
            'continent': ['Asia'],
            'population': [1444216107]
        }
        border_data = {
            'country1': [],
            'country2': []
        }
        country_df = pd.DataFrame(country_data)
        border_df = pd.DataFrame(border_data)
        expected_result = []
        result = countries_bordering_most_populated_in_asia(country_df, border_df)
        self.assertEqual(result, expected_result)

    def test_multiple_most_populated_countries(self):
        country_data = {
            'name': ['China', 'India', 'Japan', 'Pakistan', 'Nepal', 'Mongolia'],
            'capital': ['Beijing', 'New Delhi', 'Tokyo', 'Islamabad', 'Kathmandu', 'Ulaanbaatar'],
            'continent': ['Asia', 'Asia', 'Asia', 'Asia', 'Asia', 'Asia'],
            'population': [1400000000, 1400000000, 126476461, 225199937, 29136808, 3278290]
        }
        border_data = {
            'country1': ['China', 'India', 'India', 'Pakistan', 'Nepal', 'Mongolia'],
            'country2': ['Mongolia', 'Pakistan', 'Nepal', 'India', 'India', 'China']
        }
        country_df = pd.DataFrame(country_data)
        border_df = pd.DataFrame(border_data)
        expected_result = ['Mongolia', 'Nepal', 'Pakistan']
        result = countries_bordering_most_populated_in_asia(country_df, border_df)
        self.assertEqual(result, expected_result)
        
class TestGeneratedSolution41(BaseTestCase):
    Triangle = imported_classes[0]  

    def setUp(self):
        self.tri = self.Triangle(10, 5, 90, 'black')

    def test_get_given_attributes(self):
        self.assertEqual(self.tri.get('a'), 10)
        self.assertEqual(self.tri.get('b'), 5)
        self.assertEqual(self.tri.get('ab'), 90)
        self.assertEqual(self.tri.get('color'), 'black')

    def test_get_implicit_attributes(self):
        self.assertAlmostEqual(self.tri.get('c'), 11.180339887498949)
        self.assertAlmostEqual(self.tri.get('ac'), 26.565051177077994)
        self.assertAlmostEqual(self.tri.get('bc'), 63.43494882292201)

    def test_get_invalid_edge(self):
        with self.assertRaises(KeyError):
            self.tri.get('d')

    def test_get_invalid_angle(self):
        with self.assertRaises(KeyError):
            self.tri.get('cd')
    
    def test_get_invalid_attribute(self):
        with self.assertRaises(KeyError):
            self.tri.get('hello')

    def test_attribute_sorting(self):
        # Testing if 'ba' gets sorted to 'ab'
        self.assertEqual(self.tri.get('ba'), self.tri.get('ab'))   
        self.assertAlmostEqual(self.tri.get('ca'), 26.565051177077994)   
        self.assertAlmostEqual(self.tri.get('cb'), 63.43494882292201)

class TestGeneratedSolution42(BaseTestCase):
    Worker = imported_classes[1]
    
    def setUp(self):
        # Creating a worker instance for testing
        self.worker = self.Worker('12345', 'Jon', 'Cohen', 'Salesman')
        self.worker_with_second_name = self.Worker('11111', 'David', 'Cohen', 'Math teacher', second_name='Julius')

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

class TestGeneratedSolution43(unittest.TestCase):
    binaric_arithmatic = imported_classes[2]
    
    def setUp(self):
        # Creating an instance for testing
        self.zero = self.binaric_arithmatic("0")
        self.one = self.binaric_arithmatic("1")
        self.seven = self.binaric_arithmatic("111")
        
    def test_inc_positive(self):
        self.assertEqual(self.seven.inc(), "1000")
        self.assertEqual(self.one.inc(), "10")
    
    def test_inc_zero(self):    
        self.assertEqual(self.zero.inc(), "1")

    def test_dec(self):
        self.assertEqual(self.seven.dec(), "110")
        self.assertEqual(self.one.dec(), "0")
    
    def test_get(self):
        self.assertEqual(self.seven.get(), "111")
        self.assertEqual(self.one.get(), "1")
        self.assertEqual(self.zero.get(), "0")

class TestGeneratedSolution44(BaseTestCase):
    Point_2D = imported_classes[3]
    
    def setUp(self):
        # Creating Point_2D instances for testing
        self.a = self.Point_2D(1, 1)
        self.b = self.Point_2D(0, 1)
        self.c = self.Point_2D(-1, 1)
        self.d = self.Point_2D(1, 1)
        self.a_plus_b = self.Point_2D(1, 2)
        self.a_minus_b = self.Point_2D(1, 0)

    def test_repr(self):
        self.assertEqual(repr(self.a), "Point(1, 1)")

    def test_eq(self):
        self.assertTrue(self.a == self.d)
        self.assertFalse(self.a == self.b)

    # def test_add(self):
    #     self.assertEqual(self.a + self.b, self.a_plus_b)

    # def test_sub(self):
    #     self.assertEqual(self.a - self.b, self.a_minus_b)

    def test_distance(self):
        self.assertEqual(self.a.distance(self.c), 2)
        self.assertAlmostEqual(self.a.distance(self.b), math.sqrt(1))

    def test_angle_wrt_origin(self):
        self.assertAlmostEqual(self.b.angle_wrt_origin(self.c), math.pi / 4)
        self.assertAlmostEqual(self.c.angle_wrt_origin(self.b), 2 * math.pi - math.pi / 4)

class TestGeneratedSolution45(BaseTestCase):
    Roulette = imported_classes[4]

    def setUp(self):
        # Create a Roulette instance with an initial balance
        self.gambler = self.Roulette(1000)

    def test_initial_balance(self):
        self.assertEqual(self.gambler.get_balance(), 1000)

    @patch('random.randint')
    def test_bet_red_win(self, mock_randint):
        mock_randint.return_value = 1  # A red number
        self.gambler.bet(100, "red")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    def test_bet_red_lose(self, mock_randint):
        mock_randint.return_value = 2  # A black number
        self.gambler.bet(100, "red")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    def test_bet_black_win(self, mock_randint):
        mock_randint.return_value = 2  # A black number
        self.gambler.bet(100, "black")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    def test_bet_black_lose(self, mock_randint):
        mock_randint.return_value = 1  # A red number
        self.gambler.bet(100, "black")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    def test_bet_even_win(self, mock_randint):
        mock_randint.return_value = 2  # An even number
        self.gambler.bet(100, "even")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    def test_bet_even_lose(self, mock_randint):
        mock_randint.return_value = 1  # An odd number
        self.gambler.bet(100, "even")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    def test_bet_odd_win(self, mock_randint):
        mock_randint.return_value = 1  # An odd number
        self.gambler.bet(100, "odd")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    def test_bet_odd_lose(self, mock_randint):
        mock_randint.return_value = 2  # An even number
        self.gambler.bet(100, "odd")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    def test_bet_range_1_12_win(self, mock_randint):
        mock_randint.return_value = 5  # A number in range 1-12
        self.gambler.bet(100, "1-12")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    def test_bet_range_1_12_lose(self, mock_randint):
        mock_randint.return_value = 13  # A number not in range 1-12
        self.gambler.bet(100, "1-12")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    def test_bet_range_13_24_win(self, mock_randint):
        mock_randint.return_value = 15  # A number in range 13-24
        self.gambler.bet(100, "13-24")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    def test_bet_range_13_24_lose(self, mock_randint):
        mock_randint.return_value = 25  # A number not in range 13-24
        self.gambler.bet(100, "13-24")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    def test_bet_range_25_36_win(self, mock_randint):
        mock_randint.return_value = 30  # A number in range 25-36
        self.gambler.bet(100, "25-36")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    def test_bet_range_25_36_lose(self, mock_randint):
        mock_randint.return_value = 24  # A number not in range 25-36
        self.gambler.bet(100, "25-36")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    def test_bet_specific_number_win(self, mock_randint):
        mock_randint.return_value = 17  # Specific number bet
        self.gambler.bet(100, "17")
        self.assertEqual(self.gambler.get_balance(), 4500)

    @patch('random.randint')
    def test_bet_specific_number_lose(self, mock_randint):
        mock_randint.return_value = 18  # Not the specific number
        self.gambler.bet(100, "17")
        self.assertEqual(self.gambler.get_balance(), 900)

    def test_bet_insufficient_balance(self):
        with self.assertRaises(KeyError):
            self.gambler.bet(2000, "red")





# Custom TestResult class to count failures
class CustomTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_counts = [0] * (len(functions_list) + len(classes_list))
        self.error_counts = [0] * (len(functions_list) + len(classes_list))
        self.total_tests = [0] * (len(functions_list) + len(classes_list))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_case_name = test.__class__.__name__
        for index, case_name in test_cases.items():
            if case_name == test_case_name:
                self.failure_counts[index] += 1
                break
    
    def addError(self, test, err):
        super().addError(test, err)
        test_case_name = test.__class__.__name__
        test_method_name = test._testMethodName
        for index, case_name in test_cases.items():
            if case_name == test_case_name:
                self.error_counts[index] += 1
                print(f"Error in test: {test_case_name}, method: {test_method_name}, error: {err}")
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
            ((self.failure_counts[i] + self.error_counts[i]) / self.total_tests[i]) if self.total_tests[i] > 0 else 0
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
