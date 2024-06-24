import multiprocessing.process
import unittest
import random
import itertools
import threading
import multiprocessing
import time
import sys
import numpy as np
import pandas as pd
import math
from unittest.mock import patch
from scipy.ndimage import median_filter



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
""",
"""class investments: 

    def __init__(self, name, initial_investment, avg_yearly_return, monthly_income, monthly_expenses):
        self.balance = initial_investment
        self.avg_yearly_return = avg_yearly_return
        self.monthly_income = monthly_income
        self.monthly_expenses = monthly_expenses
        self.name = name
    
    def __repr__(self):
        return f"name: {self.name} \\nbalance: {self.balance}\\navg_yearly_return: {self.avg_yearly_return}\\nmonthly_income: {self.monthly_income}\\nmonthly_expenses: {self.monthly_expenses}"
    
    def get_balance(self):
        return self.balance
    
    def get_future_value(self, years):
        future_balance = self.get_balance()
        for i in range(years):
            future_balance = (future_balance + (12 * self.monthly_income - 12 * self.monthly_expenses)) * (1 + self.avg_yearly_return / 100)
        return future_balance
    
    def update_value_by_year(self, years):
        self.balance = self.get_future_value(years)
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise KeyError(f"ERROR: current balance = {self.balance}, can't withdraw {amount}.")
        self.balance -= amount
        return self.balance
""",
"""class Restaurant: 

    def __init__(self, name, cuisine, rating):
        self.name = name
        self.cuisine = cuisine
        self.rating = rating
        self.menu = {}
        self.chefs = []

    def __repr__(self):
        return f"{self.name} ({self.cuisine}) - {self.rating}/5"

    def add_dish(self, name, price):
        self.menu[name] = price
        
    def remove_dish(self, name):
        if name in self.menu:
            del self.menu[name]
        
    def add_chef(self, chef):
        self.chefs.append(chef)
    
    def remove_chef(self, chef):
        if chef in self.chefs:
            self.chefs.remove(chef)
            
    def get_menu(self):
        return self.menu
    
    def get_chefs(self):
        return self.chefs     
""",
"""class Polynomial:     

    def __init__(self, coeffs):
        self.coeffs = coeffs
    
    def __repr__(self):
        res = ""
        if len(self.coeffs) == 1:
            return str(self.coeffs[0])
        if self.coeffs[0] != 0:
            if self.coeffs[1] > 0:
                res += f"{self.coeffs[0]} + {self.coeffs[1]}x"
            elif self.coeffs[1] < 0:
                res += f"{self.coeffs[0]} - {abs(self.coeffs[1])}x"
        if self.coeffs[0] == 0 and self.coeffs[1] != 0:
            res += f"{self.coeffs[1]}x"
        if self.coeffs[0] != 0 and self.coeffs[1] == 0:
            res += f"{self.coeffs[0]}"
        for i in range(2, len(self.coeffs)):
            if self.coeffs[i] > 0:
                res += f" + {self.coeffs[i]}x^{i}"
            elif self.coeffs[i] < 0:
                res += f" - {abs(self.coeffs[i])}x^{i}"
        return res
    
    def get_deg(self):
        return len(self.coeffs) - 1
    
    def __add__(self, other):
        if len(self.coeffs) > len(other.coeffs):
            pad_other = other.coeffs + [0] * (len(self.coeffs) - len(other.coeffs))
            return Polynomial([x + y for x, y in zip(self.coeffs, pad_other)])
        else:
            pad_self = self.coeffs + [0] * (len(other.coeffs) - len(self.coeffs))
            return Polynomial([x + y for x, y in zip(pad_self, other.coeffs)])
    
    def __eq__(self, other):
        return self.coeffs == other.coeffs
""",
"""class TodoList: 

    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        self.tasks.append({'task': task, 'completed': False})

    def remove_task(self, task):
        for t in self.tasks:
            if t['task'] == task:
                self.tasks.remove(t)
                return True
        return False

    def mark_completed(self, task):
        for t in self.tasks:
            if t['task'] == task:
                t['completed'] = True
                return True
        return False

    def list_tasks(self, completed=None):
        if completed is None:
            return [t['task'] for t in self.tasks]
        return [t['task'] for t in self.tasks if t['completed'] == completed]
""",
"""class RecipeBook:
    
    def __init__(self):
        self.recipes = []

    def add_recipe(self, name, ingredients, instructions):
        self.recipes.append({'name': name, 'ingredients': ingredients, 'instructions': instructions})

    def remove_recipe(self, name):
        for recipe in self.recipes:
            if recipe['name'] == name:
                self.recipes.remove(recipe)
                return True
        return False

    def search_by_ingredient(self, ingredient):
        return [recipe for recipe in self.recipes if ingredient in recipe['ingredients']]
"""
]
gemini_func_list = ["""def sum_even(lst):
    total = 0
    for sub_list in lst:
        if isinstance(sub_list, list):
            for i, item in enumerate(sub_list):
                if i % 2 == 0:
                    total += item
    return total
""",
"""def find_num_changes(n, lst):
    if n == 0:
        return 1
    if n < 0 or len(lst) == 0:
        return 0
    # Use the last coin and see how many ways we can make the rest of the amount
    count_with_last_coin = find_num_changes(n - lst[-1], lst)
    # Don't use the last coin and see how many ways we can make the amount with the remaining coins
    count_without_last_coin = find_num_changes(n, lst[:-1])
    return count_with_last_coin + count_without_last_coin""",
"""def sum_nested(lst):
    total = 0
    for element in lst:
        if isinstance(element, list):
            total += sum_nested(element)
        else:
            total += element
    return total""",
"""def str_decomp(target, word_bank):
    memo = {}
    
    def count_ways(remainder):
        if remainder in memo:
            return memo[remainder]
        if remainder == "":
            return 1
        
        total_ways = 0
        for word in word_bank:
            if remainder.startswith(word):
                suffix = remainder[len(word):]
                total_ways += count_ways(suffix)
        
        memo[remainder] = total_ways
        return total_ways
    
    return count_ways(target)""",
"""def n_choose_k(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    coeff = 1
    for i in range(k):
        coeff *= (n - i)
        coeff //= (i + 1)
    return coeff""",
"""def dfs_level_order(tree, index=0):
    if not tree or index >= len(tree) or tree[index] is None:
        return ""
    
    stack, result = [index], []
    
    while stack:
        current_index = stack.pop()
        result.append(str(tree[current_index]))
        
        left_child_index = 2 * current_index + 1
        right_child_index = 2 * current_index + 2
        
        if right_child_index < len(tree) and tree[right_child_index] is not None:
            stack.append(right_child_index)
        
        if left_child_index < len(tree) and tree[left_child_index] is not None:
            stack.append(left_child_index)
            
    return ' '.join(result)""",
"""def half_sum_subset(lst):
    total_sum = sum(lst)
    if total_sum % 2 != 0:
        return None
    target_sum = total_sum // 2
    
    # Initialize a set to store the sums we can achieve with subsets
    possible_sums = {0}
    
    for number in lst:
        # Make a copy of the current possible sums to iterate over
        current_sums = possible_sums.copy()
        for current_sum in current_sums:
            new_sum = current_sum + number
            # If we have found the target sum, reconstruct the subset
            if new_sum == target_sum:
                subset = []
                for x in lst:
                    if x <= number and new_sum - x in possible_sums:
                        subset.append(x)
                        new_sum -= x
                return subset
            possible_sums.add(new_sum)
    
    return None""",
"""def str_dist(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Insert
                                    dp[i][j - 1],      # Remove
                                    dp[i - 1][j - 1])  # Replace

    return dp[m][n]""",
"""def is_dag(graph):
    def has_cycle(v, visited, rec_stack):
        visited[v] = True
        rec_stack[v] = True

        for neighbor in graph[v]:
            if not visited[neighbor]:
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[v] = False
        return False

    visited = [False] * len(graph)
    rec_stack = [False] * len(graph)

    for node in range(len(graph)):
        if not visited[node]:
            if has_cycle(node, visited, rec_stack):
                return False

    return True""",
"""def foo(num, x=0):
    if num < 10:
        return num, x
    else:
        return foo(2/3 * num, x+1)
""",
"""def diff_sparse_matrices(lst):
    if not lst:
        return {}
    
    result = lst[0].copy()
    
    for matrix in lst[1:]:
        for key, value in matrix.items():
            if key in result:
                result[key] -= value
            else:
                result[key] = -value
    
    # Remove zero entries which might have resulted from the subtraction
    result = {k: v for k, v in result.items() if v != 0}
    
    return result""",
"""def longest_subsequence_length(lst):
    if not lst:
        return 0

    def longest_increasing(dp, num):
        low, high = 0, len(dp)
        while low < high:
            mid = (low + high) // 2
            if dp[mid] < num:
                low = mid + 1
            else:
                high = mid
        return low

    increasing = [lst[0]]
    decreasing = [lst[0]]

    for num in lst[1:]:
        if num > increasing[-1]:
            increasing.append(num)
        else:
            idx = longest_increasing(increasing, num)
            increasing[idx] = num

        if num < decreasing[-1]:
            decreasing.append(num)
        else:
            idx = longest_increasing(decreasing, num)
            decreasing[idx] = num

    return max(len(increasing), len(decreasing))""",
"""""",
"""def find_primary_factors(n):
    factors = []
    # Divide by 2 to remove all even factors
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    # Divide by odd numbers starting from 3 to find other prime factors
    divisor = 3
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 2
    # If remaining n is a prime number greater than 2, append it to the factors list
    if n > 2:
        factors.append(n)
    return factors""",
"""def graphs_intersection(g1, g2):
    intersection = {}
    for node in g1:
        if node in g2:
            adjacent_nodes = set(g1[node]) & set(g2[node])
            if adjacent_nodes:
                intersection[node] = list(adjacent_nodes)
    return intersection""",
"""def subset_sum(lst, target):
    def find_subsets(index, current_sum, current_set):
        if current_sum == target:
            result.add(tuple(sorted(current_set)))
            return
        if index == len(lst) or current_sum > target:
            return
        find_subsets(index + 1, current_sum + lst[index], current_set + [lst[index]])
        find_subsets(index + 1, current_sum, current_set)

    result = set()
    find_subsets(0, 0, [])
    return result""",
"""def sum_mult_str(expression):
    elements = expression.split()
    result = 0
    current_multiplication = 1
    for element in elements:
        if element.isdigit():
            current_multiplication *= int(element)
        elif element == "+":
            result += current_multiplication
            current_multiplication = 1
    result += current_multiplication
    return str(result)""",
"""def str_rep(s, k):
    seen_substrings = set()
    for i in range(len(s) - k + 1):
        substring = s[i:i+k]
        if substring in seen_substrings:
            return True
        seen_substrings.add(substring)
    return False""",
"""def sort_two_sorted_lists(lst):
    sorted_list = []
    i = j = 0
    list1, list2 = lst
    
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            sorted_list.append(list1[i])
            i += 1
        else:
            sorted_list.append(list2[j])
            j += 1
            
    while i < len(list1):
        sorted_list.append(list1[i])
        i += 1
        
    while j < len(list2):
        sorted_list.append(list2[j])
        j += 1
        
    return sorted_list""",
"""def prefix_suffix_match(lst, k):
    matching_pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i][:k] == lst[j][:k] and lst[i][-k:] == lst[j][-k:]:
                matching_pairs.append((i, j))
    return matching_pairs""",
"""def rotate_matrix_clockwise(mat):
    if not mat or not mat[0]:
        return mat

    n = len(mat)
    for layer in range(n // 2):
        first, last = layer, n - layer - 1
        for i in range(first, last):
            offset = i - first
            top = mat[first][i]  # save top

            # left -> top
            mat[first][i] = mat[last - offset][first]

            # bottom -> left
            mat[last - offset][first] = mat[last][last - offset]

            # right -> bottom
            mat[last][last - offset] = mat[i][last]

            # top -> right
            mat[i][last] = top

    return mat""",
"""def cyclic_shift(lst, direction, steps):
    if not lst or steps <= 0:
        return lst
    steps = steps % len(lst)
    if direction.upper() == 'L':
        return lst[steps:] + lst[:steps]
    elif direction.upper() == 'R':
        return lst[-steps:] + lst[:-steps]
    else:
        raise ValueError("Direction must be 'L' or 'R'.")""",
"""def encode_string(s):
    encoding = ""
    i = 0

    while i < len(s):
        count = 1
        while i + 1 < len(s) and s[i] == s[i + 1]:
            i += 1
            count += 1
        encoding += str(count) + s[i]
        i += 1

    return encoding""",
"""def list_sums(lst):
    if not lst:  # Handle empty list
        return []
    
    result = [lst[0]]  # Initialize result list with the first element
    for i in range(1, len(lst)):
        result.append(lst[i] + result[i-1])
    return result""",
"""def convert_base(num, base):
    if base < 2 or base > 9 or num < 0:
        return None

    if num == 0:
        return "0"
    
    digits = []
    while num > 0:
        digits.append(str(num % base))
        num //= base

    return ''.join(reversed(digits))""",
"""def max_div_seq(n, k):
    max_count = 0
    current_count = 0
    for digit in str(n):
        if int(digit) % k == 0:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count""",
"""def find_dup(lst):
    tortoise = lst[0]
    hare = lst[0]
    
    # Phase 1: Finding the intersection point of the two runners.
    while True:
        tortoise = lst[tortoise]
        hare = lst[lst[hare]]
        if tortoise == hare:
            break
    
    # Phase 2: Finding the entrance to the cycle (the duplicate).
    tortoise = lst[0]
    while tortoise != hare:
        tortoise = lst[tortoise]
        hare = lst[hare]
    
    return hare""",
"""def lcm(a, b):
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x

    return abs(a * b) // gcd(a, b)""",
"""def f19():
    for number in range(1000, 10000):
        digits = [int(digit) for digit in str(number)]
        product = 1
        for digit in digits:
            product *= digit
        if 55 <= product <= 65:
            return number
    return None""",
"""def f20(numbers):
    if len(numbers) < 3:
        return None
    for i in range(len(numbers) - 2):
        subarray = numbers[i:i + 3]
        if sum(subarray) % 22 == 0:
            return subarray
    return None""",
"""def convolve_1d(signal, kernel):
    kernel = kernel[::-1]
    signal_length = len(signal)
    kernel_length = len(kernel)
    padding = kernel_length - 1
    padded_signal = np.pad(signal, (padding, padding), mode='constant', constant_values=0)
    result = np.zeros(signal_length + padding)
    
    for i in range(signal_length + padding):
        result[i] = np.sum(padded_signal[i:i+kernel_length] * kernel)
    
    return result[:signal_length]""",
"""def mask_n(im, n, idx):
    if not(0 <= idx < n):
        raise ValueError("idx must be in the range of 0 to n-1.")
    
    h, w = im.shape[:2]
    masks = np.zeros((n, h, w), dtype=np.uint8)
    
    for i in range(n):
        masks[i] = (im >= (i / n)) & (im < ((i + 1) / n))
    
    return masks[idx]""",
"""def entropy(mat):
    # Flatten the matrix to a 1D array
    values, counts = np.unique(mat, return_counts=True)
    probabilities = counts / counts.sum()
    # Compute the entropy
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy""",
"""def squeeze_vertical(im, factor):
    if factor <= 0:
        raise ValueError("Factor must be a positive integer.")
    return im[::factor, :]""",
"""def denoise(im):
    denoised_im = median_filter(im, size=3)
    return denoised_im""",
"""def calculate_monthly_sales(data):
    # Extract year and month from 'Date' column
    data['YearMonth'] = data['Date'].dt.to_period('M')
    
    # Group by 'Product' and 'YearMonth' to calculate monthly sales
    monthly_sales = data.groupby(['Product', 'YearMonth'])['Sales'].sum().reset_index()
    
    # Calculate average monthly sales for each product
    average_sales = data.groupby('Product')['Sales'].mean().reset_index()
    average_sales.rename(columns={'Sales': 'AverageMonthlySales'}, inplace=True)
    
    # Merge monthly sales with average monthly sales on 'Product'
    result = pd.merge(monthly_sales, average_sales, on='Product')
    
    return result""",
"""def recommendations(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def top_hours_worked_departments(employees, departments, works_on):
    # Merge the dataframes to associate employees with departments and hours worked
    merged_df = employees.merge(works_on, on='employee ID').merge(departments, on='department ID')
    
    # Calculate the total hours worked per department
    department_hours = merged_df.groupby('department name')['hours worked'].sum().reset_index()
    
    # Sort the departments by total hours worked in descending order
    sorted_departments = department_hours.sort_values('hours worked', ascending=False)
    
    # Select the top 3 departments
    top_departments = sorted_departments.head(3)
    
    return top_departments""",
"""def huge_population_countries(countries, borders):
    # Create a dictionary to hold the sum of population of border countries for each country
    border_population_sum = {country: 0 for country in countries['country']}
    
    # Calculate the sum of population of the border countries
    for _, row in borders.iterrows():
        country1, country2 = row['country1'], row['country2']
        if country2 in border_population_sum:
            border_population_sum[country2] += countries.loc[countries['country'] == country1, 'population'].values[0]
        if country1 in border_population_sum:
            border_population_sum[country1] += countries.loc[countries['country'] == country2, 'population'].values[0]
    
    # Find countries with a population larger than the sum of the population of its border countries
    result_countries = []
    for country, population in countries.itertuples(index=False):
        if population > border_population_sum[country]:
            result_countries.append(country)
    
    # Create a DataFrame with the list of countries with a larger population than their borders
    result_df = pd.DataFrame(result_countries, columns=['country'])
    
    return result_df""",
"""import pandas as pd

def countries_bordering_most_populated_in_asia(country_df, border_df):
    # Filter the DataFrame to get only Asian countries
    asian_countries = country_df[country_df['continent'] == 'Asia']

    # Assuming the DataFrame has a column 'population' to find the most populated country
    most_populated_asian_country = asian_countries.nlargest(1, 'population')['name'].iloc[0]

    # Filter the DataFrame to get borders of the most populated Asian country
    borders = border_df[(border_df['country1'] == most_populated_asian_country) | 
                        (border_df['country2'] == most_populated_asian_country)]

    # Get the list of bordering countries
    bordering_countries = borders['country1'].tolist() + borders['country2'].tolist()

    # Remove the most populated country from the list, if present
    bordering_countries = [country for country in bordering_countries if country != most_populated_asian_country]

    return sorted(bordering_countries)"""
]
gemini_classes_list = ["""class Triangle:
    def __init__(self, a, b, ab, color):
        self.a = a
        self.b = b
        self.ab = ab
        self.color = color
        # Additional attributes based on the law of sines
        self.c = None
        self.bc = None
        self.ac = None
        
    def get(self, name):
        return getattr(self, name, None)""",
"""class Worker:
    def __init__(self, id, first_name, last_name, job, salary=5000, second_name=None):
        if second_name is not None:
            self.full_name = f"{first_name} {second_name} {last_name}"
        else:
            self.full_name = f"{first_name} {last_name}"
        self.id = id
        self.job = job
        self.salary = salary

    def getFullName(self):
        return self.full_name

    def getSalary(self):
        return self.salary

    def getJob(self):
        return self.job

    def update(self, job=None, salary=None):
        if job is not  None:
            self.job = job
        if salary is not None:
            self.salary = salary""",
"""class Binaric_arithmetic:
    def __init__(self, num):
        self.num = num

    def get(self):
        return self.num

    def inc(self):
        self.num = bin(int(self.num, 2) + 1)[2:]

    def dec(self):
        if self.num != '0':
            self.num = bin(int(self.num, 2) - 1)[2:]""",
"""import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = math.sqrt(x**2 + y**2)
        self.theta = math.atan2(y, x)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        raise ValueError("Operand must be an instance of Point")

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        raise ValueError("Operand must be an instance of Point")

    def distance(self, other):
        if isinstance(other, Point):
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        raise ValueError("Argument must be an instance of Point")

    def angle_wrt_origin(self):
        return math.atan2(self.y, self.x)""",
"""class Roulette:
    def __init__(self, initial_money):
        self.balance = initial_money
        self.reds = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
        self.blacks = [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]

    def get_balance(self):
        return self.balance

    def bet(self, amount, bet_type):
        if amount > self.balance:
            raise KeyError('Bet amount exceeds current balance.')
        roll = random.randint(0, 36)
        win = False

        if bet_type == "red" and roll in self.reds:
            win = True
        elif bet_type == "black" and roll in self.blacks:
            win = True
        elif bet_type == "even" and roll % 2 == 0 and roll != 0:
            win = True
        elif bet_type == "odd" and roll % 2 == 1:
            win = True
        elif bet_type == "1-12" and 1 <= roll <= 12:
            win = True
        elif bet_type == "13-24" and 13 <= roll <= 24:
            win = True
        elif bet_type == "25-36" and 25 <= roll <= 36:
            win = True
        elif type(bet_type) == int and bet_type == roll:
            win = True

        if win:
            if type(bet_type) == int:
                self.balance += amount * 35  # Specific number pays 35 to 1
            else:
                self.balance += amount  # Other bets pay 1 to 1
        else:
            self.balance -= amount

        return win""",
"""class Investment:
    def __init__(self, name, initial_investment, avg_yearly_return, monthly_income, monthly_expenses):
        self.name = name
        self.balance = initial_investment
        self.avg_yearly_return = avg_yearly_return
        self.monthly_income = monthly_income
        self.monthly_expenses = monthly_expenses

    def __repr__(self):
        return f"Investment('{self.name}', {self.balance}, {self.avg_yearly_return}, {self.monthly_income}, {self.monthly_expenses})"

    def get_balance(self):
        return self.balance

    def get_future_value(self, years):
        future_value = self.balance * ((1 + self.avg_yearly_return) ** years)
        return future_value

    def update_value_by_year(self, years):
        self.balance = self.get_future_value(years)

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            raise ValueError("Insufficient funds for withdrawal.")""",
"""class Restaurant:
    def __init__(self, name, cuisine, rating):
        self.name = name
        self.cuisine = cuisine
        self.rating = rating
        self.menu = {}
        self.chefs = []

    def __repr__(self):
        return f'Restaurant({self.name}, {self.cuisine}, Rating: {self.rating}/5)'

    def add_dish(self, name, price):
        self.menu[name] = price

    def remove_dish(self, name):
        if name in self.menu:
            del self.menu[name]

    def add_chef(self, chef):
        self.chefs.append(chef)

    def remove_chef(self, chef):

        if chef in self.chefs:
            self.chefs.remove(chef)

    def get_menu(self):
        return self.menu

    def get_chefs(self):
        return self.chefs""",
"""class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __repr__(self):
        poly_str = " + ".join(f"{coef}x^{i}" if i > 0 else str(coef) 
                              for i, coef in enumerate(self.coeffs) if coef)
        return poly_str.replace("x^1 ", "x ").replace(" 1x", " x")

    def get_deg(self):
        return len(self.coeffs) - 1

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented

        new_coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
        # Handle the case when polynomials are of different lengths
        if len(self.coeffs) > len(other.coeffs):
            new_coeffs.extend(self.coeffs[len(other.coeffs):])
        else:
            new_coeffs.extend(other.coeffs[len(self.coeffs):])
        return Polynomial(new_coeffs)

    def __eq__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        return self.coeffs == other.coeffs""",
"""class TodoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append({'task': task, 'completed': False})

    def remove_task(self, task):
        self.tasks = [t for t in self.tasks if t['task'] != task]

    def mark_completed(self, task):
        for t in self.tasks:
            if t['task'] == task:
                t['completed'] = True
                break

    def list_tasks(self, completed=None):
        if completed is None:
            return self.tasks
        return [t for t in self.tasks if t['completed'] == completed]""",
"""class RecipeBook:
    def __init__(self):
        self.recipes = []

    def add_recipe(self, name, ingredients, instructions):
        recipe = {'name': name, 'ingredients': ingredients, 'instructions': instructions}
        self.recipes.append(recipe)

    def remove_recipe(self, name):
        self.recipes = [recipe for recipe in self.recipes if recipe['name'] != name]

    def search_by_ingredient(self, ingredient):
        return [recipe for recipe in self.recipes if ingredient in recipe['ingredients']]"""
]
gpt_func_list = ["""def sum_even(lst):
    def is_even(num):
        return num % 2 == 0

    def sum_even_recursive(sublist):
        total = 0
        for item in sublist:
            if isinstance(item, list):
                total += sum_even_recursive(item)
            elif is_even(item):
                total += item
        return total

    return sum_even_recursive(lst)""",
"""def find_num_changes(n, lst):
    ways = [0] * (n + 1)
    ways[0] = 1
    for num in lst:
        for i in range(num, n + 1):
            ways[i] += ways[i - num]
    return ways[n]""",
"""def sum_nested(lst):
    total = 0.0
    for item in lst:
        if isinstance(item, list):
            total += sum_nested(item)
        else:
            total += abs(float(item))
    return total""",
"""def str_decomp(target, word_bank, memo=None):
    if memo is None:
        memo = {}
    if target in memo:
        return memo[target]
    if target == '':
        return 1
    
    total_count = 0
    
    for word in word_bank:
        if target.startswith(word):
            suffix = target[len(word):]
            num_ways_for_rest = str_decomp(suffix, word_bank, memo)
            total_count += num_ways_for_rest
    
    memo[target] = total_count
    return total_count""",
"""def n_choose_k(n, k):
    if k == 0 or k == n:
        return 1
    return n_choose_k(n - 1, k - 1) + n_choose_k(n - 1, k)""",
"""def dfs_level_order(tree, index=0):
    if index >= len(tree) or tree[index] is None:
        return ""
    left = dfs_level_order(tree, 2 * index + 1)
    right = dfs_level_order(tree, 2 * index + 2)
    return f"{tree[index]},{left}{right}".strip(',')""",
"""def half_sum_subset(lst):
    total_sum = sum(lst)
    # The target sum we are looking for is half of the total sum
    target_sum = total_sum // 2
    # The sum needs to be even to find a subset with exactly half the sum
    if total_sum % 2 != 0:
        return None

    n = len(lst)
    # Using dynamic programming to check if a subset with sum equal to target_sum exists
    dp = [[False for _ in range(target_sum + 1)] for _ in range(n + 1)]

    # Initializing dp: with 0 items, we can only achieve sum 0
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the dp table
    for i in range(1, n + 1):
        for j in range(1, target_sum + 1):
            if lst[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - lst[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    # If there's no subset with sum equal to target_sum, return None
    if not dp[n][target_sum]:
        return None

    # Backtrack to find the elements in the subset
    subset = []
    i, j = n, target_sum
    while i > 0 and j > 0:
        # If the value comes from the top (excluding the current element), move up
        if dp[i - 1][j]:
            i -= 1
        else:
            # Include the current element in the subset and move diagonally up
            subset.append(lst[i - 1])
            j -= lst[i - 1]
            i -= 1

    return subset""",
"""def str_dist(x, y):
    def recurse(i, j):
        if i == 0:  # Base case: first string is empty
            return j
        if j == 0:  # Base case: second string is empty
            return i
        if x[i - 1] == y[j - 1]:  # Characters match
            return recurse(i - 1, j - 1)
        # Characters don't match, compute costs of operations
        insert = recurse(i, j - 1)
        delete = recurse(i - 1, j)
        replace = recurse(i - 1, j - 1)
        return 1 + min(insert, delete, replace)

    return recurse(len(x), len(y))""",
"""def is_dag(graph):
    def visit(node):
        if node in temp_mark:
            return False
        if node in perm_mark:
            return True
        temp_mark.add(node)
        for neighbour in graph.get(node, []):
            if not visit(neighbour):
                return False
        temp_mark.remove(node)
        perm_mark.add(node)
        return True

    temp_mark = set()
    perm_mark = set()
    for node in graph:
        if node not in perm_mark:
            if not visit(node):
                return False
    return True""",
"""def foo(num, x):
    # Base condition to terminate the recursion
    if x == 0:
        return num, x

    # Modifying 'num' and decrementing 'x' as per some condition(s)
    # As the conditions are not specified, an example could be:
    # - If 'num' is even, divide it by 2
    # - If 'num' is odd, multiply it by 3 and add 1
    # Decrement 'x' after each operation
    if num % 2 == 0:
        num = num // 2
    else:
        num = 3 * num + 1

    # Recur with the new value of 'num' and decremented 'x'
    return foo(num, x - 1)""",
"""def diff_sparse_matrices(lst):
    if not lst:
        return {}

    # Initialize the result with a copy of the first matrix
    result = lst[0].copy()

    # Iterate over the remaining matrices
    for matrix in lst[1:]:
        for key, value in matrix.items():
            if key in result:
                # Subtract the value from the corresponding element in the result
                result[key] -= value
            else:
                # If the key is not in the result, it means we have to subtract from 0
                result[key] = -value

    # Remove zero entries from the result
    result = {key: value for key, value in result.items() if value != 0}

    return result""",
"""from bisect import bisect_left

def longest_subsequence_length(lst):
    if not lst:
        return 0

    def longest_increasing_subsequence_length(seq):
        tails = []
        for num in seq:
            index = bisect_left(tails, num)
            if index == len(tails):
                tails.append(num)
            else:
                tails[index] = num
        return len(tails)

    def longest_decreasing_subsequence_length(seq):
        return longest_increasing_subsequence_length([-x for x in seq])

    return max(longest_increasing_subsequence_length(lst), longest_decreasing_subsequence_length(lst))
""",
"""""",
"""def find_primary_factors(n):
    factors = []
    divisor = 2
    while n >= 2:
        if n % divisor == 0:
            factors.append(divisor)
            n = n // divisor
        else:
            divisor += 1
    return factors""",
"""def graphs_intersection(g1, g2):
    intersection = {}
    for node in g1:
        if node in g2:
            common_adjacent = set(g1[node]) & set(g2[node])
            if common_adjacent:
                intersection[node] = list(common_adjacent)
    return intersection""",
"""def subset_sum(lst, target):
    result = set()
    subsets = [()]

    for number in lst:
        new_subsets = []
        for existing_subset in subsets:
            new_subset = existing_subnet + (number,)
            new_subsets.append(new_subset)
            if sum(new_subset) == target:
                result.add(new_subset)
        subsets.extend(new_subsets)

    return result""",
"""def sum_mult_str(expression):
    tokens = expression.split(' ')
    result = 0
    current = 1
    operation = '+'

    for token in tokens:
        if token.isdigit():
            if operation == '+':
                result += current
                current = int(token)
            elif operation == '*':
                current *= int(token)
        elif token in '+*':
            operation = token

    result += current
    return str(result)""",
"""def str_rep(s, k):
    seen_substrings = set()
    for i in range(len(s) - k + 1):
        substring = s[i:i+k]
        if substring in seen_substrings:
            return True
        seen_substrings.add(substring)
    return False""",
"""def sort_two_sorted_lists(lst):
    even = []
    odd = []
    
    # Separate even and odd numbers
    for num in lst:
        if num % 2 == 0:
            even.append(num)
        else:
            odd.append(num)
    
    # Merge the even and odd lists
    sorted_list = []
    i = j = 0
    while i < len(even) and j < len(odd):
        if even[i] <= odd[j]:
            sorted_list.append(even[i])
            i += 1
        else:
            sorted_list.append(odd[j])
            j += 1
            
    # Append any remaining elements
    sorted_list.extend(even[i:])
    sorted_list.extend(odd[j:])
    
    return sorted_list""",
"""def prefix_suffix_match(lst, k):
    matching_pairs = []
    for i, item in enumerate(lst):
        for j, other_item in enumerate(lst):
            if i != j and item.endswith(other_item[:k]):
                matching_pairs.append((i, j))
    return matching_pairs""",
"""def rotate_matrix_clockwise(mat):
    n = len(mat)
    for layer in range(n // 2):
        first = layer
        last = n - layer - 1
        for i in range(first, last):
            offset = i - first
            top = mat[first][i]
            mat[first][i] = mat[last - offset][first]
            mat[last - offset][first] = mat[last][last - offset]
            mat[last][last - offset] = mat[i][last]
            mat[i][last] = top
    return mat""",
"""def cyclic_shift(lst, direction, steps):
    if direction not in ('L', 'R'):
        raise ValueError("Direction must be 'L' or 'R'")
    steps = steps % len(lst)  # Ensure steps is within the range of list length
    if direction == 'L':
        lst = lst[steps:] + lst[:steps]
    else:  # direction == 'R'
        lst = lst[-steps:] + lst[:-steps]
    return lst""",
"""def encode_string(s):
    if not s:
        return ""

    encoded_str = ""
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            encoded_str += f"[{count}]{s[i-1]}"
            count = 1
    encoded_str += f"[{count}]{s[-1]}"
    
    return encoded_str""",
"""def list_sums(lst):
    for i in range(1, len(lst)):
        lst[i] += lst[i - 1]""",
"""def convert_base(num, base):
    if base < 1 or base > 9 or num < 0:
        return None
    print(f"num: {num}, base: {base}")
    converted_num = ''
    while num > 0:
        converted_num = str(num % base) + converted_num
        num //= base
    return converted_num""",
"""def max_div_seq(n, k):
    max_count = 0
    current_count = 0

    for digit in str(n):
        if int(digit) % k == 0:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count""",
"""def find_dup(lst):
    tortoise = lst[0]
    hare = lst[0]

    # Phase 1: Finding the intersection point of the two runners.
    while True:
        tortoise = lst[tortoise]
        hare = lst[lst[hare]]
        if tortoise == hare:
            break

    # Phase 2: Finding the entrance to the cycle (duplicate element).
    tortoise = lst[0]
    while tortoise != hare:
        tortoise = lst[tortoise]
        hare = lst[hare]

    return hare""",
"""""",
"""def f19():
    for number in range(1000, 10000):
        digits = [int(digit) for digit in str(number)]
        product_of_digits = 1
        for digit in digits:
            product_of_digits *= digit
        if 55 < product_of_digits < 65 and product_of_digits % 15 == 0:
            return number
    return None""",
"""def f20(number):
    num_str = str(number)
    for i in range(len(num_str)):
        for j in range(i + 1, len(num_str)):
            for k in range(j + 1, len(num_str)):
                reduced_num = int(num_str[:i] + num_str[i+1:j] + num_str[j+1:k] + num_str[k+1:])
                if reduced_num % 22 == 0:
                    return reduced_num
    return None""",
"""def convolve_1d(signal, kernel):
    # Determine the size of the output array
    output_size = len(signal) + len(kernel) - 1
    # Pad the signal with zeros on both sides for the convolution
    padded_signal = np.pad(signal, (len(kernel)-1, len(kernel)-1), mode='constant', constant_values=0)
    # Initialize the output array
    result = np.zeros(output_size)
    
    # Perform the 1D convolution
    for i in range(output_size):
        result[i] = np.sum(padded_signal[i:i+len(kernel)] * kernel[::-1])
        
    return result""",
"""import numpy as np

def mask_n(im, n, idx):
    division_size = im.shape[0] // n
    start = idx * division_size
    if idx == n - 1:  # Last division might have more elements in case of an uneven split
        end = im.shape[0]
    else:
        end = start + division_size
    mask = np.zeros_like(im, dtype=bool)
    mask[start:end] = True
    return mask""",
"""import numpy as np

def entropy(mat):
    _, counts = np.unique(mat, return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy""",
"""import numpy as np

def squeeze_vertical(im, factor):
    # Calculate the size of the output image
    out_height = im.shape[0] // factor
    out_width = im.shape[1]
    
    # Initialize the squeezed image with zeros
    squeezed_image = np.zeros((out_height, out_width), dtype=im.dtype)
    
    # Perform the vertical squeeze by averaging blocks of pixels
    for i in range(out_height):
        start_row = i * factor
        end_row = start_row + factor
        squeezed_image[i, :] = im[start_row:end_row, :].mean(axis=0)
    
    return squeezed_image""",
"""import numpy as np
from scipy.ndimage import median_filter

def denoise(im):
    return median_filter(im, size=3)""",
"""import pandas as pd

def calculate_monthly_sales(data):
    # Convert 'Date' to datetime and create 'YearMonth' column
    data['Date'] = pd.to_datetime(data['Date'])
    data['YearMonth'] = data['Date'].dt.to_period('M')

    # Group by Product and YearMonth to calculate monthly sales
    monthly_sales = data.groupby(['Product', 'YearMonth']).agg(Sales=('Sales', 'sum')).reset_index()

    # Calculate average monthly sales for each product
    average_sales = monthly_sales.groupby('Product')['Sales'].mean().reset_index()
    average_sales.rename(columns={'Sales': 'AverageMonthlySales'}, inplace=True)

    # Merge monthly sales with average monthly sales
    result = pd.merge(monthly_sales, average_sales, on='Product')

    # Reorder columns to match the desired output
    result = result[['Product', 'YearMonth', 'Sales', 'AverageMonthlySales']]

    return result""",
"""def recommendations(movies, movies_genres, genres, search_title):
    # Find the movie that matches the search_title
    base_movie = movies[movies['title'].str.contains(search_title, case=False, na=False)]
    
    # If the movie is not found, return an empty DataFrame
    if base_movie.empty:
        return pd.DataFrame()
    
    # Extract the base movie's information
    base_movie_id = base_movie.iloc[0]['id']
    base_movie_rate = base_movie.iloc[0]['rate']
    base_movie_runtime = base_movie.iloc[0]['runtime']
    
    # Find the genres of the base movie
    base_movie_genres = movies_genres[movies_genres['movie_id'] == base_movie_id]['genre_id'].tolist()
    
    # Filter movies by matching genres
    matching_genre_ids = movies_genres[movies_genres['genre_id'].isin(base_movie_genres)]['movie_id']
    genre_matched_movies = movies[movies['id'].isin(matching_genre_ids)]
    
    # Filter movies by similar rate and runtime
    recommendations = genre_matched_movies[
        (genre_matched_movies['rate'] == base_movie_rate) &
        (abs(genre_matched_movies['runtime'] - base_movie_runtime) <= 10)
    ]
    
    # Exclude the base movie from recommendations
    recommendations = recommendations[recommendations['id'] != base_movie_id]
    
    # Sort by rate and runtime difference (ascending) and return up to 3 recommendations
    recommendations = recommendations.sort_values(by=['rate', 'runtime'], ascending=[False, True]).head(3)
    
    return recommendations""",
"""def top_hours_worked_departments(employees, departments, works_on):
    if employees.empty or departments.empty or works_on.empty:
        return pd.DataFrame(columns=['department_name', 'total_hours'])

    # Merge the DataFrames to associate employee hours with their respective departments
    merged_df = pd.merge(employees, works_on, on='employee_id')
    merged_df = pd.merge(merged_df, departments, left_on='department_id', right_on='department_id')
    
    # Calculate total hours worked by department
    total_hours_by_department = merged_df.groupby('name')['hours_worked'].sum().reset_index(name='total_hours')
    
    # Sort by total hours in descending order and get the top 3 departments
    top_departments = total_hours_by_department.sort_values(by='total_hours', ascending=False).head(3)
    
    # Rename the column for clarity
    top_departments.rename(columns={'name': 'department_name'}, inplace=True)
    
    return top_departments""",
"""def huge_population_countries(countries_df, borders_df):
    # Merge the two DataFrames on the country name
    merged_df = countries_df.merge(borders_df, left_on='country', right_on='border_country')
    
    # Group by the original country and sum the populations of the bordering countries
    border_populations = merged_df.groupby('country')['border_population'].sum().reset_index()
    
    # Merge the sum of border populations back with the original countries DataFrame
    countries_with_border_populations = countries_df.merge(border_populations, on='country', how='left')
    
    # Filter out countries with a population greater than the sum of their bordering countries' populations
    filtered_countries = countries_with_border_populations[
        countries_with_border_populations['population'] <= countries_with_border_populations['border_population']
    ]
    
    # Select only the required columns
    result_df = filtered_countries[['country', 'population', 'border_population']]
    
    return result_df""",
"""def countries_bordering_most_populated_in_asia(country_df, border_df):
    # Filter the DataFrame to get only Asian countries
    asian_countries = country_df[country_df['continent'] == 'Asia']
    # Find the most populated country in Asia
    most_populated_asian_country = asian_countries.loc[asian_countries['population'].idxmax()]['country']
    # Get a list of countries that share borders with the most populated Asian country
    bordering_countries = border_df[border_df['country1'] == most_populated_asian_country]['country2'].tolist()
    # Sort the list of bordering countries
    bordering_countries_sorted = sorted(bordering_countries)
    return bordering_countries_sorted"""
]
gpt_classes_list = ["""import math

class triangle:
    def __init__(self, a, b, ab, color):
        self.a = a
        self.b = b
        self.ab = ab
        self.color = color
        self._calculate_sides_and_angles()

    def _calculate_sides_and_angles(self):
        # Calculate side 'c' using the Law of Cosines
        self.c = math.sqrt(self.a**2 + self.b**2 - 2 * self.a * self.b * math.cos(math.radians(self.ab)))

        # Calculate the remaining angles using the Law of Sines
        self.ac = math.degrees(math.asin(self.a * math.sin(math.radians(self.ab)) / self.c))
        self.bc = 180 - self.ab - self.ac

    def get(self, name):
        if not hasattr(self, name):
            raise KeyError(f"Attribute {name} does not exist in the triangle.")
        return getattr(self, name)""",
"""class worker:
    def __init__(self, id, first_name, last_name, job, salary=5000, second_name=None):
        self.id = id
        self.full_name = f"{first_name} {second_name + ' ' if second_name else ''}{last_name}"
        self.job = job
        self.salary = salary

    def getFullName(self):
        return self.full_name

    def getSalary(self):
        return self.salary

    def getJob(self):
        return self.job

    def update(self, job=None, salary=None):
        if job is not None:
            self.job = job
        if salary is not None:
            self.salary = salary""",
"""class Binaric_arithmatic:

    def __init__(self, num):
        self.num = num

    def get(self):
        return self.num

    def inc(self):
        self.num = bin(int(self.num, 2) + 1)[2:]

    def dec(self):
        if self.num != '0':
            self.num = bin(int(self.num, 2) - 1)[2:]""",
"""import math

class Point_2D:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.r = math.sqrt(x**2 + y**2)
        self.theta = math.atan2(y, x)
    
    def __repr__(self):
        return f"Point_2D(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        return Point_2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point_2D(self.x - other.x, self.y - other.y)
    
    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_wrt_origin(self, other):
        delta_x = other.x - self.x
        delta_y = other.y - self.y
        return math.atan2(delta_y, delta_x)""",
"""class Roulette:
    def __init__(self, initial_money):
        self.balance = initial_money
        self.reds = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
        self.blacks = [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]

    def get_balance(self):
        return self.balance

    def bet(self, amount, bet_type):
        # This is a simplified version of the bet method, not considering the actual rules or odds of roulette.
        if amount > self.balance:
            print("Insufficient funds to place bet.")
            return

        # Assume bet_type is either 'red' or 'black' for this example
        if bet_type == 'red':
            # Implement the game logic to determine win or loss on red
            pass
        elif bet_type == 'black':
            # Implement the game logic to determine win or loss on black
            pass

        # Update balance depending on whether the player won or lost the bet
        # This is placeholder logic for the outcome of the bet
        win = False  # Replace with actual game logic
        if win:
            self.balance += amount  # Player wins the bet
        else:
            self.balance -= amount  # Player loses the bet""",
"""class investments:
    def __init__(self, name, initial_investment, avg_yearly_return, monthly_income, monthly_expenses):
        self.name = name
        self.balance = initial_investment
        self.avg_yearly_return = avg_yearly_return
        self.monthly_income = monthly_income
        self.monthly_expenses = monthly_expenses

    def __repr__(self):
        return (f"Investment('{self.name}', {self.balance}, {self.avg_yearly_return}, "
                f"{self.monthly_income}, {self.monthly_expenses})")

    def get_balance(self):
        return self.balance

    def get_future_value(self, years):
        future_value = self.balance
        for _ in range(years * 12):  # Compounding monthly
            future_value += future_value * (self.avg_yearly_return / 100) / 12
            future_value += self.monthly_income
            future_value -= self.monthly_expenses
        return future_value

    def update_value_by_year(self, years):
        self.balance = self.get_future_value(years)

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            raise ValueError("Insufficient funds for withdrawal")""",
"""class Restaurant:
    def __init__(self, name, cuisine, rating):
        self.name = name
        self.cuisine = cuisine
        self.rating = rating
        self.menu = {}
        self.chefs = []

    def __repr__(self):
        return f"Restaurant({self.name}, {self.cuisine}, {self.rating})"

    def add_dish(self, name, price):
        self.menu[name] = price

    def remove_dish(self, name):
        if name in self.menu:
            del self.menu[name]

    def add_chef(self, chef):
        self.chefs.append(chef)

    def remove_chef(self, chef):
        if chef in self.chefs:
            self.chefs.remove(chef)

    def get_menu(self):
        return self.menu

    def get_chefs(self):
        return self.chefs""",
"""class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __repr__(self):
        terms = []
        for i, coeff in enumerate(self.coeffs[::-1]):
            if coeff != 0:
                term = f"{'' if coeff == 1 else coeff}"
                if i == 0:
                    terms.append(f"{term}")
                elif i == 1:
                    terms.append(f"{term}x")
                else:
                    terms.append(f"{term}x^{i}")
        return " + ".join(terms[::-1]) if terms else "0"

    def get_deg(self):
        return len(self.coeffs) - 1

    def __add__(self, other):
        max_deg = max(self.get_deg(), other.get_deg())
        result_coeffs = [0] * (max_deg + 1)
        for i in range(max_deg + 1):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            result_coeffs[i] = a + b
        return Polynomial(result_coeffs)

    def __eq__(self, other):
        return self.coeffs == other.coeffs""",
"""class TodoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append({'task': task, 'completed': False})

    def remove_task(self, task):
        self.tasks = [t for t in self.tasks if t['task'] != task]

    def mark_completed(self, task):
        for t in self.tasks:
            if t['task'] == task:
                t['completed'] = True
                break

    def list_tasks(self, completed=None):
        if completed is None:
            return self.tasks
        else:
            return [t for t in self.tasks if t['completed'] == completed]""",
"""class RecipeBook:
    def __init__(self):
        self.recipes = []

    def add_recipe(self, name, ingredients, instructions):
        new_recipe = {
            'name': name,
            'ingredients': ingredients,
            'instructions': instructions
        }
        self.recipes.append(new_recipe)

    def remove_recipe(self, name):
        self.recipes = [recipe for recipe in self.recipes if recipe['name'] != name]

    def search_by_ingredient(self, ingredient):
        return [recipe for recipe in self.recipes if ingredient in recipe['ingredients']]"""
]
claude_func_list = ["""def sum_even(lst):
    def sum_nested_list(nested_lst):
        return sum(sum_nested_list(item) if isinstance(item, list) else item for item in nested_lst)

    total_sum = 0
    for i, item in enumerate(lst):
        if i % 2 == 0:
            if isinstance(item, list):
                total_sum += sum_nested_list(item)
            else:
                total_sum += item
    return total_sum""",
"""def find_num_changes(n, lst):
    combinations = [0] * (n + 1)
    combinations[0] = 1
    
    for coin in lst:
        for amount in range(coin, n + 1):
            combinations[amount] += combinations[amount - coin]
    
    return combinations[n]""",
"""def sum_nested(lst):
    def flatten_and_convert(item):
        if isinstance(item, list):
            return sum(flatten_and_convert(subitem) for subitem in item)
        elif isinstance(item, (int, float, str)):
            return abs(float(item))
        else:
            raise ValueError("Unsupported data type in the nested structure.")

    return flatten_and_convert(lst)""",
"""def str_decomp(target, word_bank):
    memo = {}
    
    def decompose(remaining):
        if remaining in memo:
            return memo[remaining]
        if remaining == "":
            return 1
        
        total_ways = 0
        for word in word_bank:
            if remaining.startswith(word):
                suffix = remaining[len(word):]
                total_ways += decompose(suffix)
        
        memo[remaining] = total_ways
        return total_ways
    
    return decompose(target)""",
"""def n_choose_k(n, k):
    def factorial(num):
        if num == 0:
            return 1
        else:
            return num * factorial(num - 1)

    return factorial(n) // (factorial(k) * factorial(n - k))""",
"""from collections import deque

def dfs_level_order(tree, index=0):
    if not tree:
        return ""

    # Helper function to perform DFS traversal
    def dfs(node_index):
        if node_index >= len(tree) or tree[node_index] is None:
            return []
        left_index = 2 * node_index + 1
        right_index = 2 * node_index + 2
        return [tree[node_index]] + dfs(left_index) + dfs(right_index)

    # Perform DFS traversal to get nodes in DFS order
    dfs_result = dfs(index)

    # Prepare a dictionary to hold nodes at each depth level
    level_dict = {}
    for position, value in enumerate(dfs_result):
        level = int(math.log2(position + 1))
        if level not in level_dict:
            level_dict[level] = []
        level_dict[level].append(value)

    # Flatten the dictionary values to get level-order traversal
    level_order = []
    for level in sorted(level_dict.keys()):
        level_order.extend(level_dict[level])

    # Convert nodes list to a comma separated string
    return ', '.join(map(str, level_order))""",
"""def half_sum_subset(lst):
    total_sum = sum(lst)
    half_sum = total_sum // 2
    if total_sum % 2 != 0:  # If total sum is odd, cannot find subset summing to half
        return None

    n = len(lst)
    # Create a DP table to store results of subproblems
    dp = [[False for _ in range(half_sum + 1)] for _ in range(n + 1)]

    # Initialize first column as true, as zero sum is possible with all subsets
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the partition table in a bottom-up manner
    for i in range(1, n + 1):
        for j in range(1, half_sum + 1):
            if j >= lst[i - 1]:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - lst[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    # If there's no subset with half the total sum, return None
    if not dp[n][half_sum]:
        return None

    # Now reconstruct the subset from the DP table
    subset = []
    i, j = n, half_sum
    while i > 0 and j > 0:
        if dp[i][j] != dp[i - 1][j]:
            subset.append(lst[i - 1])
            j -= lst[i - 1]
        i -= 1

    return subset""",
"""def str_dist(x, y):
    if len(x) < len(y):
        return str_dist(y, x)

    if len(y) == 0:
        return len(x)

    previous_row = range(len(y) + 1)
    for i, c1 in enumerate(x):
        current_row = [i + 1]
        for j, c2 in enumerate(y):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_dom[-1]""",
"""def is_dag(graph):
    def dfs(node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    visited = set()
    for node in graph:
        if node not in visited:
            if dfs(node, visited, set()):
                return False
    return True""",
"""def foo(num, x=0):
    if num < 10:
        return num, x
    else:
        return foo(num * (2/3), x + 1)""",
"""def diff_sparse_matrices(matrices):
    result = {}
    for matrix in matrices:
        for key, value in matrix.items():
            if key in result:
                result[key] -= value
            else:
                result[key] = -value
    return result""",
"""def longest_subsequence_length(lst):
    def lis(seq):
        l = [1] * len(seq)
        for i in range(1, len(seq)):
            for j in range(i):
                if seq[i] > seq[j] and l[i] < l[j] + 1:
                    l[i] = l[j] + 1
        return max(l)

    def lds(seq):
        l = [1] * len(seq)
        for i in range(1, len(seq)):
            for j in range(i):
                if seq[i] < seq[j] and l[i] < l[j] + 1:
                    l[i] = l[j] + 1
        return max(l)

    return lis(lst), lds(lst)""",
"""""",
"""def find_primary_factors(n):
    factors = []
    # Start with the smallest prime factor, 2.
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    # Continue checking for factors from 3 onwards
    factor = 3
    while factor * factor <= n:
        while n % factor == 0:
            factors.append(factor)
            n = n // factor
        factor += 2
    # If n is a prime number greater than 2
    if n > 1:
        factors.append(n)
    return factors""",
"""def graphs_intersection(g1, g2):
    intersection = {}
    for node in g1:
        if node in g2:
            common_adjacent = set(g1[node]).intersection(g2[node])
            intersection[node] = list(common_adjacent)
    return intersection""",
"""def subset_sum(lst, target):
    def find_subsets_recursive(current, index, current_sum):
        if current_sum == target:
            subsets.add(tuple(current))
            return
        if index == len(lst) or current_sum > target:
            return
        find_subsets_recursive(current, index + 1, current_sum)
        find_subsets_recursive(current + [lst[index]], index + 1, current_sum + lst[index])

    subsets = set()
    find_subsets_recursive([], 0, 0)
    return subsets""",
"""def sum_mult_str(expression):
    while '(' in expression:
        start = expression.rfind('(')
        end = expression.find(')', start)
        substring = expression[start+1:end]
        expression = expression[:start] + evaluate_expression(substring) + expression[end+1:]
    
    elements = expression.split(' ')
    result = ''
    operator = '+'
    for element in elements:
        if element in ('+', '*'):
            operator = element
        else:
            if operator == '+':
                result += element
            elif operator == '*':
                result *= int(element)
    return result""",
"""def str_rep(s, k):
    for i in range(1, len(s) - k + 1):
        if s[i:k+i] in s[:i]:
            return True
    return False""",
"""def sort_two_sorted_lists(lst):
    mid = len(lst) // 2
    list1 = lst[:mid]
    list2 = lst[mid:]
    result = []

    for a, b in zip(list1, list2):
        result.append(a)
        result.append(b)

    return result""",
"""def prefix_suffix_match(lst, k):
    res_lst = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i][:k] == lst[j][-k:]:
                res_lst.append((i, j))
            elif lst[i][-k:] == lst[j][:k]:
                res_lst.append((j, i))
    return res_lst""",
"""def rotate_matrix_clockwise(mat):
    n = len(mat)
    for layer in range(n // 2):
        first = layer
        last = n - 1 - layer
        for i in range(first, last):
            # Save the top element
            top = mat[first][i]
            # Move left element to top
            mat[first][i] = mat[-1 - i][first]
            # Move bottom element to left
            mat[-1 - i][first] = mat[last][-1 - i]
            # Move right element to bottom
            mat[last][-1 - i] = mat[i][last]
            # Assign top element to right
            mat[i][last] = top
    return mat""",
"""def cyclic_shift(lst, direction, steps):
    if not lst:
        return lst
    steps %= len(lst)
    if direction == 'L':
        lst = lst[steps:] + lst[:steps]
    elif direction == 'R':
        lst = lst[-steps:] + lst[:-steps]
    return lst""",
"""def encode_string(s):
    encoded = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            encoded.append(f"{count}[{s[i - 1]}]")
            count = 1

    encoded.append(f"{count}[{s[-1]}]")
    return ''.join(encoded)""",
"""def list_sums(lst):
    if not lst:  # Check if the list is empty
        return []
    
    cum_sum = [lst[0]]  # Start with the first element
    for i in range(1, len(lst)):
        cum_sum.append(cum_sum[i-1] + lst[i])
    return cum_sum""",
"""def convert_base(num, base):
    if not isinstance(num, int) or not isinstance(num, int) or not (1 <= base <= 36):
        return None
    
    if num == 0:
        return '0'
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ''
    
    while num > 0:
        result = digits[num % base] + result
        num //= base
    
    return result""",
"""def max_div_seq(n, k):
    max_length = 0
    current_length = 0
    
    while n > 0:
        digit = n % 10
        if digit % k == 0:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
        n //= 10
    
    return max_length""",
"""def find_dup(lst):
    if len(lst) <= 1:
        return None
    
    slow = lst[0]
    fast = lst[lst[0]]
    
    while fast != slow:
        slow = lst[slow]
        fast = lst[lst[fast]]
        
    fast = 0
    while fast != slow:
        fast = lst[fast]
        slow = lst[slow]
    
    return slow""",
"""def lcm(a, b):
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    
    return abs(a * b) // gcd(a, b)""",
"""def f19():
    for number in range(1000, 10000):
        if number % 15 == 0:
            digits = [int(digit) for digit in str(number)]
            product = 1
            for digit in digits:
                product *= digit
            if 55 <= product <= 65:
                return number
    return None""",
"""from itertools import permutations

def f20():
    number = '14563743'
    for perm in permutations(number):
        perm_number = int(''.join(perm))
        if perm_number % 22 == 0:
            return perm_number
    return None""",
"""""",
"""import numpy as np

def mask_n(im, n, idx):
    region_size = im.size // n
    lower_bound = idx * region_size
    upper_bound = (idx + 1) * region_size if idx < n - 1 else im.size
    
    mask = (im.flat >= lower_bound) & (im.flat < upper_bound)
    return mask.reshape(im.shape)""",
"""import numpy as np

def entropy(mat):
    flat_mat = mat.flatten()
    probabilities = np.bincount(flat_mat) / len(flat_mat)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy""",
"""import numpy as np

def squeeze_vertical(im, factor):
    h, w = im.shape
    # Reshape the image to a 3D array where blocks of rows are grouped together
    reshaped = im.reshape(h // factor, factor, w)
    # Average over the blocks (axis=1) and then reshape back to a 2D array
    squeezed = reshaped.mean(axis=1)
    return squeezed""",
"""import numpy as np
from scipy.ndimage import median_filter

def denoise(im):
    denoised_im = im.copy()
    while True:
        prev_im = denoised_im.copy()
        denoised_im = median_filter(denoised_im, size=3)
        if np.array_equal(denoised_im, prev_im):
            break
    return denoised_im""",
"""import pandas as pd

def calculate_monthly_sales(transaction_data):
    # Group data by product and month
    grouped_data = transaction_data.groupby(['product', 'month'])
    
    # Calculate total and average monthly sales
    summary = grouped_data['sales'].agg(total_sales=('sales', 'sum'), average_sales=('sales', 'mean')).reset_index()
    
    return summary""",
"""def recommendations(movies, movies_genres, genres, search_title):
    # Find the movie by title
    movie = movies[movies['title'] == search_title].iloc[0]
    movie_id = movie['id']
    movie_genres = set(movies_genres[movies_genres['movie_id'] == movie_id]['genre'])

    # Filter out movies with the same genres
    candidate_movies = movies[movies['id'].isin(movies_genres[movies_genres['genre'].isin(genres)]['movie_id'])]

    # Calculate similarity score based on rating and runtime
    candidate_movies['similarity'] = candidate_movies.apply(
        lambda row: (1 - abs(row['rate'] - movie['rate']) / 10) * (1 - abs(row['runtime'] - movie['runtime']) / 180),
        axis=1
    )

    # Sort the candidate movies based on similarity score
    top_recommendations = candidate_movies.sort_values(by='similarity', ascending=False).head(3)

    # Select only the required columns
    top_recommendations = top_recommendations[['id', 'title', 'rate', 'runtime']]

    return top_recommendations""",
"""def top_hours_worked_departments(employees, departments, works_on):
    if employees.empty or departments.empty or works_on.empty:
        return pd.DataFrame()
    
    # Calculate total hours worked by each employee
    total_hours_by_employee = works_on.groupby('employee_id')['hours'].sum().reset_index()
    
    # Join with the employees DataFrame to get the department IDs
    employee_hours_with_dept = total_hours_by_employee.merge(employees, left_on='employee_id', right_on='id')
    
    # Group by department ID and calculate the sum of total hours worked for each department
    total_hours_by_dept = employee_hours_with_dept.groupby('department_id')['hours'].sum().reset_index()
    
    # Join with the departments DataFrame to get department names
    department_hours_with_names = total_hours_by_dept.merge(departments, left_on='department_id', right_on='id')
    
    # Sort by total hours in descending order and get the top 3
    top_departments = department_hours_with_names.sort_values(by='hours', ascending=False).head(3)
    
    return top_departments""",
"""def huge_population_countries(countries, borders):
    # Outer merge to get the population of the neighboring country
    merged_df = borders.merge(countries, how='outer', left_on='neighbor', right_on='name')
    merged_df.rename(columns={'population': 'neighbor_population'}, inplace=True)
    
    # Another outer merge to get the population of the other neighboring country
    merged_df = merged_df.merge(countries, how='outer', left_on='country', right_on='name', suffixes=('_neighbor', '_country'))
    
    # Group by country name and get the sum of neighboring countries' populations
    group_df = merged_df.groupby('country')['neighbor_population'].sum().reset_index()
    group_df.rename(columns={'neighbor_population': 'border_population_sum'}, inplace=True)
    
    # Outer merge to bring in the country's main population
    final_df = countries.merge(group_df, how='outer', left_on='name', right_on='country')
    
    # Filter countries with a population greater than the summed population of neighboring countries
    result_df = final_df[final_df['population'] > final_df['border_population_sum']]
    
    # Select the required columns and return the DataFrame
    return result_df[['name', 'population', 'border_population_sum']]""",
"""def countries_bordering_most_populated_in_asia(country_df, border_df):
    # Find the most populated country in Asia
    asian_countries = country_df[country_df['Continent'] == 'Asia']
    most_populated_country = asian_countries.loc[asian_countries['Population'].idxmax()]

    # Get the borders of the most populated country
    country_borders = border_df[border_df['Country'] == most_populated_country['Name']]
    
    # Extract the list of countries that border the most populated country
    bordering_countries_list = country_borders['Borders'].tolist()
    
    return bordering_countries_list"""
]
claude_classes_list = ["""class triangle:
    def __init__(self, a, b, ab, color):
        self.d = {'a': a, 'b': b, 'ab': ab, 'color': color}

    def get(self, name):
        return self.d.get(name)""",
"""class worker:
    def __init__(self, id, full_name, job, salary):
        self.id = id
        self.full_name = full_name
        self.job = job
        self.salary = salary

    def getFullName(self):
        return self.full_name

    def getSalary(self):
        return self.salary

    def getJob(self):
        return self.job

    def update(self, job=None, salary=None):
        if job is not None:
            self.job = job
        if salary is not None:
            self.salary = salary""",
"""class Binaric_arithmatic:
    def __init__(self, num):
        self.num = num

    def get(self):
        return self.num

    def inc(self):
        binary_number = int(self.num, 2)
        binary_number += 1
        self.num = bin(binary_number)[2:]

    def dec(self):
        binary_number = int(self.num, 2)
        if binary_number > 0:
            binary_number -= 1
        self.num = bin(binary_number)[2:]""",
"""import math

class Point_2D:
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
    
    def angle_wrt_origin(self):
        return self.theta""",
"""import random

class Roulette:
    def __init__(self, initial_money):
        self.balance = initial_money
        self.reds = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
        self.blacks = [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]

    def get_balance(self):
        return self.balance

    def bet(self, amount, bet_type):
        if amount > self.balance:
            print("Insufficient balance to place the bet.")
            return

        result = random.randint(0, 36)
        win = False

        if bet_type == 'red' and result in self.reds:
            win = True
        elif bet_type == 'black' and result in self.blacks:
            win = True
        elif bet_type == 'even' and result % 2 == 0:
            win = True
        elif bet_type == 'odd' and result % 2 == 1:
            win = True
        elif bet_type.isdigit() and int(bet_type) == result:
            win = True
        elif '-' in bet_type:
            start, end = map(int, bet_type.split('-'))
            if start <= result <= end:
                win = True

        if win:
            print(f"You won! The result was {result}.")
            if bet_type == 'red' or bet_type == 'black':
                self.balance += amount
            elif bet_type == 'even' or bet_type == 'odd':
                self.balance += amount
            elif bet_type.isdigit():
                self.balance += amount * 35
            elif '-' in bet_type:
                self.balance += amount * 2
        else:
            print(f"You lost! The result was {result}.")
            self.balance -= amount

        return self.get_balance()""",
"""class investments:
    def __init__(self, name, initial_investment, avg_yearly_return, monthly_income, monthly_expenses):
        self.name = name
        self.initial_investment = initial_investment
        self.avg_yearly_return = avg_yearly_return
        self.monthly_income = monthly_income
        self.monthly_expenses = monthly_expenses
        self.balance = initial_investment

    def __repr__(self):
        return f"InvestmentAccount(name='{self.name}', balance={self.balance})"

    def get_balance(self):
        return self.balance

    def get_future_value(self, years):
        net_monthly_cashflow = self.monthly_income - self.monthly_expenses
        future_balance = self.balance
        for _ in range(years * 12):
            future_balance = future_balance * (1 + (self.avg_yearly_return / 100) / 12) + net_monthly_cashflow
        return future_balance

    def update_value_by_year(self, years):
        self.balance = self.get_future_value(years)

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            raise ValueError("Insufficient funds for withdrawal")""",
"""class Restaurant:
    def __init__(self, name, cuisine, rating):
        self.name = name
        self.cuisine = cuisine
        self.rating = rating
        self.menu = {}
        self.chefs = []

    def add_dish(self, name, price):
        self.menu[name] = price

    def remove_dish(self, name):
        if name in self.menu:
            del self.menu[name]

    def add_chef(self, chef):
        self.chefs.append(chef)

    def remove_chef(self, chef):
        if chef in self.chefs:
            self.chefs.remove(chef)

    def get_menu(self):
        return self.menu

    def get_chefs(self):
        return self.chefs""",
"""class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __repr__(self):
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if coeff != 0:
                if i == 0:
                    terms.append(str(coeff))
                elif i == 1:
                    terms.append(f"{coeff}x")
                else:
                    terms.append(f"{coeff}x^{i}")
        return " + ".join(reversed(terms))

    def get_deg(self):
        return len(self.coeffs) - 1

    def __add__(self, other):
        max_deg = max(self.get_deg(), other.get_deg())
        new_coeffs = [0] * (max_deg + 1)
        for i in range(max_deg + 1):
            coeff1 = self.coeffs[i] if i < len(self.coeffs) else 0
            coeff2 = other.coeffs[i] if i < len(other.coeffs) else 0
            new_coeffs[i] = coeff1 + coeff2
        return Polynomial(new_coeffs)

    def __eq__(self, other):
        return self.coeffs == other.coeffs""",
"""class TodoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append({'task': task, 'completed': False})

    def remove_task(self, task):
        self.tasks = [t for t in self.tasks if t['task'] != task]

    def mark_completed(self, task):
        for t in self.tasks:
            if t['task'] == task:
                t['completed'] = True
                break

    def list_tasks(self, completed=None):
        if completed is None:
            return self.tasks
        else:
            return [t for t in self.tasks if t['completed'] == completed]""",
"""class RecipeBook:
    def __init__(self):
        self.recipes = []

    def add_recipe(self, name, ingredients, instructions):
        recipe = {
            'name': name,
            'ingredients': ingredients,
            'instructions': instructions
        }
        self.recipes.append(recipe)

    def remove_recipe(self, name):
        self.recipes = [recipe for recipe in self.recipes if recipe['name'] != name]

    def search_by_ingredient(self, ingredient):
        return [recipe for recipe in self.recipes if ingredient in recipe['ingredients']]"""
]
t5_func_list = ["""def sum_even(lst, delimiter):
    if not lst or delimiter not in lst:
        return 0
    start = lst.index(delimiter) + 1
    end = len(lst) - lst[::-1].index(delimiter) - 1
    return sum(lst[start:end])""",
"""def find_num_changes(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def sum_nested(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def str_decomp(input_string):
    words = input_string.split()
    return len(words)""",
"""def n_choose_k(matrix, k, value):
    count = 0
    for row in matrix:
        if row[k] > value:
            count += 1
    return count""",
"""def dfs_level_order():
    # Implement the function logic here
    # Since the docstring provided is not informative enough,
    # and does not specify inputs or the structure of "nice tree",
    # the function body is left intentionally empty.
    pass""",
"""def half_sum_subset(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""import random

def str_dist():
    point1 = (random.random(), random.random())
    point2 = (random.random(), random.random())
    distance = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
    return distance""",
"""def is_dag(graph):
    visited = set()
    recursion_stack = set()

    def visit(node):
        if node in recursion_stack:
            return False
        if node in visited:
            return True
        visited.add(node)
        recursion_stack.add(node)
        for neighbour in graph.get(node, []):
            if not visit(neighbour):
                return False
        recursionStack.remove(node)
        return True

    for node in graph:
        if node not in visited:
            if not visit(node):
                return False
    return True""",
"""def foo(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def diff_sparse_matrices(matrix_list):
    if not matrix_list:
        raise ValueError("The list of matrices must not be empty.")
    
    result = matrix_list[0]
    for matrix in matrix_list[1:]:
        result = [[result[i][j] - matrix[i][j] for j in range(len(result[0]))] for i in range(len(result))]
    
    return result""",
"""def longest_subsequence_length(seq):
    if not seq:
        return 0

    lengths = [1] * len(seq)
    for i in range(1, len(seq)):
        for j in range(i):
            if seq[i] > seq[j]:
                lengths[i] = max(lengths[i], lengths[j] + 1)

    return max(lengths)""",
"""def find_median(store, minimum):
    for item in store:
        if item >= minimum:
            return item
    return None""",
"""def find_primary_factors(number):
    factors = []
    divisor = 2
    while number > 1:
        while number % divisor == 0:
            factors.append(divisor)
            number //= divisor
        divisor += 1
    return factors""",
"""def graphs_intersection(graph1, graph2):
    intersection = {}
    for node in graph1:
        if node in graph2:
            edges_in_both = set(graph1[node]) & set(graph2[node])
            if edges_in_both:
                intersection[node] = list(edges_in_both)
    return intersection""",
"""def subset_sum(total_sum):
    if total_sum == 0:
        return [[0]]

    combinations = []
    for num in range(1, total_sum + 1):
        remaining_sum = total_sum - num
        sub_combinations = subset_sum(remaining_sum)
        for sub_combination in sub_combinations:
            combinations.append([num] + sub_combination)
            combinations.append([-num] + sub_combination)
    
    # Filter out duplicates by converting the inner lists to tuples and then back to lists
    combinations = [list(t) for t in set(tuple(sorted(comb, key=abs)) for comb in combinations)]
    
    return combinations""",
"""def sum_mult_str(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def str_rep(str_rep):
    try:
        isinstance(str_rep, str)
        return True
    except TypeError:
        return False""",
"""def sort_two_sorted_lists(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def prefix_suffix_match(lst, prefix='', suffix=''):
    return [f"{prefix}{item}{suffix}" for item in lst]""",
"""def rotate_matrix_clockwise(matrix, degrees):
    if degrees not in [0, 90, 180, 270, 360]:
        raise ValueError("Degrees must be one of the following values: 0, 90, 180, 270, 360")

    if degrees == 0 or degrees == 360:
        return matrix

    rotated_matrix = matrix
    num_rotations = degrees // 90

    for _ in range(num_rotations):
        rotated_matrix = list(zip(*rotated_matrix[::-1]))

    return [list(row) for row in rotated_matrix]""",
"""def cyclic_shift(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def encode_string(strings):
    return ','.join(s.replace(',', '\\,') for s in strings)""",
"""def list_sums(matrix):
    if not matrix or not all(isinstance(row, list) for row in matrix) or not all(matrix):
        raise ValueError("Input must be a non-empty matrix (list of non-empty lists).")
    return [sum(col) for col in zip(*matrix)]""",
"""def convert_base(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def max_div_seq(ball_dict):
    return max(ball_dict.values())""",
"""def find_dup(lst):
    best_pointer = 0
    best_value = lst[0] if lst else None

    for i in range(1, len(lst)):
        if lst[i] > best_value:
            best_pointer, best_value = i, lst[i]

    return best_pointer""",
"""def lcm():
    return len(str(1))""",
"""def f19(lst):
    res = 0
    for i, item in enumerate(lst):
        res += item * i 
    return res""",
"""def f20(lst):
    if not lst:
        return 0
    return lst.count(lst[0])""",
"""""",
"""def mask_n(n):
    return n & (n - 1)""",
"""from math import log2

def entropy(data):
    from collections import Counter
    if not data:
        return 0
    counts = Counter(data)
    probabilities = [count / len(data) for count in counts.values()]
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy""",
"""def squeeze_vertical(impact, half_space):
    # Placeholder for the actual computation to reduce the impact size.
    # Replace 'reduced_impact' with the actual computation.
    reduced_impact = None

    # TODO: Implement the logic to reduce the impact size in aqueous half-space.

    return reduced_impact""",
"""import numpy as np
from skimage import segmentation, filters, measure

def denoise(image):
    # Obtain the gradient of the image
    gradient = filters.sobel(image)
    
    # Find markers for watershed
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < np.mean(image)] = background
    markers[image > np.mean(image)] = foreground
    
    # Compute the watershed segmentation
    segmented = segmentation.watershed(gradient, markers)
    
    # Label the regions
    labeled_image, _ = measure.label(segmented, return_num=True)
    
    # Denoise image by only keeping the largest segment
    props = measure.regionprops(labeled_image, intensity_image=image)
    max_area = 0
    max_label = 0
    for prop in props:
        if prop.area > max_area:
            max_area = prop.area
            max_label = prop.label
            
    denoised_image = np.where(labeled_image == max_label, image, 0)
    
    return denoised_image""",
"""def calculate_monthly_sales(df):
    # Placeholder function to extract information from a pandas DataFrame
    # The actual implementation will depend on the type of information needed to be extracted
    # As the docstring does not specify what information to extract, a general example is provided

    summary_info = {
        'column_names': df.columns.tolist(),
        'data_types': df.dtypes.to_dict(),
        'number_of_rows': len(df),
        'number_of_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return summary_info""",
"""def recommendations(movies, title):
    movie_ids = []
    for movie in movies:
        if movie['title'] == title:
            movie_ids.append(movie['id'])
    return movie_ids""",
"""def top_hours_worked_departments(df):
    duplicates = df[df.duplicated(subset='hours', keep=False)]
    max_hours = duplicates.groupby('hours').idxmax()
    for hour, idx in max_hours.iterrows():
        df.loc[idx[0], 'hours'] += df.loc[duplicates[duplicates['hours'] == hour].index, 'hours'].sum()
    return df""",
"""def huge_population_countries(populations, country_sets):
    uncovered_count = {}
    for population in populations:
        covered = any(population in country_set for country_set in country_sets)
        if not covered:
            uncovered_count[population] = uncovered_count.get(population, 0) + 1
    return uncovered_count""",
"""def countries_bordering_most_populated_in_asia(iso_hybridization):
    # The logic to calculate the border count should be added here.
    # Since the calculation method is not specified in the docstring,
    # this is just a placeholder for the actual logic.
    
    calculated_border_count = 0  # Placeholder for the actual calculation.
    
    # Assuming iso_hybridization has an attribute called border_count
    # that needs to be updated with the calculated value.
    iso_hybridization.border_count = calculated_border_count"""
""""""]
t5_classes_list = ["""""",
"""class worker:
    def __init__(self, first_name):
        self.name = first_name""",
"""""",
"""""",
"""""",
"""""",
"""""",
"""""",
"""""",
""""""]

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
    func_name = func_str.split('(')[0].split()[-1] if func_str else "func"
    
    # Define a placeholder in the local namespace
    local_namespace = {}
    exec(f"def {func_name}(): pass", globals(), local_namespace)
    
    # Compile and execute the function string in the local namespace
    code = compile(func_str, '<string>', 'exec')
    exec(code, globals(), local_namespace)

    globals()[func_name] = local_namespace[func_name]
    
    # Return the function object from the local namespace
    return local_namespace[func_name]

class dummy_class:
    def __init__(self):
        pass

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
    # raise ValueError("Class definition not found in the provided string.")
    return dummy_class

# Dictionary to map function indices to their respective test cases
test_cases = {i: f'TestGeneratedSolution{i+1}' for i in range(len(functions_list) + len(classes_list))}

# Transform string representations to runnable functions
imported_functions = [string_to_function(func_str) for func_str in claude_func_list]

imported_classes = [string_to_class(class_str) for class_str in claude_classes_list]

Point_2D = imported_classes[3]
Polynomial = imported_classes[7]


class TimeoutException(Exception):
    pass

def timeout(default_seconds=5):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Use the timeout defined in the test class if available, otherwise use the default
            seconds = getattr(self, 'timeout', default_seconds)
            
            def target(result):
                try:
                    result.append(func(self, *args, **kwargs))
                except Exception as e:
                    result.append(e)
            
            result = []
            thread = threading.Thread(target=target, args=(result,))
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutException(f"Test {func.__name__} timed out after {seconds} seconds")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator


# Base class for test cases to inherit from
class BaseTestCase(unittest.TestCase):
    def setUp(self):
        timeout = 5

    def tearDown(self):
        pass  # Optional teardown method to run after each test

class TestGeneratedSolution1(BaseTestCase):
    sum_even = imported_functions[0]
    
    @timeout()
    def test_empty_list(self):
        self.assertEqual(sum_even([]), 0)
    
    @timeout()
    def test_single_item_list(self):
        self.assertEqual(sum_even([1]), 1)
    
    @timeout()
    def test_flat_list(self):
        lst = [0, 1, 2, 3, 4, 5]
        self.assertEqual(sum_even(lst), 6)
    
    @timeout()
    def test_nested_list(self):
        lst = [1, [2, 3, [4, 5]]]
        self.assertEqual(sum_even(lst), 7)
    
    @timeout()
    def test_multiple_exclusive_nesting(self):
        lst = [1, [2, 3, [4, 5]], 6, [7, 8]]
        self.assertEqual(sum_even(lst), 20)

    @timeout()
    def test_two_level_nesting(self):
        lst = [[[1, 2, 3], [4, 5, 6]]]
        self.assertEqual(sum_even(lst), 14)



class TestGeneratedSolution2(BaseTestCase):
    find_num_changes = imported_functions[1]

    @timeout()
    def test_empty_list(self):
        self.assertEqual(find_num_changes(5, []), 0)
    
    @timeout()
    def test_zero_candidate(self):
        lst = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(find_num_changes(0, lst), 1)
    
    @timeout()
    def test_negative_candidate(self):
        lst = [1,2,3,4,5]
        self.assertEqual(find_num_changes(-1, lst) , 0)
    
    @timeout()
    def test_candidate_not_in_list(self):
        lst = [2, 5, 7]
        self.assertEqual(find_num_changes(1, lst), 0)
    
    @timeout()
    def test_candidate_in_list(self):
        lst = [1, 2, 5, 6]
        self.assertEqual(find_num_changes(4, lst), 3)

class TestGeneratedSolution3(BaseTestCase):
    sum_nested = imported_functions[2]

    @timeout()
    def test_empty_list(self):
        self.assertEqual(sum_nested([]), 0.0)

    @timeout()
    def test_flat_list_positives(self):
        lst = [1, 2, 3]
        self.assertEqual(sum_nested(lst), 6.0)
    
    @timeout()
    def test_flat_list_nums(self):
        lst = [1, -2, 3]
        self.assertEqual(sum_nested(lst), 6.0)
    
    @timeout()
    def test_flat_list_mixed(self):
        lst = [1, 2, "ab", -4]
        self.assertEqual(sum_nested(lst), 7.0)
    
    @timeout()
    def test_flat_list_strings(self):
        lst = ['a', "abd", 'zzz', "hello world"]
        self.assertEqual(sum_nested(lst), 0.0)
    
    @timeout()
    def test_nested_list_positives(self):
        lst = [0.5, 2.5, [3, 4], [5, [6, 7], 8], 9.4]
        self.assertEqual(sum_nested(lst), 45.4)

    @timeout()
    def test_nested_list_nums(self):
        lst = [0.5, -2.5, [3, -4], [5, [-6, 7], 8], 9.4]
        self.assertEqual(sum_nested(lst), 45.4)

    @timeout()
    def test_nested_list_mixed(self):
        lst = ["aa", [-3, -4.6], 'abc', [5, 'abc', [-4, 0.5]]]
        self.assertEqual(sum_nested(lst), 17.1)
    
    @timeout()
    def test_nested_list_strings(self):
        lst = ["aa", "b", ["hello"]]
        self.assertEqual(sum_nested(lst), 0.0)

class TestGeneratedSolution4(BaseTestCase):
    str_decomp = imported_functions[3]

    @timeout()
    def test_empty_target(self):
        word_bank = ["aa", "bb", "cc"]
        self.assertEqual(str_decomp('', word_bank), 1)
    
    @timeout()
    def test_empty_word_bank(self):
        target = 'abc'
        self.assertEqual(str_decomp(target, []), 0)

    @timeout()
    def test_target_not_in_bank(self):
        target = "abc"
        word_bank = ["z", "x"]
        self.assertEqual(str_decomp(target, word_bank), 0)
    
    @timeout()
    def test_target_in_bank(self):
        target = "abc"
        word_bank = ["z", "x", "y", "abc"]
        self.assertEqual(str_decomp(target, word_bank), 1)
    
    @timeout()
    def test_overlaping_words_only(self):
        target = "abcdef"
        word_bank = ["ab", "cd", "def", "abcd"]
        self.assertEqual(str_decomp(target, word_bank), 0)

    @timeout()
    def test_repeatidly_using_word(self):
        target = "purple"
        word_bank = ["p", "ur", 'le']
        self.assertEqual(str_decomp(target, word_bank), 1)
    
    @timeout()
    def test_multiple_options(self):
        target = "purple"
        word_bank = ["purp", "e", "purpl", 'le']
        self.assertEqual(str_decomp(target, word_bank), 2)

    @timeout()
    def test_multiple_options_with_reps(self):
        target = "aabbcc"
        word_bank = ["a", "ab", "b", "bc", "c", "abc", "abcd"]
        self.assertEqual(str_decomp(target, word_bank), 4)

class TestGeneratedSolution5(BaseTestCase):
    n_choose_k = imported_functions[3]

    @timeout()
    def test_n_negative(self):
        n, k = -1, -1 
        self.assertEqual(n_choose_k(n, k), 0)

    @timeout()
    def test_k_larger(self):
        n, k = 1, 2
        self.assertEqual(n_choose_k(n, k), 0)
    
    @timeout()
    def test_k_negative(self):
        n, k = 4, -1
        self.assertEqual(n_choose_k(n, k), 0)
    
    @timeout()
    def test_n_equals_k(self):
        n, k = 3, 3
        self.assertEqual(n_choose_k(n, k), 1)
    
    @timeout()
    def test_k_is_one(self):
        n, k = 8, 1
        self.assertEqual(n_choose_k(n, k), 8)
    
    @timeout()
    def test_k_is_zero(self):
        n, k = 5, 0
        self.assertEqual(n_choose_k(n, k), 1)
    
    @timeout()
    def test_k_one_smaller(self):
        n, k = 9, 8
        self.assertEqual(n_choose_k(n, k), 9)
    
    @timeout()
    def test_regular_case(self):
        n, k = 10, 3
        self.assertEqual(n_choose_k(n, k), 120)

    
class TestGeneratedSolution6(BaseTestCase):
    dfs_level_order = imported_functions[5]

    @timeout()
    def test_empty_tree(self):
        self.assertEqual(dfs_level_order([]), "")

    @timeout()
    def test_single_node_tree(self):
        self.assertEqual(dfs_level_order([1]), "1")

    @timeout()
    def test_full_tree(self):
        tree = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(dfs_level_order(tree), "1,2,4,5,3,6,7")

    @timeout()
    def test_incomplete_tree(self):
        tree = [1, 2, 3, None, 5, None, 7]
        self.assertEqual(dfs_level_order(tree), "1,2,5,3,7")

    @timeout()
    def test_complex_tree(self):
        tree = [1, 2, 3, 4, None, None, 5, 6, None, None, None, None, None, 7]
        self.assertEqual(dfs_level_order(tree), "1,2,4,6,3,5,7")

    @timeout()
    def test_all_none_tree(self):
        tree = [None, None, None]
        self.assertEqual(dfs_level_order(tree), "")

    @timeout()
    def test_large_tree(self):
        tree = list(range(1, 16))
        self.assertEqual(dfs_level_order(tree), "1,2,4,8,9,5,10,11,3,6,12,13,7,14,15")

class TestGeneratedSolution7(BaseTestCase):
    half_sum_subset = imported_functions[6]
        
    @timeout()
    def test_empty_list(self):
        self.assertEqual(half_sum_subset([]), [])

    @timeout()
    def test_single_item_list(self):
        self.assertIsNone(half_sum_subset([1]))
    
    @timeout()
    def test_two_items_no_half_sum(self):
        self.assertIsNone(half_sum_subset([1, 2]))
    
    @timeout()
    def test_two_items_with_half_sum(self):
        self.assertEqual(half_sum_subset([2, 2]), [2])
    
    @timeout()
    def test_multiple_items_no_half_sum(self):
        self.assertIn(sorted(half_sum_subset([1, 2, 3])), [[1, 2], [3]])
    
    @timeout()
    def test_multiple_items_with_half_sum(self):
        self.assertIn(sorted(half_sum_subset([1, 2, 3, 4])), [[1, 4], [2, 3]])
    
    @timeout()
    def test_list_with_zero(self):
        self.assertIn(sorted(half_sum_subset([0, 1, 2, 3])), [[1, 2], [3], [0, 1, 2], [0, 3]])
    
    @timeout()
    def test_with_negatives(self):
        self.assertIn(sorted(half_sum_subset([-1, -2, 3, 6])), [[-2, -1, 6], [3]])
        
    @timeout()
    def test_with_even_total_sum_but_no_half_sum(self):
        self.assertIsNone(half_sum_subset([1, 3, 5, 13]))
    
    @timeout()
    def test_large_list(self):
        self.assertIn(sorted(half_sum_subset([3, 1, 4, 2, 2])), [[1, 2, 3], [2, 4]])


class TestGeneratedSolution8(BaseTestCase):
    str_dist = imported_functions[7]

    @timeout()
    def test_empty_strings(self):
        self.assertEqual(str_dist("", ""), 0)

    @timeout()
    def test_empty_first_string(self):
        self.assertEqual(str_dist("", "abc"), 3)

    @timeout()
    def test_empty_second_string(self):
        self.assertEqual(str_dist("abc", ""), 3)

    @timeout()
    def test_equal_strings(self):
        self.assertEqual(str_dist("abc", "abc"), 0)

    @timeout()
    def test_one_char_insert(self):
        self.assertEqual(str_dist("a", "ab"), 1)

    @timeout()
    def test_one_char_delete(self):
        self.assertEqual(str_dist("ab", "a"), 1)

    @timeout()
    def test_one_char_replace(self):
        self.assertEqual(str_dist("a", "b"), 1)

    @timeout()
    def test_multiple_operations(self):
        self.assertEqual(str_dist("flaw", "lawn"), 2)
        self.assertEqual(str_dist("intention", "execution"), 5)

class TestGeneratedSolution9(BaseTestCase):
    is_dag = imported_functions[8]

    @timeout()
    def test_empty_graph(self):
        self.assertTrue(is_dag([]))

    @timeout()
    def test_single_node_graph(self):
        self.assertTrue(is_dag([[]]))

    @timeout()
    def test_two_node_acyclic_graph(self):
        self.assertTrue(is_dag([[1], []]))

    @timeout()
    def test_two_node_cyclic_graph(self):
        self.assertFalse(is_dag([[1], [0]]))

    @timeout()
    def test_three_node_acyclic_graph(self):
        self.assertTrue(is_dag([[1], [2], []]))

    @timeout()
    def test_three_node_cyclic_graph(self):
        self.assertFalse(is_dag([[1], [2], [0]]))

    @timeout()
    def test_complex_acyclic_graph(self):
        self.assertTrue(is_dag([[1, 2], [3], [3], []]))

    @timeout()
    def test_complex_cyclic_graph(self):
        self.assertFalse(is_dag([[1, 2], [3], [3], [1]]))

    @timeout()
    def test_self_loop(self):
        self.assertTrue(is_dag([[0]]))

    @timeout()
    def test_disconnected_graph(self):
        self.assertTrue(is_dag([[1], [], [3], []]))

class TestGeneratedSolution10(BaseTestCase):
    foo = imported_functions[9]

    @timeout()
    def test_initial_below_10(self):
        self.assertEqual(foo(5), (5, 0))

    @timeout()
    def test_initial_exactly_10(self):
        self.assertEqual(foo(10), (10 * (2/3), 1))

    @timeout()
    def test_initial_above_10(self):
        self.assertEqual(foo(20), (20 * (2/3) * (2/3), 2))

    @timeout()
    def test_initial_below_10_with_counter(self):
        self.assertEqual(foo(5, 2), (5, 2))

    @timeout()
    def test_initial_above_10_with_counter(self):
        self.assertEqual(foo(30, 2), (30 * (2/3) * (2/3) * (2/3), 5))

    @timeout()
    def test_large_initial_value(self):
        self.assertEqual(foo(1000), (7.707346629258933, 12))

    @timeout()
    def test_large_initial_value_with_counter(self):
        self.assertEqual(foo(1000, 3), (7.707346629258933, 15))

    @timeout()
    def test_initial_zero(self):
        self.assertEqual(foo(0), (0, 0))

    @timeout()
    def test_initial_negative(self):
        self.assertEqual(foo(-20), (-20, 0))

class TestGeneratedSolution11(BaseTestCase):
    diff_sparse_matrices = imported_functions[10]

    @timeout()
    def test_single_matrix(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}]), {(1, 3): 2, (2, 7): 1})

    @timeout()
    def test_two_matrices(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}, {(1, 3): 6}]), {(1, 3): -4, (2, 7): 1})

    @timeout()
    def test_two_matrices_zero_difference(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}, {(1, 3): 2}]), {(1, 3): 0, (2, 7): 1})

    @timeout()
    def test_multiple_matrices(self):
        self.assertEqual(diff_sparse_matrices([{(1, 3): 2, (2, 7): 1}, {(1, 3): 6, (9, 10): 7}, {(2, 7): 0.5, (4, 2): 10}]), 
                         {(1, 3): -4, (2, 7): 0.5, (9, 10): -7, (4, 2): -10})

    @timeout()
    def test_no_difference(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 5}, {(1, 1): 5}]), {(1, 1): 0})

    @timeout()
    def test_negative_values(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 3}, {(1, 1): -4}]), {(1, 1): 7})

    @timeout()
    def test_disjoint_matrices(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 3}, {(2, 2): 4}]), {(1, 1): 3, (2, 2): -4})

    @timeout()
    def test_multiple_entries(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 1, (2, 2): 2, (3, 3): 3}, {(1, 1): 1, (2, 2): 2}, {(3, 3): 3}]), 
                         {(1, 1): 0, (2, 2): 0, (3, 3): 0})

    @timeout()
    def test_zero_values(self):
        self.assertEqual(diff_sparse_matrices([{(1, 1): 0}, {(1, 1): 3}]), {(1, 1): -3})

class TestGeneratedSolution12(BaseTestCase):
    longest_subsequence_length = imported_functions[11]

    @timeout()
    def test_empty_list(self):
        self.assertEqual(longest_subsequence_length([]), 0)

    @timeout()
    def test_all_increasing(self):
        self.assertEqual(longest_subsequence_length([1, 2, 3, 4, 5]), 5)

    @timeout()
    def test_all_decreasing(self):
        self.assertEqual(longest_subsequence_length([5, 4, 3, 2, 1]), 5)

    @timeout()
    def test_mixed_increasing_decreasing(self):
        self.assertEqual(longest_subsequence_length([1, -4, 7, -5]), 3)

    @timeout()
    def test_single_element(self):
        self.assertEqual(longest_subsequence_length([-4]), 1)

    @timeout()
    def test_mixed_with_long_subsequence(self):
        self.assertEqual(longest_subsequence_length([1, -4, 2, 9, -8, 10, -6]), 4)

    @timeout()
    def test_increasing_then_decreasing(self):
        self.assertEqual(longest_subsequence_length([1, 3, 5, 4, 2]), 3)

    @timeout()
    def test_equal_elements(self):
        self.assertEqual(longest_subsequence_length([2, 2, 2, 2, 2]), 1)

class TestGeneratedSolution13(BaseTestCase):
    find_median = imported_functions[12]
    
    @timeout()
    def test_odd_number_of_elements(self):
        self.assertEqual(find_median([1, 2, 3, 4, 5]), 3)
        self.assertEqual(find_median([5, 4, 3, 2, 1]), 3)
        self.assertEqual(find_median([3, 1, 2]), 2)

    @timeout()
    def test_even_number_of_elements(self):
        self.assertAlmostEqual(find_median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(find_median([1, -4, 7, -5]), -1.5)
        self.assertAlmostEqual(find_median([1, 2, -4, -7]), -1.5)

    @timeout()
    def test_single_element(self):
        self.assertEqual(find_median([7]), 7)
        self.assertEqual(find_median([-1]), -1)

    @timeout()
    def test_negative_numbers(self):
        self.assertEqual(find_median([-1, -2, -3, -4, -5]), -3)
        self.assertEqual(find_median([-5, -3, -1, -2, -4]), -3)

    @timeout()
    def test_mixed_positive_and_negative_numbers(self):
        self.assertEqual(find_median([1, -1, 2, -2, 0]), 0)
        self.assertEqual(find_median([-1, 0, 1]), 0)

    @timeout()
    def test_duplicate_numbers(self):
        self.assertEqual(find_median([1, 2, 2]), 2)
        self.assertEqual(find_median([2, 2, 2, 2, 2]), 2)
        
    @timeout()
    def test_large_list(self):
        self.assertEqual(find_median(list(range(1, 101))), 50.5)
        
    @timeout()
    def test_float_list(self):
        self.assertEqual(find_median([1.5, 2.5, 3.5, 4.5]), 3.0)
    
    @timeout()
    def test_mix_float_and_int(self):
        self.assertEqual(find_median([1.5, 2, 3.5, 4]), 2.75)

class TestGeneratedSolution14(BaseTestCase):
    find_primary_factors = imported_functions[13]

    @timeout()
    def test_prime_number(self):
        self.assertEqual(find_primary_factors(7), [7])

    @timeout()
    def test_composite_number(self):
        self.assertEqual(find_primary_factors(12), [2, 2, 3])
        self.assertEqual(find_primary_factors(105), [3, 5, 7])

    @timeout()
    def test_large_prime_number(self):
        self.assertEqual(find_primary_factors(101), [101])
        self.assertEqual(find_primary_factors(1000000007), [1000000007])

    @timeout()
    def test_large_composite_number(self):
        self.assertEqual(find_primary_factors(1001), [7, 11, 13])
        self.assertEqual(find_primary_factors(1524878), [2, 29, 61, 431])
        self.assertEqual(find_primary_factors(97*89), [89, 97])

    @timeout()
    def test_large_composite_number_with_repeated_factors(self):
        self.assertEqual(find_primary_factors(1000), [2, 2, 2, 5, 5, 5])
        self.assertEqual(find_primary_factors(2**10), [2]*10)
        self.assertEqual(find_primary_factors(2**4 * 3**3), [2, 2, 2, 2, 3, 3, 3])

    @timeout()
    def test_large_composite_number_with_large_prime_factors_and_repeated_factors(self):
        self.assertEqual(sorted(find_primary_factors(1524878*29)), [2, 29, 29, 61, 431])
    
    @timeout()
    def test_no_factors(self):
        self.assertEqual(find_primary_factors(1), [])
    
    @timeout()
    def test_smallest_prime(self):
        self.assertEqual(find_primary_factors(2), [2])
        
class TestGeneratedSolution15(BaseTestCase):
    graphs_intersection = imported_functions[14]

    @timeout()
    def test_empty_graphs(self):
        self.assertEqual(graphs_intersection({}, {}), {})

    @timeout()
    def test_single_node_graphs(self):
        self.assertEqual(graphs_intersection({1: []}, {1: []}), {})

    @timeout()
    def test_single_edge_graphs(self):
        self.assertEqual(graphs_intersection({1: [2]}, {1: [2]}), {1: [2]})

    @timeout()
    def test_single_edge_graphs_no_intersection(self):
        self.assertEqual(graphs_intersection({1: [2]}, {1: [3]}), {})

    @timeout()
    def test_single_edge_graphs_different_nodes(self):
        self.assertEqual(graphs_intersection({1: [2]}, {3: [4]}), {})

    @timeout()
    def test_single_edge_graphs_opposite_direction(self):
        self.assertEqual(graphs_intersection({1: [2]}, {2: [1]}), {})

    @timeout()
    def test_single_edge_graphs_shared_node_no_intersection(self):
        self.assertEqual(graphs_intersection({1: [2]}, {2: [3]}), {})

    @timeout()
    def test_graphs_form_a_directed_cycle_no_intersection(self):
        self.assertEqual(graphs_intersection({1: [2]}, {2: [3], 3: [1]}), {})

    @timeout()
    def test_same_multiple_edges_graphs(self):
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [3]}, {1: [2, 3], 2: [3]}), {1: [2, 3], 2: [3]})

    @timeout()
    def test_intersection_is_subset_of_the_graphs(self):
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [3]}, {1: [3], 2: [4]}), {1: [3]})     
    
    @timeout()
    def test_intersection_lager_graphs(self):
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]} , {1: [3, 4], 2: [3, 5], 3: [1, 2], 4: [1], 5: [2]}), {1: [3], 2: [3], 3: [1, 2]})
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]} , {1: [2, 3, 5], 2: [1, 3], 3: [1, 2], 4: [5], 5: [1, 4]}), {1: [2, 3], 2: [1, 3], 3: [1, 2]})
        self.assertEqual(graphs_intersection({1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]} , {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3]}), {1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2]})   

class TestGeneratedSolution16(BaseTestCase):
    subset_sum = imported_functions[15]

    @timeout()
    def test_empty_list_zero_target(self):
        self.assertEqual(subset_sum([], 0), {()})
        
    @timeout()
    def test_empty_list_non_zero_target(self):
        self.assertEqual(subset_sum([], 1), set())

    @timeout()
    def test_single_item_list(self):
        self.assertEqual(subset_sum([1], 1), {(1,)})

    @timeout()
    def test_single_item_list_no_sum(self):
        self.assertEqual(subset_sum([1], 2), set())

    @timeout()
    def test_list_sums_to_target(self):
        self.assertEqual(subset_sum([1, 2], 3), {(1, 2)})

    @timeout()
    def test_target_greater_than_sum_of_list(self):
        self.assertEqual(subset_sum([1, 2], 4), set())

    @timeout()
    def test_no_subset_sum(self):
        self.assertEqual(subset_sum([1, 2, 6], 5), set())
    
    @timeout()
    def test_multiple_subsets(self):
        self.assertEqual(subset_sum([1, 2, 3], 3), {(3,), (1, 2)})
    
    @timeout()
    def test_multiple_subsets_with_duplicates_in_list(self):
        self.assertEqual(subset_sum([1, 2, 2], 3), {(1, 2)})
        self.assertEqual(subset_sum([1, 2, 2, 3], 3), {(3,), (1, 2)})
        
    @timeout()
    def test_multiple_subsets_with_repeats_in_subsets(self):
        self.assertEqual(subset_sum([1, 2, 2], 5), {(1, 2, 2)})
        self.assertEqual(subset_sum([1, 2, 2, 3], 3), {(3,), (1, 2)})
    
    @timeout()
    def test_negatives(self):
        self.assertEqual(subset_sum([-2, -1, 3], 2), {(-1, 3)})
    
    @timeout()
    def test_negatives_with_zero_target(self):
        self.assertEqual(subset_sum([-2, -1, 3], 0), {(-2, -1, 3), ()})

    @timeout()
    def test_negatives_with_negative_target(self):
        self.assertEqual(subset_sum([-2, -1, 3], -3), {(-2, -1)})
        
    @timeout()
    def test_negatives_with_zero_element_and_zero_target(self):
        self.assertEqual(subset_sum([-2, -1, 0, 3], 0), {(0,), (), (-2, -1, 0, 3), (-2, -1, 3)})
    
    @timeout()
    def test_list_is_singelton_zero_and_zero_target(self):
        self.assertEqual(subset_sum([0], 0), {(0,), ()})

class TestGeneratedSolution17(BaseTestCase):
    sum_mult_str = imported_functions[16]
    
    @timeout()
    def test_single_string(self):
        self.assertEqual(sum_mult_str("'hello'"), "hello")
    
    @timeout()
    def test_two_string_operands_addition_only(self):
        self.assertEqual(sum_mult_str("'a'+'b'"), "ab")
    
    @timeout()
    def test_string_multiplication_only(self):
        self.assertEqual(sum_mult_str("'a'*'3'"), "aaa")
    
    @timeout()
    def test_multiple_operands_addition_only(self):
        self.assertEqual(sum_mult_str("'a'+'b'+'c'"), "abc")
        
    @timeout()
    def test_multiple_operands_multiplication_only(self):
        self.assertEqual(sum_mult_str("'a'*'3'*'2'"), "aaaaaa")
    
    @timeout()
    def test_numbers_numltiplication_(self):
        self.assertEqual(sum_mult_str("'3'*'3'"), "333")
    
    @timeout()
    def test_mixed_operands(self):
        self.assertEqual(sum_mult_str("'abc'*'3'+'def'"), "abcabcabcdef")
    
    @timeout()
    def test_mixed_operands_unintuitive_order(self):
        self.assertEqual(sum_mult_str("'12'+'aa'*'2'"), "12aa12aa")
    
    @timeout()
    def test_empty_string(self):
        self.assertEqual(sum_mult_str("''"), "")
    
    @timeout()
    def test_add_empty_string(self):
        self.assertEqual(sum_mult_str("'a'+''"), "a")
    
    @timeout()
    def test_multiply_by_zero(self):
        self.assertEqual(sum_mult_str("'a'*'0'"), "")
    
    @timeout()
    def test_no_operations(self):
        self.assertEqual(sum_mult_str("'a'"), "a")
        self.assertEqual(sum_mult_str("'73'"), "73")
        
class TestGeneratedSolution18(BaseTestCase):
    str_rep = imported_functions[17]
    
    @timeout()
    def test_with_repeats(self):
        self.assertTrue(str_rep("abcabc", 3))
        self.assertTrue(str_rep("aab2bab22", 3))
        
    @timeout()
    def test_no_repeats(self):
        self.assertFalse(str_rep("abcabc", 4))
        self.assertFalse(str_rep("aab2bab22", 4))
    
    @timeout()
    def test_single_char(self):
        self.assertFalse(str_rep("a", 1))
        self.assertFalse(str_rep("a", 2))
    
    @timeout()
    def test_empty_string(self):
        self.assertFalse(str_rep("", 1))
        self.assertFalse(str_rep("", 2))
    
    @timeout()
    def test_repeating_substring_of_length_1(self):
        self.assertTrue(str_rep("aba", 1))
        
    @timeout()
    def test_long_string_with_repeating_substring(self):
        self.assertTrue(str_rep("abcdefghijklmnopabcdefghijklmnop", 16))
    
    @timeout()
    def test_repeating_substring_with_overlap(self):
        self.assertTrue(str_rep("ababa", 3))

class TestGeneratedSolution19(BaseTestCase):
    sort_two_sorted_lists = imported_functions[18]
    
    @timeout()
    def test_empty_list(self):
        self.assertEqual(sort_two_sorted_lists([]), [])
    
    @timeout()
    def test_two_items_list(self):
        self.assertEqual(sort_two_sorted_lists([1, 2]), [1, 2])
        self.assertEqual(sort_two_sorted_lists([2, 1]), [1, 2])
        
    @timeout()
    def test_with_negatives(self):
        self.assertEqual(sort_two_sorted_lists([-3, 1, -1, -2]), sorted([-3, 1, -1, -2]))
    
    @timeout()
    def test_large_list(self):
        self.assertEqual(sort_two_sorted_lists([7, 6, 11, 4, 12, 0, 20, -10, 30, -30]), sorted([7, 6, 11, 4, 12, 0, 20, -10, 30, -30]))
    
class TestGeneratedSolution20(BaseTestCase):
    prefix_suffix_match = imported_functions[19]
    
    @timeout()
    def test_empty_list(self):
        self.assertEqual(prefix_suffix_match([], 1), [])
        self.assertEqual(prefix_suffix_match([], 6), [])
        
    @timeout()
    def test_single_item_list(self):
        self.assertEqual(prefix_suffix_match(["abc"], 1), [])
        self.assertEqual(prefix_suffix_match(["abc"], 3), [])
    
    @timeout()
    def test_no_matches(self):
        self.assertEqual(prefix_suffix_match(["abc", "def"], 1), [])
        self.assertEqual(prefix_suffix_match(["abc", "def"], 3), [])
    
    @timeout()
    def test_single_match(self):
        self.assertEqual(prefix_suffix_match(["abc", "cde"], 1), [(1, 0)])
    
    @timeout()
    def test_symetric_match(self):
        self.assertEqual(prefix_suffix_match(["aa", "aa"], 1), [(0, 1), (1, 0)])
        self.assertEqual(prefix_suffix_match(["aa", "aa"], 2), [(0, 1), (1, 0)])
        
    @timeout()
    def test_multiple_matches(self):
        self.assertEqual(prefix_suffix_match(["aaa", "cba", "baa"], 2), [(0, 2), (2, 1)])
        self.assertEqual(prefix_suffix_match(["abc", "bc", "c"], 1), [(2, 0), (2, 1)])

    @timeout()
    def test_empty_string_no_match(self):
        self.assertEqual(prefix_suffix_match(["", "abc"], 1), [])
        
    @timeout()
    def test_mix_empty_string_match(self):
        self.assertEqual(prefix_suffix_match(["", "abc", "", "cba"], 1), [(1, 3), (3, 1)])
        
    @timeout()
    def test_k_greater_than_string_length(self):
        self.assertEqual(prefix_suffix_match(["abc", "abc"], 4), [])

    @timeout()
    def test_special_characters(self):
        self.assertEqual(prefix_suffix_match(["ab#c", "#cde"], 2), [(1, 0)])
        self.assertEqual(prefix_suffix_match(["a!c", "b!c"], 2), [])

    @timeout()
    def test_strings_with_spaces(self):
        self.assertEqual(prefix_suffix_match(["abc ", " cde"], 1), [(1, 0)])
        self.assertEqual(prefix_suffix_match(["abc ", "c de"], 2), [(1, 0)])
    
    @timeout()
    def test_all_identical_strings(self):
        self.assertEqual(prefix_suffix_match(["aaa", "aaa", "aaa"], 2), [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
        
    @timeout()
    def test_case_sensitivity(self): 
        self.assertEqual(prefix_suffix_match(["Ab", "Ba"], 1), [])
    
class TestGeneratedSolution21(BaseTestCase):
    rotate_matrix_clockwise = imported_functions[20]
    
    @timeout()
    def test_empty_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([]), [])
    
    @timeout()
    def test_single_item_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1]]), [[1]])
        
    @timeout()
    def test_two_by_two_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1, 2], [3, 4]]), [[3, 1], [4, 2]])
    
    @timeout()
    def test_three_by_three_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[7, 4, 1], [8, 5, 2], [9, 6, 3]])
    
    @timeout()
    def test_four_by_four_matrix(self):
        self.assertEqual(rotate_matrix_clockwise([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), 
                         [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]])
    
    @timeout()
    def test_matrix_with_negative_numbers(self):
        self.assertEqual(rotate_matrix_clockwise([[1, -2, 3], [-4, 5, -6], [7, -8, 9]]), [[7, -4, 1], [-8, 5, -2], [9, -6, 3]])
    
class TestGeneratedSolution22(BaseTestCase):
    cyclic_shift = imported_functions[21]
    
    @timeout()
    def test_empty_list(self):
        self.assertEqual(cyclic_shift([], 'R', 2), [])    
        
    @timeout()
    def test_single_item_list(self):
        self.assertEqual(cyclic_shift([1], 'R', 1), [1])
        self.assertEqual(cyclic_shift([1], 'L', 1), [1])
        
    @timeout()
    def test_two_item_list(self):
        self.assertEqual(cyclic_shift([1, 2], 'R', 1), [2, 1])
        self.assertEqual(cyclic_shift([1, 2], 'L', 1), [2, 1])
        self.assertEqual(cyclic_shift([1, 2], 'R', 2), [1, 2])
        self.assertEqual(cyclic_shift([1, 2], 'L', 2), [1, 2])
        
    @timeout()
    def test_three_item_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 1), [3, 1, 2])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 1), [2, 3, 1])
    
    @timeout()
    def test_shift_larger_than_length_of_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 4), [3, 1, 2])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 4), [2, 3, 1])
        
    @timeout()
    def test_shift_negative(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', -1), [2, 3, 1])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', -1), [3, 1, 2])
          
    @timeout()
    def test_shift_negative_larger_than_length_of_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', -4), [3, 1, 2])
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', -4), [2, 3, 1])
    
    @timeout()
    def test_shift_zero(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 0), [1, 2, 3])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 0), [1, 2, 3])
    
    @timeout()
    def test_shift_equal_to_length_of_list(self):
        self.assertEqual(cyclic_shift([1, 2, 3], 'R', 3), [1, 2, 3])
        self.assertEqual(cyclic_shift([1, 2, 3], 'L', 3), [1, 2, 3])
        
class TestGeneratedSolution23(BaseTestCase):
    encode_string = imported_functions[22]         
    
    @timeout()
    def test_empty_string(self):
        self.assertEqual(encode_string(""), "")
        
    @timeout()
    def test_string_no_repetitions(self):
        self.assertEqual(encode_string("abc"), "1[a]1[b]1[c]")
    
    @timeout()
    def test_string_with_repetitions(self):
        self.assertEqual(encode_string("aabbcc"), "2[a]2[b]2[c]")
        self.assertEqual(encode_string("aaa"), "3[a]")
        self.assertEqual(encode_string("abbcdbaaa"), "1[a]2[b]1[c]1[d]1[b]3[a]")
    
    @timeout()
    def test_string_with_numbers(self):
        self.assertEqual(encode_string("a222bb"), "1[a]3[2]2[b]")
    
    @timeout()
    def test_string_with_special_characters(self):
        self.assertEqual(encode_string("a##b$"), "1[a]2[#]1[b]1[$]")
    
    @timeout()
    def test_string_with_spaces(self):
        self.assertEqual(encode_string("a   b c"), "1[a]3[ ]1[b]1[ ]1[c]")
    
    @timeout()
    def test_very_long_string(self):
        long_string = "a" * 1000 + "b" * 1000 + "c" * 1000
        expected_result = "1000[a]1000[b]1000[c]"
        self.assertEqual(encode_string(long_string), expected_result)
    
    @timeout()
    def test_single_character_string(self):
        self.assertEqual(encode_string("a"), "1[a]")
        self.assertEqual(encode_string(" "), "1[ ]")
        self.assertEqual(encode_string("#"), "1[#]")    

class TestGeneratedSolution24(BaseTestCase):
    list_sums = imported_functions[23]
    
    @timeout()
    def test_empty_list(self):
        self.assertIsNone(list_sums([]))
    
    @timeout()
    def test_single_item_list(self):
        single_item_list = [1] 
        list_sums(single_item_list)
        self.assertEqual(single_item_list, [1])
        
    @timeout()
    def test_multiple_item_list(self):
        multiple_item_list = [1, 2, 3, 4, 5]
        list_sums(multiple_item_list)
        self.assertEqual(multiple_item_list, [1, 3, 6, 10, 15])
    
    @timeout()
    def test_list_with_negative_numbers(self):
        negative_numbers_list = [1, -2, 3, -4, 5]
        list_sums(negative_numbers_list)
        self.assertEqual(negative_numbers_list, [1, -1, 2, -2, 3])
    
    @timeout()
    def test_list_with_zeros(self):
        zeros_list = [0, 0, 0, 0, 0]
        list_sums(zeros_list)
        self.assertEqual(zeros_list, [0, 0, 0, 0, 0])
        
    @timeout()
    def test_list_with_repeated_numbers(self):
        repeated_numbers_list = [1, 1, 1, 1, 1]
        list_sums(repeated_numbers_list)
        self.assertEqual(repeated_numbers_list, [1, 2, 3, 4, 5])
    
    @timeout()
    def test_list_with_floats(self):
        floats_list = [1.5, 2.5, 3.5, 4.5, 5.5]
        list_sums(floats_list)
        self.assertEqual(floats_list, [1.5, 4.0, 7.5, 12.0, 17.5])
        
    @timeout()
    def test_multiple_calls(self):
        multiple_calls_list = [1, 2, 3, 4, 5]
        list_sums(multiple_calls_list)
        list_sums(multiple_calls_list)
        self.assertEqual(multiple_calls_list, [1, 4, 10, 20, 35])
    
class TestGeneratedSolution25(BaseTestCase):
    convert_base = imported_functions[24]
    
    @timeout()
    def test_base_2(self):
        self.assertEqual(convert_base(10, 2), "1010")
        self.assertEqual(convert_base(15, 2), "1111")
        self.assertEqual(convert_base(255, 2), "11111111")

    @timeout()
    def test_base_8(self):
        self.assertEqual(convert_base(10, 8), "12")
        self.assertEqual(convert_base(15, 8), "17")
        self.assertEqual(convert_base(255, 8), "377")    
    
    @timeout()
    def test_unaric_base(self):
        self.assertEqual(convert_base(10, 1), "1" * 10)
        self.assertEqual(convert_base(15, 1), "1" * 15)
        
    @timeout()
    def test_base_5(self):
        self.assertEqual(convert_base(80, 5), "310")
        
    @timeout()
    def test_zero(self):
        self.assertEqual(convert_base(0, 2), "0")
        self.assertEqual(convert_base(0, 8), "0")
        self.assertEqual(convert_base(0, 5), "0")
    
    @timeout()
    def test_zero_unaric_base(self):
        self.assertEqual(convert_base(0, 1), "")
        
    @timeout()
    def test_negative_number(self):
        self.assertIsNone(convert_base(-10, 2))
    
    @timeout()
    def test_negative_base(self):
        self.assertIsNone(convert_base(10, -2))
        
    @timeout()
    def test_base_out_of_range(self):
        self.assertIsNone(convert_base(10, 37))
        self.assertIsNone(convert_base(10, 10))
        self.assertIsNone(convert_base(10, 0))
        
    @timeout()
    def test_large_number(self):
        self.assertEqual(convert_base(1024, 2), "10000000000")
        self.assertEqual(convert_base(1024, 8), "2000")
        self.assertEqual(convert_base(1024, 5), "13044")
    
class TestGeneratedSolution26(BaseTestCase):
    max_div_seq = imported_functions[25]
    
    @timeout()
    def test_div_seq_length_1(self):
        self.assertEqual(max_div_seq(123456, 3), 1)
        self.assertEqual(max_div_seq(123456, 5), 1)
    
    @timeout()
    def test_n_is_single_digit_and_divisable_by_k(self):
        self.assertEqual(max_div_seq(6, 2), 1)
    
    @timeout()
    def test_n_is_single_digit_and_not_divisable_by_k(self):
        self.assertEqual(max_div_seq(7, 2), 0)
    
    @timeout()
    def test_dev_seq_greater_than_1(self):
        self.assertEqual(max_div_seq(124568633, 2), 3)
    
    @timeout()
    def test_k_equals_1(self):
        self.assertEqual(max_div_seq(124568633, 1), 9)
        
    @timeout()
    def test_no_digits_divisible_by_k(self):
        self.assertEqual(max_div_seq(123456, 7), 0)
    
class TestGeneratedSolution27(BaseTestCase):
    find_dup = imported_functions[26]
    
    @timeout()
    def test_large_input(self):
        self.assertEqual(find_dup(list(range(1, 10001)) + [5000]), 5000)

    @timeout()
    def test_duplicate_at_end(self):
        self.assertEqual(find_dup([1, 3, 4, 2, 5, 5]), 5)
    
    @timeout()
    def test_duplicate_at_start(self):
        self.assertEqual(find_dup([1, 1, 2, 3, 4, 5]), 1)
    
    @timeout()
    def test_duplicate_in_middle(self):
        self.assertEqual(find_dup([1, 2, 3, 4, 3, 5, 6]), 3)
    
    @timeout()
    def test_two_elements(self):
        self.assertEqual(find_dup([1, 1]), 1)
    
class TestGeneratedSolution28(BaseTestCase):
    lcm = imported_functions[27]
    
    @timeout()
    def test_basic_cases(self):
        self.assertEqual(lcm(3, 5), 15)
        self.assertEqual(lcm(4, 6), 12)

    @timeout()
    def test_one_is_multiple_of_other(self):
        self.assertEqual(lcm(6, 3), 6)
        self.assertEqual(lcm(10, 5), 10)
    
    @timeout()
    def test_prime_numbers(self):
        self.assertEqual(lcm(7, 11), 7 *11)
        self.assertEqual(lcm(13, 17), 13 * 17)

    @timeout()
    def test_large_numbers_and_lcm_is_not_their_product(self):
        self.assertEqual(lcm(123456, 789012), 8117355456)

    @timeout()
    def test_one_is_one(self):
        self.assertEqual(lcm(1, 99), 99)
        self.assertEqual(lcm(1, 1), 1)
    
    @timeout()
    def test_equal_numbers(self):
        self.assertEqual(lcm(8, 8), 8)

    @timeout()
    def test_coprime_numbers(self):
        self.assertEqual(lcm(9, 14), 9 * 14)
        self.assertEqual(lcm(15, 22), 15 * 22)
    
    @timeout()
    def test_numbers_with_common_factors(self):
        self.assertEqual(lcm(18, 24), 72)
        self.assertEqual(lcm(40, 60), 120)
    
class TestGeneratedSolution29(BaseTestCase):
    f19 = imported_functions[28]
    
    @timeout()
    def test_smallest_valid_number(self):
        self.assertEqual(f19(), 2235)
    
    @timeout()
    def test_product_of_digits_within_range(self):
        result = f19()
        digits = [int(digit) for digit in str(result)]
        product_of_digits = 1
        for digit in digits:
            product_of_digits *= digit
        self.assertTrue(56 <= product_of_digits <= 64)
    
    @timeout()
    def test_divisibility_by_15(self):
        result = f19()
        self.assertEqual(result % 15, 0)
    
    @timeout()
    def test_is_four_digits(self):
        result = f19()
        self.assertTrue(1000 <= result <= 9999)    
    
class TestGeneratedSolution30(BaseTestCase):
    f20 = imported_functions[29]
    
    @timeout()
    def test_expected_output(self):
        self.assertEqual(f20(), 14674)

    @timeout()
    def test_divisibility_by_22(self):
        result = f20()
        self.assertIsNotNone(result)
        self.assertEqual(result % 22, 0)

    @timeout()
    def test_length_of_result(self):
        result = f20()
        self.assertIsNotNone(result)
        self.assertEqual(len(str(result)), len(str(14563743)) - 3)

    @timeout()
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
    
    @timeout()
    def test_basic_convolution(self):
        signal = np.array([1, 2, 3, 4, 5])
        kernel = np.array([0.2, 0.5, 0.2])
        expected_result = np.array([0.2, 0.9, 1.8, 2.7, 3.6, 3.3, 1.0])
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))

    @timeout()
    def test_zero_kernel(self):
        signal = np.array([1, 2, 3, 4, 5])
        kernel = np.array([0])
        expected_result = np.zeros(len(signal))
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))

    @timeout()
    def test_identity_kernel(self):
        signal = np.array([1, 2, 3, 4, 5])
        kernel = np.array([1])
        expected_result = np.array([1, 2, 3, 4, 5])
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))

    @timeout()
    def test_unity_signal(self):
        signal = np.ones(5)
        kernel = np.array([0.2, 0.5, 0.2])
        expected_result = np.array([0.2, 0.7, 0.9, 0.9, 0.9, 0.7, 0.2])
        convolved_signal = convolve_1d(signal, kernel)
        self.assertTrue(np.allclose(convolved_signal, expected_result))    
    
class TestGeneratedSolution32(BaseTestCase):
    mask_n = imported_functions[31]
    
    @timeout()
    def test_idx_zero(self):
        im = np.array([[3, 5, 9], [8, 1, 2], [7, 6, 4]])
        expected_mask = np.array([[True, False, False], [False, True, True], [False, False, False]])
        self.assertTrue(np.all(mask_n(im, 3, 0) == expected_mask))

    @timeout()
    def test_idx_one(self):
        im = np.array([[3, 5, 9], [8, 1, 2], [7, 6, 4]])
        expected_mask = np.array([[False, True, False], [False, False, False], [False, True, True]])
        self.assertTrue(np.all(mask_n(im, 3, 1) == expected_mask))
        
    @timeout()
    def test_n_equals_1(self):
        im = np.array([[3, 5, 9], [8, 1, 2], [7, 6, 4]])
        expected_mask = np.array([[False, False, True], [False, False, False], [False, False, False]])
        self.assertTrue(np.all(mask_n(im, 1, 1) == expected_mask))
    
    @timeout()
    def test_same_pixels_image(self):
        im = np.ones((3, 3))
        expected_mask = np.array([[True, True, True], [True, True, True], [True, True, True]])
        self.assertTrue(np.all(mask_n(im, 3, 1) == expected_mask))
        
    @timeout()
    def test_large_differece_between_pixels(self):
        im = np.array([[1, 100, 1], [100, 1, 100], [1, 100, 1]])
        expected_mask = np.array([[False, False, False], [False, False, False], [False, False, False]])
        self.assertTrue(np.all(mask_n(im, 3, 1) == expected_mask))
    
class TestGeneratedSolution33(BaseTestCase):
    entropy = imported_functions[32]

    @timeout()
    def test_single_value_matrix(self):
        mat = np.array([[5]])
        self.assertAlmostEqual(entropy(mat), 0.0)

    @timeout()
    def test_all_same_values(self):
        mat = np.array([[1, 1], [1, 1]])
        self.assertAlmostEqual(entropy(mat), 0.0)

    @timeout()
    def test_uniform_distribution(self):
        mat = np.array([[1, 2], [2, 1]])
        self.assertAlmostEqual(entropy(mat), 1.0)

    @timeout()
    def test_non_uniform_distribution(self):
        mat = np.array([[3, 5, 3], [8, 1, 1], [7, 7, 7]])
        self.assertAlmostEqual(entropy(mat), 2.197159723424149)

    @timeout()
    def test_small_matrix(self):
        mat = np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(entropy(mat), 1.0)

    @timeout()
    def test_unique_values(self):
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # Entropy for a matrix with all unique values should be log2(9)
        self.assertAlmostEqual(entropy(mat), np.log2(9))

    @timeout()
    def test_matrix_with_zeros(self):
        mat = np.array([[0, 0, 0], [1, 1, 1]])
        # Probability: 3/6 for 0 and 3/6 for 1
        expected_entropy = -(0.5 * np.log2(0.5) + 0.5 * np.log2(0.5))
        self.assertAlmostEqual(entropy(mat), expected_entropy)

    @timeout()
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

    @timeout()
    def test_squeeze_vertical_basic(self):
        im = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        factor = 2
        expected = np.array([[2, 3], [6, 7]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    @timeout()
    def test_squeeze_vertical_whole_image(self):
        im = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        factor = 4
        expected = np.array([[4, 5]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    @timeout()
    def test_squeeze_vertical_no_squeeze(self):
        im = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        factor = 1
        expected = im
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    @timeout()
    def test_squeeze_vertical_different_image(self):
        im = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]])
        factor = 2
        expected = np.array([[25, 35, 45], [85, 95, 105]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    @timeout()
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

    @timeout()
    def test_squeeze_vertical_with_padding(self):
        im = np.array([[1, 2, 5], [3, 4], [5, 6, 8], [7, 8]])
        factor = 2
        expected = np.array([[2.0, 3.0, 2.5], [6.0, 7.0, 4.0]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    @timeout()
    def test_squeeze_vertical_single_row(self):
        im = np.array([[1, 2, 3]])
        factor = 1
        expected = np.array([[1, 2, 3]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)
    
    @timeout()
    def test_squeeze_vertical_single_column(self):
        im = np.array([[1], [2], [3], [4]])
        factor = 2
        expected = np.array([[1.5], [3.5]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)

    @timeout()
    def test_squeeze_vertical_single_element(self):
        im = np.array([[1]])
        factor = 1
        expected = np.array([[1]])
        result = squeeze_vertical(im, factor)
        np.testing.assert_array_equal(result, expected)

class TestGeneratedSolution35(unittest.TestCase):
    denoise = imported_functions[34]
    
    @timeout()
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

    @timeout()
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

    @timeout()
    def test_denoise_single_value(self):
        im = np.array([[100]])
        
        expected_result = np.array([[100]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    @timeout()
    def test_denoise_single_row(self):
        im = np.array([[1, 2, 3, 4, 5]])
        
        expected_result = np.array([[1.5, 2, 3, 4, 4.5]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    @timeout()
    def test_denoise_single_column(self):
        im = np.array([[1], [2], [3], [4], [5]])
        
        expected_result = np.array([[1.5], [2], [3], [4], [4.5]])

        result = denoise(im)
        
        # Check shape
        self.assertEqual(result.shape, expected_result.shape)
        
        # Check values
        np.testing.assert_almost_equal(result, expected_result)

    @timeout()
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

    @timeout()
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

    @timeout()
    def test_denoise_fail_1D_array(self):
        im = np.array([1, 2, 3, 4, 5])
        
        with self.assertRaises(Exception):
            denoise(im)
        

class TestGeneratedSolution36(BaseTestCase):
    calculate_monthly_sales = imported_functions[35]
    
    @timeout()
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
    
    @timeout()
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
    
    @timeout()
    def test_no_sales(self):
        data = pd.DataFrame({
            'Date': [],
            'Product': [],
            'Sales': []
        })
        expected = pd.DataFrame(columns=['Product', 'YearMonth', 'Sales', 'AverageMonthlySales'])
        result = calculate_monthly_sales(data)
        pd.testing.assert_frame_equal(result, expected)

    @timeout()
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
    
    @timeout()
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

    @timeout()
    def test_no_matching_titles(self):
        search_title = 'Nonexistent Movie'
        expected = pd.DataFrame(columns=['id', 'title', 'rate', 'runtime'])
        result = recommendations(self.movies, self.movies_genres, self.genres, search_title)
        pd.testing.assert_frame_equal(result, expected)
    
    @timeout()
    def test_no_similar_movies(self):
        search_title = 'Interstellar'
        expected = pd.DataFrame(columns=['id', 'title', 'rate', 'runtime'])
        result = recommendations(self.movies, self.movies_genres, self.genres, search_title)
        # Check if the shapes match
        self.assertEqual(result.shape, expected.shape)
        # Check if both DataFrames are empty
        self.assertTrue(result.empty)
        self.assertTrue(expected.empty)

    @timeout()
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

    @timeout()
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

    @timeout()
    def test_basic(self):
        expected = pd.DataFrame({
            'department_name': ['Engineering', 'HR', 'Sales'],
            'total_hours': [350, 200, 100]
        })
        result = top_hours_worked_departments(self.employees, self.departments, self.works_on)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    @timeout()
    def test_no_employees(self):
        employees_empty = pd.DataFrame(columns=['employee_id', 'name', 'department_id', 'salary'])
        expected = pd.DataFrame(columns=['department_name', 'total_hours'])
        result = top_hours_worked_departments(employees_empty, self.departments, self.works_on)
        pd.testing.assert_frame_equal(result, expected)

    @timeout()
    def test_no_projects(self):
        works_on_empty = pd.DataFrame(columns=['employee_id', 'project_id', 'hours_worked'])
        expected = pd.DataFrame(columns=['department_name', 'total_hours'])
        result = top_hours_worked_departments(self.employees, self.departments, works_on_empty)
        pd.testing.assert_frame_equal(result, expected)

    @timeout()
    def test_no_departments(self):
        departments_empty = pd.DataFrame(columns=['department_id', 'name'])
        expected = pd.DataFrame(columns=['department_name', 'total_hours'])
        result = top_hours_worked_departments(self.employees, departments_empty, self.works_on)
        pd.testing.assert_frame_equal(result, expected)

    @timeout()
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
        
    @timeout()
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

    @timeout()
    def test_basic(self):
        expected = pd.DataFrame({
            'name': ['B', 'D'],
            'population': [2000, 700],
            'border_population_sum': [1500, 300]
        })
        result = huge_population_countries(self.countries, self.borders)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    @timeout()
    def test_no_countries(self):
        countries = pd.DataFrame(columns=['name', 'population'])
        borders = pd.DataFrame({
            'country1': ['A', 'A', 'B'],
            'country2': ['B', 'C', 'C']
        })
        expected = pd.DataFrame(columns=['name', 'population', 'border_population_sum'])
        result = huge_population_countries(countries, borders)
        pd.testing.assert_frame_equal(result, expected)

    @timeout()
    def test_no_borders(self):
        countries = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'population': [1000, 2000, 500]
        })
        borders = pd.DataFrame(columns=['country1', 'country2'])
        expected = pd.DataFrame(columns=['name', 'population', 'border_population_sum'])
        result = huge_population_countries(countries, borders)
        pd.testing.assert_frame_equal(result, expected)

    @timeout()
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
    
    @timeout()
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

    @timeout()
    def test_countries_bordering_most_populated_in_asia(self):
        expected_result = ['China', 'Nepal', 'Pakistan']
        result = countries_bordering_most_populated_in_asia(self.country_df, self.border_df)
        self.assertEqual(result, expected_result)

    @timeout()
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

    @timeout()
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

    @timeout()
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

    @timeout()
    def test_get_given_attributes(self):
        self.assertEqual(self.tri.get('a'), 10)
        self.assertEqual(self.tri.get('b'), 5)
        self.assertEqual(self.tri.get('ab'), 90)
        self.assertEqual(self.tri.get('color'), 'black')

    @timeout()
    def test_get_implicit_attributes(self):
        self.assertAlmostEqual(self.tri.get('c'), 11.180339887498949)
        self.assertAlmostEqual(self.tri.get('ac'), 26.565051177077994)
        self.assertAlmostEqual(self.tri.get('bc'), 63.43494882292201)

    @timeout()
    def test_get_invalid_edge(self):
        with self.assertRaises(KeyError):
            self.tri.get('d')

    @timeout()
    def test_get_invalid_angle(self):
        with self.assertRaises(KeyError):
            self.tri.get('cd')
    
    @timeout()
    def test_get_invalid_attribute(self):
        with self.assertRaises(KeyError):
            self.tri.get('hello')

    @timeout()
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

    @timeout()
    def test_get_full_name_basic(self):
        self.assertEqual(self.worker.getFullName(), "Jon Cohen")
    
    @timeout()
    def test_get_full_name_with_second_name(self):
        self.assertEqual(self.worker_with_second_name.getFullName(), "David Julius Cohen")

    @timeout()
    def test_get_job(self):
        self.assertEqual(self.worker.getJob(), "Salesman")

    @timeout()
    def test_get_salary(self):
        self.assertEqual(self.worker.getSalary(), 5000)

    @timeout()
    def test_update_job_and_salary(self):
        self.worker.update(job='Engineer', salary=7000)
        self.assertEqual(self.worker.getJob(), "Engineer")
        self.assertEqual(self.worker.getSalary(), 7000)

    @timeout()
    def test_update_salary_only(self):
        self.worker.update(salary=9000)
        self.assertEqual(self.worker.getSalary(), 9000)
        self.assertEqual(self.worker.getJob(), "Salesman")  # Job should remain unchanged
        
    @timeout()
    def test_update_job_only(self):
        self.worker.update(job='Manager')
        self.assertEqual(self.worker.getJob(), "Manager")
        self.assertEqual(self.worker.getSalary(), 5000)

class TestGeneratedSolution43(BaseTestCase):
    binaric_arithmatic = imported_classes[2]
    
    def setUp(self):
        # Creating an instance for testing
        self.zero = self.binaric_arithmatic("0")
        self.one = self.binaric_arithmatic("1")
        self.seven = self.binaric_arithmatic("111")
        
    @timeout()
    def test_inc_positive(self):
        self.assertEqual(self.seven.inc(), "1000")
        self.assertEqual(self.one.inc(), "10")
    
    @timeout()
    def test_inc_zero(self):    
        self.assertEqual(self.zero.inc(), "1")

    @timeout()
    def test_dec(self):
        self.assertEqual(self.seven.dec(), "110")
        self.assertEqual(self.one.dec(), "0")
    
    @timeout()
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

    @timeout()
    def test_repr(self):
        self.assertEqual(repr(self.a), "Point(1, 1)")

    @timeout()
    def test_eq(self):
        self.assertTrue(self.a == self.d)
        self.assertFalse(self.a == self.b)

    @timeout()
    def test_add(self):
        self.assertEqual(self.a + self.b, self.a_plus_b)

    @timeout()
    def test_sub(self):
        self.assertEqual(self.a - self.b, self.a_minus_b)

    @timeout()
    def test_distance(self):
        self.assertEqual(self.a.distance(self.c), 2)
        self.assertAlmostEqual(self.a.distance(self.b), math.sqrt(1))
        self.assertEqual(self.a.distance(self.d), 0)

    @timeout()
    def test_angle_wrt_origin(self):
        self.assertAlmostEqual(self.b.angle_wrt_origin(self.c), math.pi / 4)
        self.assertAlmostEqual(self.c.angle_wrt_origin(self.b), 2 * math.pi - math.pi / 4)

class TestGeneratedSolution45(BaseTestCase):
    Roulette = imported_classes[4]

    def setUp(self):
        # Create a Roulette instance with an initial balance
        self.gambler = self.Roulette(1000)

    @timeout()
    def test_initial_balance(self):
        self.assertEqual(self.gambler.get_balance(), 1000)

    @patch('random.randint')
    @timeout()
    def test_bet_red_win(self, mock_randint):
        mock_randint.return_value = 1  # A red number
        self.gambler.bet(100, "red")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    @timeout()
    def test_bet_red_lose(self, mock_randint):
        mock_randint.return_value = 2  # A black number
        self.gambler.bet(100, "red")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    @timeout()
    def test_bet_black_win(self, mock_randint):
        mock_randint.return_value = 2  # A black number
        self.gambler.bet(100, "black")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    @timeout()
    def test_bet_black_lose(self, mock_randint):
        mock_randint.return_value = 1  # A red number
        self.gambler.bet(100, "black")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    @timeout()
    def test_bet_even_win(self, mock_randint):
        mock_randint.return_value = 2  # An even number
        self.gambler.bet(100, "even")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    @timeout()
    def test_bet_even_lose(self, mock_randint):
        mock_randint.return_value = 1  # An odd number
        self.gambler.bet(100, "even")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    @timeout()
    def test_bet_odd_win(self, mock_randint):
        mock_randint.return_value = 1  # An odd number
        self.gambler.bet(100, "odd")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    @timeout()
    def test_bet_odd_lose(self, mock_randint):
        mock_randint.return_value = 2  # An even number
        self.gambler.bet(100, "odd")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    @timeout()
    def test_bet_range_1_12_win(self, mock_randint):
        mock_randint.return_value = 5  # A number in range 1-12
        self.gambler.bet(100, "1-12")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    @timeout()
    def test_bet_range_1_12_lose(self, mock_randint):
        mock_randint.return_value = 13  # A number not in range 1-12
        self.gambler.bet(100, "1-12")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    @timeout()
    def test_bet_range_13_24_win(self, mock_randint):
        mock_randint.return_value = 15  # A number in range 13-24
        self.gambler.bet(100, "13-24")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    @timeout()
    def test_bet_range_13_24_lose(self, mock_randint):
        mock_randint.return_value = 25  # A number not in range 13-24
        self.gambler.bet(100, "13-24")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    @timeout()
    def test_bet_range_25_36_win(self, mock_randint):
        mock_randint.return_value = 30  # A number in range 25-36
        self.gambler.bet(100, "25-36")
        self.assertEqual(self.gambler.get_balance(), 1100)

    @patch('random.randint')
    @timeout()
    def test_bet_range_25_36_lose(self, mock_randint):
        mock_randint.return_value = 24  # A number not in range 25-36
        self.gambler.bet(100, "25-36")
        self.assertEqual(self.gambler.get_balance(), 900)

    @patch('random.randint')
    @timeout()
    def test_bet_specific_number_win(self, mock_randint):
        mock_randint.return_value = 17  # Specific number bet
        self.gambler.bet(100, "17")
        self.assertEqual(self.gambler.get_balance(), 4500)

    @patch('random.randint')
    @timeout()
    def test_bet_specific_number_lose(self, mock_randint):
        mock_randint.return_value = 18  # Not the specific number
        self.gambler.bet(100, "17")
        self.assertEqual(self.gambler.get_balance(), 900)

    @timeout()
    def test_bet_insufficient_balance(self):
        with self.assertRaises(KeyError):
            self.gambler.bet(2000, "red")

class TestGeneratedSolution46(BaseTestCase):
    investments = imported_classes[5]
    def setUp(self):
        self.jon = self.investments("jon", 100000, 10, 15000, 10000)
    
    @timeout()
    def test_initial_balance(self):
        self.assertEqual(self.jon.get_balance(), 100000)
    
    @timeout()
    def test_future_value_after_3_years(self):
        self.assertAlmostEqual(self.jon.get_future_value(3), 351560, places=0)

    @timeout()
    def test_future_value_balance_stays_the_same(self):
        self.assertEqual(self.jon.get_future_value(3), 351560)
        self.assertEqual(self.jon.get_balance(), 100000)
       
    @timeout()
    def test_withdraw_100000(self):
        self.jon.update_value_by_year(3)  # Update balance before withdrawal
        self.assertAlmostEqual(self.jon.withdraw(100000), 251560, places=0)
    
    @timeout()
    def test_future_value_equals_updated_value(self):
        predicted_value = self.jon.get_future_value(4)
        self.jon.update_value_by_year(4)
        self.assertEqual(predicted_value, self.jon.get_balance())

    @timeout()
    def test_withdraw_more_than_balance(self):
        with self.assertRaises(KeyError):
            self.jon.withdraw(200000)

    @timeout()
    def test_repr(self):
        self.assertEqual(repr(self.jon), "name: jon \nbalance: 100000\navg_yearly_return: 10\nmonthly_income: 15000\nmonthly_expenses: 10000")

    @timeout()
    def test_repr_after_update(self):
        self.jon.withdraw(5000)
        self.assertEqual(repr(self.jon), "name: jon \nbalance: 95000\navg_yearly_return: 10\nmonthly_income: 15000\nmonthly_expenses: 10000")

class TestGeneratedSolution47(BaseTestCase):
    Restaurant = imported_classes[6]

    def setUp(self):
        self.restaurant = self.Restaurant("Ragazzo", "Italian", 4.5)

    @timeout()
    def test_initialization(self):
        self.assertEqual(self.restaurant.name, "Ragazzo")
        self.assertEqual(self.restaurant.cuisine, "Italian")
        self.assertEqual(self.restaurant.rating, 4.5)
        self.assertEqual(self.restaurant.menu, {})
        self.assertEqual(self.restaurant.chefs, [])

    @timeout()
    def test_repr(self):
        self.assertEqual(repr(self.restaurant), "Ragazzo (Italian) - 4.5/5")

    @timeout()
    def test_add_dish(self):
        self.restaurant.add_dish("pasta", 10)
        self.assertEqual(self.restaurant.menu, {"pasta": 10})
        self.restaurant.add_dish("pizza", 20)
        self.assertEqual(self.restaurant.menu, {"pasta": 10, "pizza": 20})

    @timeout()
    def test_remove_dish(self):
        self.restaurant.add_dish("pasta", 10)
        self.restaurant.add_dish("pizza", 20)
        self.restaurant.remove_dish("pasta")
        self.assertEqual(self.restaurant.menu, {"pizza": 20})
        self.restaurant.remove_dish("burger")  # Trying to remove a non-existent dish
        self.assertEqual(self.restaurant.menu, {"pizza": 20})

    @timeout()
    def test_add_chef(self):
        self.restaurant.add_chef("Mario")
        self.assertEqual(self.restaurant.chefs, ["Mario"])
        self.restaurant.add_chef("Luigi")
        self.assertEqual(self.restaurant.chefs, ["Mario", "Luigi"])

    @timeout()
    def test_remove_chef(self):
        self.restaurant.add_chef("Mario")
        self.restaurant.add_chef("Luigi")
        self.restaurant.remove_chef("Mario")
        self.assertEqual(self.restaurant.chefs, ["Luigi"])
        self.restaurant.remove_chef("Peach")  # Trying to remove a non-existent chef
        self.assertEqual(self.restaurant.chefs, ["Luigi"])

    @timeout()
    def test_get_menu(self):
        self.restaurant.add_dish("pasta", 10)
        self.restaurant.add_dish("pizza", 20)
        self.assertEqual(self.restaurant.get_menu(), {"pasta": 10, "pizza": 20})

    @timeout()
    def test_get_chefs(self):
        self.restaurant.add_chef("Mario")
        self.restaurant.add_chef("Luigi")
        self.assertEqual(self.restaurant.get_chefs(), ["Mario", "Luigi"])

class TestGeneratedSolution48(BaseTestCase):
    Polynomial = imported_classes[7]
    
    def setUp(self):
        self.p1 = self.Polynomial([1, 2, 0, 4])
        self.p2 = self.Polynomial([0, 2, -5, 0])
        self.p3 = self.Polynomial([-7, 2, 0, 4])
        self.p4 = self.Polynomial([-6, -2, 0, 4, 5])
        self.p5 = self.Polynomial([0])

    @timeout()
    def test_initialization(self):
        self.assertEqual(self.p1.coeffs, [1, 2, 0, 4])
        self.assertEqual(self.p2.coeffs, [0, 2, -5, 0])
        self.assertEqual(self.p3.coeffs, [-7, 2, 0, 4])
        self.assertEqual(self.p4.coeffs, [-6, -2, 0, 4, 5])
        self.assertEqual(self.p5.coeffs, [0])

    @timeout()
    def test_repr(self):
        self.assertEqual(str(self.p1), "1 + 2x + 4x^3")
        self.assertEqual(str(self.p2), "2x - 5x^2")
        self.assertEqual(str(self.p3), "-7 + 2x + 4x^3")
        self.assertEqual(str(self.p4), "-6 - 2x + 4x^3 + 5x^4")
        self.assertEqual(str(self.p5), "0")

    @timeout()
    def test_get_deg(self):
        self.assertEqual(self.p1.get_deg(), 3)
        self.assertEqual(self.p2.get_deg(), 3)
        self.assertEqual(self.p3.get_deg(), 3)
        self.assertEqual(self.p4.get_deg(), 4)
        self.assertEqual(self.p5.get_deg(), 0)

    @timeout()
    def test_add(self):
        p1_plus_p2 = self.p1 + self.p2
        p3_plus_p4 = self.p3 + self.p4
        p2_plus_p5 = self.p2 + self.p5

        self.assertEqual(str(p1_plus_p2), "1 + 4x - 5x^2 + 4x^3")
        self.assertEqual(str(p3_plus_p4), "-13 + 8x^3 + 5x^4")
        self.assertEqual(p2_plus_p5, self.p2)

    @timeout()
    def test_eq(self):
        self.assertTrue(self.p1 == self.Polynomial([1, 2, 0, 4]))
        self.assertFalse(self.p1 == self.p2)
        self.assertTrue(self.p2 == self.Polynomial([0, 2, -5, 0]))
        self.assertFalse(self.p3 == self.p4)
        
class TestGeneratedSolution49(BaseTestCase):
    ToDoList = imported_classes[8]
    
    def setUp(self):
        self.todo_list = self.ToDoList()
        self.todo_list.add_task("Buy groceries")
        self.todo_list.add_task("Go to school")
        self.todo_list.add_task("Do HW")

    @timeout()
    def test_initialization(self):
        empty_list = self.ToDoList()
        self.assertEqual(empty_list.tasks, [])

    @timeout()
    def test_add_task(self):
        self.todo_list.add_task("Read book")
        self.assertIn({'task': "Read book", 'completed': False}, self.todo_list.tasks)

    @timeout()
    def test_remove_task(self):
        result = self.todo_list.remove_task("Do HW")
        self.assertTrue(result)
        self.assertNotIn({'task': "Do HW", 'completed': False}, self.todo_list.tasks)
        result = self.todo_list.remove_task("Nonexistent task")
        self.assertFalse(result)

    @timeout()
    def test_mark_completed(self):
        result = self.todo_list.mark_completed("Buy groceries")
        self.assertTrue(result)
        self.assertIn({'task': "Buy groceries", 'completed': True}, self.todo_list.tasks)
        result = self.todo_list.mark_completed("Nonexistent task")
        self.assertFalse(result)

    @timeout()
    def test_list_tasks(self):
        self.assertEqual(self.todo_list.list_tasks(), ["Buy groceries", "Go to school", "Do HW"])
        self.todo_list.mark_completed("Buy groceries")
        self.assertEqual(self.todo_list.list_tasks(completed=True), ["Buy groceries"])
        self.assertEqual(self.todo_list.list_tasks(completed=False), ["Go to school", "Do HW"])

class TestGeneratedSolution50(BaseTestCase):
    RecipeBook = imported_classes[9]
    
    def setUp(self):
        self.book = self.RecipeBook()
        self.book.add_recipe("Pasta", ["pasta", "tomato sauce", "cheese"], "Cook pasta, add sauce and cheese")
        self.book.add_recipe("Pizza", ["dough", "tomato sauce", "cheese"], "Bake dough, add sauce and cheese")
        self.book.add_recipe("Salad", ["lettuce", "tomato", "cucumber"], "Mix all ingredients")

    @timeout()
    def test_add_recipe(self):
        self.book.add_recipe("Burger", ["burger bun", "beef patty", "lettuce", "tomato", "onion"], "Grill patty, assemble burger")
        self.assertEqual(len(self.book.recipes), 4)

    @timeout()
    def test_remove_recipe(self):
        self.assertTrue(self.book.remove_recipe("Pasta"))
        self.assertEqual(len(self.book.recipes), 2)
        self.assertFalse(self.book.remove_recipe("Nonexistent Recipe"))
        self.assertEqual(len(self.book.recipes), 2)

    @timeout()
    def test_search_by_ingredient(self):
        cheese_recipes = self.book.search_by_ingredient("cheese")
        self.assertEqual(len(cheese_recipes), 2)
        for recipe in cheese_recipes:
            self.assertIn("cheese", recipe["ingredients"])


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
    df = pd.read_csv("unit_test_results.csv")
    df['unit_test_fail_rate_claude'] = result.failure_fractions()
    df.to_csv("unit_test_results.csv", index = False)
    print("SAVED CSV!")
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()

    sys.quit()
    
