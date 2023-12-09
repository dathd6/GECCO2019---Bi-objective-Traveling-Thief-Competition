import numpy as np

class TTP:
    # Task 2: Initial the solution #
    def __init__(self, distance_matrix, knapsack_capacity, min_speed, max_speed ,profit_list, weight_list, item_location, renting_ratio, route, stolen_items):
        self.distance_matrix = distance_matrix
        self.knapsack_capacity = knapsack_capacity
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.profit_list = profit_list
        self.weight_list = weight_list
        self.item_location = np.array(item_location)
        self.number_of_cities = len(self.distance_matrix)

        self.route = route
        self.stolen_items = stolen_items
        self.renting_ratio = renting_ratio

        self.travelling_time = self.calc_fitness_travelling_time() 
        # self.total_profit = self.calc_fitness_total_profit() - self.travelling_time * self.renting_ratio
        self.total_profit = self.calc_fitness_total_profit()
    
    # Task 19: Compare solution
    def __gt__(self, other):
        # Dominance
        return (self.travelling_time < other.travelling_time) and \
               (self.total_profit > other.total_profit)
    def __eq__(self, other):
        # Equal
        return (self.travelling_time == other.travelling_time) and \
               (self.total_profit == other.total_profit)
    def __ge__(self, other):
        # Weakly dominance
        return self.total_profit >= other.total_profit and self.travelling_time <= other.travelling_time
    
    # Task 5: Calculate weight at city i #
    def cal_weight_at_city(self, i):
        stolen_items = self.stolen_items
        weight = self.weight_list
        total_weight = 0
        item_in_city = np.where(self.item_location == i)[0]
        for j in item_in_city:
            if stolen_items[j]:  #Z[j] denotes take or leave
                total_weight += weight[j]
        return total_weight
        
    # Task 6: Calculate velocity
    def cal_velocity_at_city(self, current_weight):
        # function takes route index, i, and reutrns current velocity at that point in route
        current_capacity = current_weight/self.knapsack_capacity # Calc current capacity
        velocity_reduction = current_capacity*(self.max_speed - self.min_speed) # Calc velocity reduction due to weight
        if current_weight <= self.knapsack_capacity:
            current_velocity = self.max_speed - velocity_reduction # If weight<=capacity return reduced velocity
        else:
            current_velocity = self.min_speed # return min velocity if overcapacity for invalid solutions
        return current_velocity


    # Task 7: Function to calculate fitness: travelling time (t)
    def calc_fitness_travelling_time(self):
        total_time = 0
        curr_weight = 0
        route = self.route  # get the current route
        # Loop over the current route and calculate the total time
        for i in range(len(route) - 1):
            curr_distance = self.distance_matrix[route[i]][route[i + 1]]  # distance between city i and city i+1
            curr_weight += self.cal_weight_at_city(route[i])  # weight of the knapsack at city i
            curr_speed = self.cal_velocity_at_city(curr_weight)  # speed of the vehicle at city i
            total_time += curr_distance / curr_speed  # add the travelling time to the total time

        curr_weight += self.cal_weight_at_city(route[-1])  # weight of the knapsack at city i
        curr_speed = self.cal_velocity_at_city(curr_weight)  # speed of the vehicle at city i
        total_time += self.distance_matrix[route[-1]][route[0]] \
                     / curr_speed  # add the travelling time from the last city to the first city
        return total_time

    # Task 8: Fitness KP: Total profit #
    def calc_fitness_total_profit(self):
        total_profit = 0
        for item,is_stolen in enumerate(self.stolen_items):
            if is_stolen:
                total_profit += self.profit_list[item]
        return total_profit

    # Task 28: Local search TSP #
    def two_opt(self, best_improvement=False):
        new_route = self.route.copy()
        route = self.route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue  # changes nothing, skip then
                    new_route = route[:]    # Creates a copy of route
                    new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2-optSwap since j >= i we use -1
                    if self.calc_fitness_travelling_time(new_route) < self.calc_fitness_travelling_time(route):
                        route = new_route    # change current route to best
                        if not best_improvement:
                            self.route = route
                            return

        self.route = route

    def get_fitness(self):
        return np.array([self.travelling_time, self.total_profit])