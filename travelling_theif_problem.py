import numpy as np

class TTP:
    # Task 2: Initial the solution #
    def __init__(self, distance_matrix, knapsack_capacity, min_speed, max_speed ,profit_list, weight_list, item_location, route, stolen_items):
        self.distance_matrix = distance_matrix
        self.knapsack_capacity = knapsack_capacity
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.profit_list = profit_list
        self.weight_list = weight_list
        self.city_of_item = item_location
        self.number_of_cities = len(self.distance_matrix)

        self.route = route
        self.stolen_items = stolen_items

        self.travelling_time = self.calc_fitness_travelling_time() 
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
        return (self.total_profit >= other.total_profit and self.travelling_time < other.travelling_time) or \
               (self.travelling_time <= other.travelling_time and self.total_profit > other.total_point)
    
    # Task 5: Calculate weight at city i #
    def cal_weight_at_city(self, i):
        pass

    # Task 6: Calculate velocity at city i #
    def cal_velocity_at_city(self, i):
        pass

    # Task 7: Fitness TSP: Travelling time #
    def calc_fitness_travelling_time(self, route=None):
        return 0

    # Task 8: Fitness KP: Total profit #
    def calc_fitness_total_profit(self):
        pass

    # Task 28: Local search TSP #
    def two_opt_change(self, first, second):
        new_route = np.zeros(self.number_of_cities)
        new_route[:first] = self.route[:first]
        new_route[]

    def two_opt(self, best_improvement=False):
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
