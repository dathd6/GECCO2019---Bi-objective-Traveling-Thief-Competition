class TTP:
    # Task 2: Initial the solution #
    def __init__(self, distance_matrix, knapsack_capacity, min_speed, max_speed ,profit_list, weight_list, city_of_item):
        self.distance_matrix = distance_matrix
        self.knapsack_capacity = knapsack_capacity
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.profit_list = profit_list
        self.weight_list = weight_list
        self.city_of_item = city_of_item
        self.number_of_cities = len(self.distance_matrix)

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
    def calc_fitness_travelling_time(self):
        pass

    # Task 8: Fitness KP: Total profit #
    def calc_fitness_total_profit(self):
        pass
