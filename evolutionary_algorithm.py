import numpy as np
from constants import TEST_FOLDER

class EA:
    def __init__(self, test_name='test-example-n4.txt') -> None:
        self.population = []
        self.size_p = 0
        self.distance_matrix = [] # 2D Array contains cost of the edges between vertices

        # Task 1: Extract data from test problem file #
        content_list = []
        test_file = open(f"{TEST_FOLDER}/{test_name}")
        for i in test_file :
            content_list.append(i.split())

        self.number_of_cities = int(content_list[2][-1])    # total number of cities
        self.knapsack_capacity = int(content_list[4][-1])  # threshold value
        self.min_speed = float(content_list[5][-1])        # minimum speed
        self.max_speed = float(content_list[6][-1])       # maximum speed
        del content_list[0:10]                     
        node_list = []                            
        for i in range(self.number_of_cities):
            node_list.append([eval(j) for j in content_list[i]])  # list of node's coordinates
        del content_list[0:self.number_of_cities+1]
        
        self.distance_matrix = self.map_node_coord_to_matrix_distance(node_list) # create distance matrix

        self.profit_list = []
        self.weight_list = []
        self.city_of_item = []

        for row in content_list:
            self.profit_list.append(int(row[1]))         #profits of each bags in the nodes 
            self.weight_list.append(int(row[2]))         # weights of individual bags
            self.city_of_item.append(int(row[3]))        # List entail the i item in which city

    # Task 2: Generate initial population
    def generate_initial_population(self, size_p):
        pass

    # Task 4: Map imported data node coord to matrix distance matrix
    def map_node_coord_to_matrix_distance(self, node_list):
        node_coords = np.array(node_list).reshape(-1,3)[:,1:] # convert node list into numpy array of x and y coords
        distance_matrix = np.sqrt(np.sum((node_coords[:, np.newaxis] - node_coords) ** 2, axis=-1)) # create distance matrix from coords
        return distance_matrix

    # Task 24: Non-dominated sorting
    def non_dominated_sorting(self):
        # Calculate dominated set for each individual
        dominating_sets = []
        dominated_counts = []
        for solution_1 in self.population:
            current_dominating_set = set()
            dominated_counts.append(0)
            for i, solution_2 in enumerate(self.population):
                if solution_1 > solution_2 or solution_1 >= solution_2:
                    current_dominating_set.add(i)
                elif solution_2 > solution_1 or solution_2 >= solution_1:
                    dominated_counts[-1] += 1

            dominating_sets.append(current_dominating_set)

        dominated_counts = np.array(dominated_counts)
        fronts = []
        while True:
            current_front = np.where(dominated_counts==0)[0]
            if len(current_front) == 0:
                break
            fronts.append(current_front)

            for individual in current_front:
                dominated_counts[individual] = -1 # this solution is already accounted for, make it -1 so  ==0 will not find it anymore
                dominated_by_current_set = dominating_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    dominated_counts[dominated_by_current] -= 1
        return fronts
                
    # Task 25: Crowding Distance
    def crowding_distance(self, front):
        fitnesses = np.array([
            [solution.total_profit for solution in self.population],
            [solution.travelling_time for solution in self.population]
        ])
    
        num_objectives = 2
        num_solutions = self.size_p * 2
        
        # Normalise each objectives, so they are in the range [0,1]
        # This is necessary, so each objective's contribution have the same magnitude to the crowding distance.
        normalized_fitnesses = np.zeros_like(fitnesses)

        for i in range(num_objectives):
            min_val = np.min(fitnesses[i, :])
            max_val = np.max(fitnesses[i, :])
            val_range = max_val - min_val
            normalized_fitnesses[i, :] = fitnesses[i, :] - min_val / val_range
        
        crowding_distance = [{ 'value': 0, 'index': i } for i in range(num_solutions)]

        for i in range(num_objectives):
            sorted_front = sorted(front, key = lambda x : fitnesses[i, x])
            
            crowding_distance[sorted_front[0]]['value'] = np.inf
            crowding_distance[sorted_front[-1]]['value'] = np.inf
            if len(sorted_front) > 2:
                for i in range(1, len(sorted_front) - 1):
                    crowding_distance[sorted_front[i]]['value'] += fitnesses[i, sorted_front[i + 1]] - fitnesses[i, sorted_front[i - 1]]

        return crowding_distance
    
    # Task 15: Replace techniques (Elitism)
    def replacement(self):
        fronts = self.non_dominated_sorting()
        elitism = self.population.copy()
        self.population = []
        len_current_pop = 0
        self.fronts = []
        
        i = 0
        while len(fronts[i]) + len(self.population) <= self.size_p:
            self.fronts.append([])
            for index in fronts[i]:
                self.population.append(elitism[index])
                self.fronts[i].append(len_current_pop)
                len_current_pop += 1
            i += 1
        ranking_front = sorted(self.crowding_distance(fronts[i]), key = lambda x: -x['value'])
        for r in ranking_front[0:(self.size_p - len(self.population))]:
            self.population.append(elitism[r['index']])
            self.fronts[i].append(len_current_pop)
            len_current_pop += 1 
