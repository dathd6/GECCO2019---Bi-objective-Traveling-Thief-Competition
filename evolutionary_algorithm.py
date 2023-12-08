import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from constants import TEST_FOLDER
from travelling_theif_problem import TTP

LIMIT_SOLUTION = {
    'a280-n279': 100,
    'a280-n1395': 100,
    'a280-n2790': 100,
    'fnl4461-n4460': 50,
    'fnl4461-n22300': 50,
    'fnl4461-n44600': 50,
    'pla33810-n33809': 20,
    'pla33810-n169045': 20,
    'pla33810-n338090': 20
}

class MOEA:
    def __init__(self, test_name='test-example-n4') -> None:
        self.n_objectives = 2
        self.population = []
        self.size_p = 0
        self.distance_matrix = [] # 2D Array contains cost of the edges between vertices
        self.test_name = test_name

        # Task 1: Extract data from test problem file #
        content_list = []
        test_file = open(f"{TEST_FOLDER}/{test_name}.txt")
        for i in test_file :
            content_list.append(i.split())

        self.number_of_cities = int(content_list[2][-1])    # total number of cities
        self.knapsack_capacity = int(content_list[4][-1])  # threshold value
        self.min_speed = float(content_list[5][-1])        # minimum speed
        self.max_speed = float(content_list[6][-1])       # maximum speed
        self.renting_ratio = float(content_list[7][-1]) # renting ratio
        del content_list[0:10]                     
        node_list = []                            
        for i in range(self.number_of_cities):
            node_list.append([eval(j) for j in content_list[i]])  # list of node's coordinates
        del content_list[0:self.number_of_cities+1]
        
        self.distance_matrix = self.map_node_coord_to_matrix_distance(node_list) # create distance matrix

        self.profit_list = []
        self.weight_list = []
        self.item_location = []

        for row in content_list:
            self.profit_list.append(int(row[1]))         #profits of each bags in the nodes 
            self.weight_list.append(int(row[2]))         # weights of individual bags
            self.item_location.append(int(row[3]))        # List entail the i item in which city

        list_zip = zip(self.item_location, self.profit_list, self.weight_list)
        list_zip_sorted = sorted(list_zip)
        self.item_location, self.profit_list, self.weight_list = zip(*list_zip_sorted)

    # Task 2: Generate initial population
    def generate_initial_population(self, size_p):
        self.size_p = size_p
        population = []
        
        '''
        Generate initial population
        '''
        for _ in range(size_p):
            #Generate TSP initial population
            route = random.sample(range(1, self.number_of_cities+1), self.number_of_cities)
            #Generate KP initial population
            number_of_items = len(self.item_location)
            #stolen_items = np.random.randint(2, size=number_of_items)
            stolen_items = [random.choice([0,1]) for _ in range(number_of_items)]
            '''
            #The total pack weight cannot over capasity
            total_weight = 0
            while True:
                stolen_items = [random.choice([0,1]) for _ in range((number_of_cities-1)*item_num)]
                for i, item in enumerate(stolen_item):
                    if item == 1:
                        total_weight += weight_list_sorted[i]
                if total_weight <= knapscak_capacity:
                    break
            '''
            
            population.append(
                TTP(
                    self.distance_matrix,
                    self.knapsack_capacity,
                    self.min_speed,
                    self.max_speed,
                    self.profit_list,
                    self.weight_list,
                    self.item_location,
                    route,
                    stolen_items
                )
            )
        self.population = np.array(population)
        
    # Task 4: Map imported data node coord to matrix distance matrix
    '''
    Generates distance matrix for problem from node_list
    '''
    def map_node_coord_to_matrix_distance(self, node_list):
        '''
        Parameters
        ----------
        node_list: list of node coordinates as (node, x, y)
        distance_matrix: distance matrix is n by n array that gives distance from city i to city j
        returns
            distance matrix
        '''
        node_coords = np.array(node_list).reshape(-1,3)[:,1:] # convert node list into numpy array of x and y coords
        distance_matrix = np.sqrt(np.sum((node_coords[:, np.newaxis] - node_coords) ** 2, axis=-1)) # create distance matrix from coords
        return distance_matrix

    # Task 11: Crossover for TSP(two functions. two-points crossover with fix & ordered crossover. choose one)
    '''
    following crossover function is a two-points crossover with fix
    '''
    def tsp_two_points_crossover(self, parent1 = [], parent2 = []):
        '''
        Parameters
        ----------
        parent1 : chromosome number one after tournament selection
        parent2 : chromosome number two after tournament selection
        children1 : chromosome after crossover operation
        children1 : chromosome after crossover operation
        number_of_cities : how many cities in each chromosome
        Returns
            children
        ----------
        '''
        '''
        use deep copy so that further operation won't affect original chromosome
        '''
        p1 = copy.deepcopy(parent1)
        p2 = copy.deepcopy(parent2) 
        '''
        random generate two unequal crossover point
        '''
        crossover_point1 = random.randint(0, self.number_of_cities-1)
        crossover_point2 = random.randint(0, self.number_of_cities-1)
        while crossover_point2 == crossover_point1:
            crossover_point2 = random.randint(0, self.number_of_cities-1)
        if crossover_point1 > crossover_point2:
            temp = crossover_point1
            crossover_point1 = crossover_point2
            crossover_point2 = temp
        '''
        store the crossover part into a temporary chain
        '''
        chain1 = p1[crossover_point1:crossover_point2]
        chain2 = p2[crossover_point1:crossover_point2]
        '''
        do the crossover
        break the two father chromosome
        add head of father1,crossover part of father2, tail of father1 together as the children1
        add head of father2,crossover part of father1, tail of father2 together as the children1
        '''
        p1_head = p1[:crossover_point1]
        p1_tail = p1[crossover_point2:]
        p1_c = p1_head + chain2 + p1_tail
        p2_head = p2[:crossover_point1]
        p2_tail = p2[crossover_point2:]
        p2_c = p2_head + chain1 + p2_tail
        '''
        fix p1
        Compare each gene of the parent1 before crossover point 1 and the crossover part of the offspring to find the duplicate genes
        find the INDEX of each duplicate gene in the parent2
        replace the gene in the corresponding position in the part of the parent that was swapped away
        do these again after crossover point 2, as the tail of the chromosome
        '''
        p1_head_fix = []
        for i in p1[:crossover_point1]:
            while i in chain2: 
                i = chain1[chain2.index(i)] 
            p1_head_fix.append(i)
        p1_tail_fix = []
        for i in p1[crossover_point2:]:
            while i in chain2:
                i = chain1[chain2.index(i)]
            p1_tail_fix.append(i)
        p1_c_f = p1_head_fix + chain2 + p1_tail_fix #set the crossover part untouched and add fixed head part and tail part
        '''
        fix p2
        same method with p1
        '''
        p2_head_fix = []
        for i in p2[:crossover_point1]: 
            while i in chain1: 
                i = chain2[chain1.index(i)]
            p2_head_fix.append(i)
        p2_tail_fix = []
        for i in p2[crossover_point2:]:
            while i in chain1:
                i = chain2[chain1.index(i)]
            p2_tail_fix.append(i)
        p2_c_f = p2_head_fix + chain1 + p2_tail_fix
        '''
        use deepcopy copy the chromosomes to offspring that have finished two-points crossover and fixed
        '''
        children1 = copy.deepcopy(p1_c_f)
        children2 = copy.deepcopy(p2_c_f)
        
        return children1, children2
    '''
    following crossover function is a ordered crossover with fix
    '''
    def tsp_ordered_crossover(self, parent1 = [], parent2 = []):
        '''
        Parameters
        ----------
        parent1 : chromosome number one after tournament selection
        parent2 : chromosome number two after tournament selection
        children1 : chromosome after crossover operation
        children1 : chromosome after crossover operation
        city_num : how many cities in each chromosome
        Returns
            children
        ----------
        '''
        '''
        use deep copy so that further operation won't affect original chromosome
        '''
        p1 = copy.deepcopy(parent1)
        p2 = copy.deepcopy(parent2) 
        '''
        random generate two unequal order point
        '''
        order_point1 = random.randint(0, self.number_of_cities-1)
        order_point2 = random.randint(0, self.number_of_cities-1)
        while order_point2 == order_point1:
            order_point2 = random.randint(0, self.number_of_cities-1)
        if order_point1 > order_point2:
            temp = order_point1
            order_point1 = order_point2
            order_point2 = temp
        '''
        copy genes of father1 between tow order points to the children1
        '''
        p1_head = [None]*order_point1
        p1_tail = [None]*(self.number_of_cities - order_point2)
        chain1 = p1[order_point1:order_point2]
        p1_o = p1_head + chain1 + p1_tail
        '''
        copy genes of father2 between tow order points to the children2
        '''
        p2_head = [None]*order_point1
        p2_tail = [None]*(self.number_of_cities - order_point2)
        chain2 = p2[order_point1:order_point2]
        p2_o = p2_head + chain2 + p2_tail
        '''
        Fill the p1 remaining genes in the order of parent 2
        '''
        p1_remain = [i for i in parent2 if i not in p1_o]
        p1_o[:order_point1] = p1_remain[:order_point1]
        p1_o[order_point2:] = p1_remain[order_point1:]
        '''
        Fill the p2 remaining genes in the order of parent 1
        '''
        p2_remain = [i for i in parent1 if i not in p2_o]
        p2_o[:order_point1] = p2_remain[:order_point1]
        p2_o[order_point2:] = p2_remain[order_point1:]
        '''
        use deepcopy copy the chromosomes to offspring that have finished ordered corssover
        '''
        children1 = copy.deepcopy(p1_o)
        children2 = copy.deepcopy(p2_o)
        return children1, children2

    # Task 12: Inversion mutation for TSP
    '''
    following mutation function is inversion mutation
    '''
    def tsp_inversion_mutation(self, parent1 = [], parent2 = []):
        '''
        Parameters
        ----------
        parent1 : chromosome number one after corssover operation
        parent2 : chromosome number two after corssover operation
        children1 : chromosome after mutation operation
        children1 : chromosome after mutation operation
        number_of_cities : how many cities in each chromosome
        Returns
            children
        ----------
        '''
        '''
        use deep copy so that further operation won't affect original chromosome
        '''
        p1 = copy.deepcopy(parent1)
        p2 = copy.deepcopy(parent2)
        '''
        random generate two unequal inverse point for parent1
        '''
        inverse_point1 = random.randint(0, self.number_of_cities-1)
        inverse_point2 = random.randint(0, self.number_of_cities-1)
        '''
        inversion
        '''
        p1_head = p1[:inverse_point1]
        p1_tail = p1[inverse_point1:]
        p1_tail.reverse()
        p1_i = p1_head + p1_tail
        p2_head = p2[:inverse_point2]
        p2_tail = p2[inverse_point2:]
        p2_tail.reverse()
        p2_i = p2_head + p2_tail
        '''
        use deepcopy copy the chromosomes to offspring that have finished two-points crossover and fixed
        '''
        children1 = copy.deepcopy(p1_i)
        children2 = copy.deepcopy(p2_i)
        return children1, children2
    
    # Task 13: Crossover for KP
    '''
    Performs single point crossover for binary 1D arrays for knapsack problem
    '''
    def kp_crossover(self, parent_A, parent_B):
        '''
        Parameters
        ----------
        parent_A: first chromosome for crossover as 1D numpy array
        parent_B: second chromosome for crossover as 1D numpy array
        child_A: first chromosome after crossover as 1D numpy array
        child_B: second chromosome after crossover as 1D numpy array
        returns
            children of crossover
        '''
        p1 = copy.deepcopy(parent_A)
        p2 = copy.deepcopy(parent_B)
        # parent_A, parent_B = parent_A.tolist(), parent_B.tolist() # Convert parents to lists
        crossover_point = np.random.randint(0,len(p1)) # Generate random crossover point
        child_A = p1[:crossover_point] + p2[crossover_point:] # Gererate child_A from parents
        child_B = p2[:crossover_point] + p1[crossover_point:] # Generate child_B from parents
        return child_A, child_B

    # Task 14: Mutation for KP (inversion mutation)
    def kp_mutation(self, parent):
        point1, point2 = sorted(random.sample(range(len(parent)), 2))  # choose 2 different points
        sub = parent[point1:point2 + 1]  # get the subsequence from z
        parent[point1:point2 + 1] = sub[::-1]  # reverse the subsequence
        return parent  # return the mutated parent(new stolen_items

    # Task 24: Non-dominated sorting
    def non_dominated_sorting(self):
        """Fast non-dominated sorting to get list Pareto Fronts"""
        dominating_sets = []
        dominated_counts = []

        # For each solution:
        # - Get solution index that dominated by current solution
        # - Count number of solution dominated current solution
        for solution_1 in self.population:
            current_dominating_set = set()
            dominated_counts.append(0)
            for i, solution_2 in enumerate(self.population):
                if solution_1 >= solution_2 and not solution_1 == solution_2:
                    current_dominating_set.add(i)
                elif solution_2 >= solution_1 and not solution_2 == solution_1:
                    dominated_counts[-1] += 1
            dominating_sets.append(current_dominating_set)

        dominated_counts = np.array(dominated_counts)
        self.fronts = []

        # Append all the pareto fronts and stop when there is no solution being dominated (domintead count = 0)
        while True:
            current_front = np.where(dominated_counts==0)[0]
            if len(current_front) == 0:
                break
            self.fronts.append(current_front)
            for individual in current_front:
                dominated_counts[individual] = -1 # this solution is already accounted for, make it -1 so will not find it anymore
                dominated_by_current_set = dominating_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    dominated_counts[dominated_by_current] -= 1
                
    # Task 25: Crowding Distance
    def calc_crowding_distance(self):
        self.crowding_distance = np.zeros(len(self.population))

        for front in self.fronts:
            fitnesses = np.array([
                solution.get_fitness() for solution in self.population[front]
            ])
        
            # Normalise each objectives, so they are in the range [0,1]
            # This is necessary, so each objective's contribution have the same magnitude to the crowding distance.
            normalized_fitnesses = np.zeros_like(fitnesses)

            for j in range(self.n_objectives):
                min_val = np.min(fitnesses[:, j])
                max_val = np.max(fitnesses[:, j])
                val_range = max_val - min_val
                normalized_fitnesses[:, j] = (fitnesses[:, j] - min_val) / val_range

            for j in range(self.n_objectives):
                idx = np.argsort(fitnesses[:, j])
                
                self.crowding_distance[idx[0]] = np.inf
                self.crowding_distance[idx[-1]] = np.inf
                if len(idx) > 2:
                    for i in range(1, len(idx) - 1):
                        self.crowding_distance[idx[i]] += normalized_fitnesses[idx[i + 1], j] - normalized_fitnesses[idx[i - 1], j]
        
    #Task 16: visualize and like in the requirement
    def visualize(self):
        for front in self.fronts:
            pareto_value = np.array([solution.get_fitness() for solution in self.population[front]])
            plt.scatter(
                pareto_value[:, 0],
                pareto_value[:, 1],
            )
        plt.xlabel('travelling time')
        plt.ylabel('total profit')
        plt.grid()
        plt.show()

    def export_result(self):
        DIR = 'test_results/'
        with open(f'{DIR}/TeamU_{self.test_name}.f','w') as f:
            count = 0
            for solution in self.population[self.fronts[0]]:
                f.write(f"{solution.travelling_time} {solution.total_profit}\n")
                count += 1
                if count == LIMIT_SOLUTION[self.test_name]:
                    break

        with open(f'{DIR}/TeamU_{self.test_name}.x','w') as f:
            count = 0
            for solution in self.population[self.fronts[0]]:
                f.write(f"{str(solution.route)[1:-1].replace(',', '')}\n")
                f.write(f"{str(solution.stolen_items)[1:-1].replace(',', '')}\n")
                f.write('\n')
                count += 1
                if count == LIMIT_SOLUTION[self.test_name]:
                    break
    
    # Task 15: Replace techniques (Elitism)
    def elitism_replacement(self):
        elitism = copy.deepcopy(self.population)
        population = []
        
        i = 0
        while len(self.fronts[i]) + len(population) <= self.size_p:
            for solution in elitism[self.fronts[i]]:
                population.append(solution)
            i += 1

        front = self.fronts[i]
        ranking_index = front[np.argsort(self.crowding_distance[front])]
        current_pop_len = len(population)
        for index in ranking_index[current_pop_len:self.size_p]:
            population.append(elitism[index])
        self.population = np.array(population)


    # Task 10: Tournament selection
    def tournament_selection(self):
        tournament = np.array([True] * self.size_t + [False] * (self.size_p - self.size_t))
        results = []
        for _ in range(2):
            np.random.shuffle(tournament)
            front = []
            for f in self.fronts:
                front = []
                for index in f:
                    if tournament[index] == 1:
                        front.append(index)
                if len(front) > 0:
                    break
            max_index = np.argmax(self.crowding_distance[front])
            results.append(self.population[front[max_index]])
        return results


    # Task 36: Optimization
    def optimize(self, generations, tournament_size, crossover='OX'):
        self.size_t = tournament_size

        for generation in range(generations):
            print('Generation: ', generation + 1)
            new_solutions = []
            self.non_dominated_sorting()
            self.calc_crowding_distance()
            while len(self.population) + len(new_solutions) < 2 * self.size_p:
                parents = self.tournament_selection()
                if crossover == 'PMX':
                    route_child_a, route_child_b = self.tsp_two_points_crossover(parents[0].route, parents[1].route)
                else:
                    route_child_a, route_child_b = self.tsp_ordered_crossover(parents[0].route, parents[1].route)
                stolen_child_a, stolen_child_b = self.kp_crossover(parents[0].stolen_items, parents[1].stolen_items)
                new_route_c, new_route_d = self.tsp_inversion_mutation(route_child_a, route_child_b)
                new_stolen_c = self.kp_mutation(stolen_child_a) 
                new_stolen_d = self.kp_mutation(stolen_child_b) 

                new_solutions.append(
                    TTP(
                        self.distance_matrix,
                        self.knapsack_capacity,
                        self.min_speed,
                        self.max_speed,
                        self.profit_list,
                        self.weight_list,
                        self.item_location,
                        new_route_c,
                        new_stolen_c
                    )
                )
                new_solutions.append(
                    TTP(
                        self.distance_matrix,
                        self.knapsack_capacity,
                        self.min_speed,
                        self.max_speed,
                        self.profit_list,
                        self.weight_list,
                        self.item_location,
                        new_route_d,
                        new_stolen_d
                    )
                )

            self.population = np.append(self.population, new_solutions)
            self.non_dominated_sorting()
            self.calc_crowding_distance()
            self.elitism_replacement()
        self.non_dominated_sorting()
        self.calc_crowding_distance()

    # Task10 Crowding Tournament selection
    def crowding_tournament_selection(self):
        results = []
        tournament_size = self.size_t

        self.non_dominated_sorting()     # apply non-dominated sorting to update the non-dominant rank

        for _ in range(self.size_p):
            tournament_select = np.random.choice(range(len(self.population)), tournament_size, replace=False)  # random selection

            selected = [self.population[i] for i in tournament_select] # find the selected individuals

            # determine the dominate rank of each individual
            front_ranks = []
            for individual in selected:
                for i, front in enumerate(self.fronts):
                    if individual in front:
                        front_ranks.append(i)
                        break
                else:
                    front_ranks.append(len(self.fronts))  # if not in any front, then it is the last front

            # select the highest rank of individual
            min_rank = min(front_ranks)
            candidates = [selected[i] for i in range(len(selected)) if front_ranks[i] == min_rank]

            # if more than one candidate, select the one with the highest crowding distance
            if len(candidates) > 1:
                crowding_distances = [self.crowding_distance[self.population.index(individual)] for individual in candidates]
                max_distance = max(crowding_distances)
                max_index = crowding_distances.index(max_distance)
                results = candidates[max_index]
            else:
                results.append(candidates[0])

        return results