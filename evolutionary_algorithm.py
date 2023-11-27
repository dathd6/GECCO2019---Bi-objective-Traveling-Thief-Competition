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
    def generate_initial_population(self, size_p, number_of_cities):
        for i in range(size_p):
            chromosome = random.sample(range(number_of_cities), number_of_cities)
            population.append(chromosome)

        return population

    # Task 4: Map imported data node coord to matrix distance matrix
    def map_node_coord_to_matrix_distance(self, node_list):
        node_coords = np.array(node_list).reshape(-1,3)[:,1:] # convert node list into numpy array of x and y coords
        distance_matrix = np.sqrt(np.sum((node_coords[:, np.newaxis] - node_coords) ** 2, axis=-1)) # create distance matrix from coords
        return distance_matrix

    # Task 11: Crossover for TSP(two functions. two-points crossover with fix & ordered crossover. choose one)
    '''
    following crossover function is a two-points crossover with fix
    '''
    def tsp_two_points_crossover(self, number_of_cities, parent1 = [], parent2 = []):
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
        print("p1 is", p1)
        print("p2 is", p2)
        '''
        random generate two unequal crossover point
        '''
        crossover_point1 = random.randint(0, number_of_cities-1)
        crossover_point2 = random.randint(0, number_of_cities-1)
        while crossover_point2 == crossover_point1:
            crossover_point2 = random.randint(0, number_of_cities-1)
        if crossover_point1 > crossover_point2:
            temp = crossover_point1
            crossover_point1 = crossover_point2
            crossover_point2 = temp
        #print("c_point1 is ", crossover_point1)
        #print("c_point2 is ", crossover_point2)    
        '''
        store the crossover part into a temporary chain
        '''
        chain1 = p1[crossover_point1:crossover_point2]
        chain2 = p2[crossover_point1:crossover_point2]
        print("chain1 is " , chain1)
        print("chain2 is " , chain2)
        '''
        do the crossover
        break the two father chromosome
        add head of father1,crossover part of father2, tail of father1 together as the children1
        add head of father2,crossover part of father1, tail of father2 together as the children1
        '''
        p1_head = p1[:crossover_point1]
        p1_tail = p1[crossover_point2:]
        p1_c = p1_head + chain2 + p1_tail
        print("p1 crossover is ", p1_c)
        p2_head = p2[:crossover_point1]
        p2_tail = p2[crossover_point2:]
        p2_c = p2_head + chain1 + p2_tail
        print("p2 crossover is ", p2_c)
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
    def tsp_ordered_crossover(self, number_of_cities, parent1 = [], parent2 = []):
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
        #print("p1 is", p1)
        #print("p2 is", p2)
        '''
        random generate two unequal order point
        '''
        order_point1 = random.randint(0, number_of_cities-1)
        order_point2 = random.randint(0, number_of_cities-1)
        while order_point2 == order_point1:
            order_point2 = random.randint(0, number_of_cities-1)
        if order_point1 > order_point2:
            temp = order_point1
            order_point1 = order_point2
            order_point2 = temp
        #print("o_point1 is ", order_point1)
        #print("o_point2 is ", order_point2)
        '''
        copy genes of father1 between tow order points to the children1
        '''
        p1_head = [None]*order_point1
        p1_tail = [None]*(number_of_cities - order_point2)
        chain1 = p1[order_point1:order_point2]
        p1_o = p1_head + chain1 + p1_tail
        #print("p1_o is ", p1_o)
        '''
        copy genes of father2 between tow order points to the children2
        '''
        p2_head = [None]*order_point1
        p2_tail = [None]*(number_of_cities - order_point2)
        chain2 = p2[order_point1:order_point2]
        p2_o = p2_head + chain2 + p2_tail
        #print("p2_o is ", p2_o)
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
    def tsp_inversion_mutation(number_of_cities, parent1 = [], parent2 = []):
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
        inverse_point1 = random.randint(0, number_of_cities-1)
        inverse_point2 = random.randint(0, number_of_cities-1)
        print("inverse points for parent1&2 are ", inverse_point1+1, " ", inverse_point2+1)
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
