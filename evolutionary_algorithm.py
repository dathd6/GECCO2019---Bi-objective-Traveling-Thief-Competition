from constants import TEST_FOLDER

class EA:
    def __init__(self, test_name='test-example-n4.txt') -> None:
        self.population = []
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
        
        for i in range(len(node_list)):
            self.distance_matrix.append([])
            for j in range(len(node_list)):
                self.distance_matrix[i].append(self.calc_distance(node_list[j], node_list[j]))

        self.profit_list = []
        self.weight_list = []
        self.city_of_item = []

        for row in content_list:
            self.profit_list.append(int(row[1]))         #profits of each bags in the nodes 
            self.weight_list.append(int(row[2]))         # weights of individual bags
            self.city_of_item.append(int(row[3]))        # List entail the i item in which city

    # Task 2: Generate initial population
    def generate_initial_population(self):
        pass

    # Task 3: Calculate the distance between two cities #
    def calc_distance(self, city_i, city_j):
        return 0

    # Task 4: Map imported data node coord to matrix distance matrix
    def map_node_coord_to_matrix_distance(self, node_list):
        pass

    # Task 15: Replace techniques
    def replacement(self, new_solution_E, new_solution_F):
        replacement_list = []
        if new_solution_E > new_solution_F:
            replacement_list.append(new_solution_E)
        elif new_solution_F > new_solution_E:
            replacement_list.append(new_solution_F)
        elif new_solution_F == new_solution_E:
            replacement_list.append(new_solution_E)
        else:
            replacement_list.append(new_solution_E)
            replacement_list.append(new_solution_F)

        list_index = []
        # steady approach
        for new_solution in replacement_list:
            # Iterate through all solution
            flag = False
            index = 0
            while new_solution > self.population[index] or new_solution >= self.population[index]:
                list_index.append(index)
                index += 1
            if flag:
                self.population.append(new_solution)
        list_index.sort()
        for index in list_index:
            self.population.pop(index)