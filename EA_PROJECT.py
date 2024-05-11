import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read distance data
distances_df = pd.read_csv('distance.csv')
distances = distances_df.to_numpy()

# Read order data
orders_df = pd.read_csv('order_large.csv')
orders = orders_df.to_numpy()

# Select a sample of orders
items = orders[:, [2, 3, 4, 5, 6, 8, 9]].tolist()  # Selecting columns 3, 4, 5, 6, 7, 9, 10

# Select the first two cities from distances
cities = distances[:2]

# Define truck types
truck_types = [
    {'length': 16.5, 'inner_size': (16.1, 2.5), 'weight_capacity': 10000, 'cost_per_km': 3, 'speed': 40},
    {'length': 12.5, 'inner_size': (12.1, 2.5), 'weight_capacity': 5000, 'cost_per_km': 2, 'speed': 40},
    {'length': 9.6, 'inner_size': (9.1, 2.3), 'weight_capacity': 2000, 'cost_per_km': 1, 'speed': 40}
]

# Maximum stops per truck
max_stops = 10

# Stop time in minutes
stop_time = 30

# Define Item class
class Item:
    def __init__(self, id, source_city, destination_city, availability_time, deadline, area, weight, compatibility_groups, delivery_cost):
        self.id = id
        self.source_city = source_city
        self.destination_city = destination_city
        self.availability_time = availability_time
        self.deadline = deadline
        self.area = area
        self.weight = weight
        self.compatibility_groups = compatibility_groups  # List of compatibility groups
        self.delivery_cost = delivery_cost

# Define Truck class
class Truck:
    def __init__(self, truck_type, area_capacity, weight_capacity, max_stops):
        self.type = truck_type
        self.area_capacity = area_capacity
        self.weight_capacity = weight_capacity
        self.items = []  # List of assigned items
        self.start_time = None
        self.end_time = None
        self.cost = 0
        self.max_stops = max_stops

    def can_add_item(self, item):
        return (self.area_capacity - sum(item_.area for item_ in self.items) >= item[5] and
                self.weight_capacity - sum(item_.weight for item_ in self.items) >= item[6] and
                self.is_compatible(item) and
                len(self.items) < self.max_stops)

    def add_item(self, item):
        self.items.append(item)
        self.update_availability(item)

    def update_availability(self, item):
        if self.start_time is None:
            self.start_time = item[3]
        else:
            self.start_time = max(self.start_time, item[3])
            self.end_time = max(self.end_time or (self.start_time + stop_time + item[9]), item[4])

    def is_compatible(self, item):
        for item_ in self.items:
            if item_[8] and item[8]:
                # Check for any incompatibility between groups
                if any(group in item_[8] for group in item[8]):
                    return False
        return True

    def calculate_cost(self):
        self.cost = sum(item[9] for item in self.items) + len(self.items) * stop_cost

# Define Route class
class Route:
    def __init__(self, cities, items, trucks):
        self.cities = cities
        self.items = items
        self.trucks = trucks

# Define function to calculate distance cost
def calculate_distance_cost(city1, city2):
    # Retrieve distance between city1 and city2 from distances array
    distance_data = distances[np.where((distances[:, 0] == city1) & (distances[:, 1] == city2))]
    if len(distance_data) == 0:
        # Handle potential missing distances (e.g., set to a high value)
        return 100000  # Replace with a suitable large value
    distance_cost = distance_data[0][2]  # Assuming distance is in the 3rd column
    return distance_cost

# Define function to calculate route cost
def calculate_route_cost(route):
    total_cost = 0
    for truck in route.trucks:
        total_cost += truck.cost
        # Add distance cost based on the route taken by the truck
        # (consider item order and city locations)
        # You can implement a function to calculate this based on your specific needs
        # (e.g., using the order of items in the truck and distances between cities)
        total_cost += calculate_distance_cost(truck.items[0][1], truck.items[-1][2])  # Simple example using first and last item locations
    return total_cost

# Define function to generate a random route
def generate_random_route(cities, items, truck_types, num_trucks):
    route = Route(cities, items.copy(), [])
    for _ in range(num_trucks):
        truck = random.choice(truck_types)
        area_capacity = truck["length"] * truck["inner_size"][0] * truck["inner_size"][1]
        route.trucks.append(Truck(truck, area_capacity, truck["weight_capacity"], max_stops))
    for item in items:
        # Find a suitable truck for the item
        for truck in route.trucks:
            if truck.can_add_item(item):
                truck.add_item(item)
                items.remove(item)
                break
    # If any items are unassigned, the route is invalid
    if items:
        return None
    return route

# Define function to evaluate route
def evaluate_route(route, max_availability_window):
    total_cost = calculate_route_cost(route)
    # Penalize routes that exceed the max availability window
    if route.trucks[-1].end_time > max_availability_window:
        total_cost *= 2  # Penalize doubly
    return 1 / total_cost  # Inverse of cost, assuming minimizing

# Define function to create initial population
def create_initial_population(population_size, items, truck_types, num_trucks):
    population = []
    for _ in range(population_size):
        route = generate_random_route(cities, items, truck_types, num_trucks)
        if route:  # Ensure valid routes are added to the population
            population.append(route)
    return population

# Define function to calculate fitness
def fitness(route, max_availability_window):
    cost = calculate_route_cost(route)
    # Penalize routes that exceed the max availability window
    if route.trucks[-1].end_time > max_availability_window:
        cost *= 2  # Penalize doubly
    return 1 / cost  # Inverse of cost, assuming minimizing

# Define function for selection
def selection(population, k=2):
    return random.choices(population, k=k, weights=[fitness(route) for route in population])

# Define function for crossover
def crossover(parent1, parent2):
    # Determine crossover points randomly
    crossover_point1 = random.randint(1, len(parent1.trucks) - 2)
    crossover_point2 = random.randint(crossover_point1, len(parent1.trucks) - 1)

    # Create offspring trucks based on crossover points
    offspring1_trucks = parent1.trucks[:crossover_point1] + parent2.trucks[crossover_point1:crossover_point2] + parent1.trucks[crossover_point2:]
    offspring2_trucks = parent2.trucks[:crossover_point1] + parent1.trucks[crossover_point1:crossover_point2] + parent2.trucks[crossover_point2:]

    # Create offspring routes with new truck configurations
    offspring1 = Route(route.cities, items.copy(), offspring1_trucks)
    offspring2 = Route(route.cities, items.copy(), offspring2_trucks)

    # Ensure offspring routes are valid (all items assigned)
    offspring1 = validate_and_fix_route(offspring1)
    offspring2 = validate_and_fix_route(offspring2)

    return offspring1, offspring2

# Define function for mutation
def mutate(route):
    # Determine mutation type (randomly choose between swap_items and move_item)
    mutation_type = random.choice(["swap_items", "move_item"])

    if mutation_type == "swap_items":
        # Swap items between two trucks
        truck1_index, truck2_index = random.sample(range(len(route.trucks)), 2)
        truck1, truck2 = route.trucks[truck1_index], route.trucks[truck2_index]
        item_index1 = random.randint(0, len(truck1.items) - 1)
        item_index2 = random.randint(0, len(truck2.items) - 1)
        truck1.items[item_index1], truck2.items[item_index2] = truck2.items[item_index2], truck1.items[item_index1]
    elif mutation_type == "move_item":
        # Move an item from one truck to another
        truck_index1, truck_index2 = random.sample(range(len(route.trucks)), 2)
        truck1, truck2 = route.trucks[truck_index1], route.trucks[truck_index2]
        if truck1.items and truck2.can_add_item(truck1.items[-1]):
            item = truck1.items.pop()
            truck2.add_item(item)

# Define function to validate and fix route
def validate_and_fix_route(route):
    # Check if all items are assigned
    if any(item in route.items for item in items):
        return None  # Route is invalid if items are unassigned

    # Update truck availabilities and calculate route cost
    for truck in route.trucks:
        truck.update_availability(truck.items[0])  # Assuming first item is picked up at start time
        truck.calculate_cost()

    # Route is valid, return it
    return route

# Define function for genetic algorithm
def genetic_algorithm(cities, items, truck_types, num_trucks, population_size, max_availability_window, generations):
    population = create_initial_population(population_size, items, truck_types, num_trucks)
    for _ in range(generations):
        # Select parents
        parent1, parent2 = selection(population)
        # Crossover
        offspring = crossover(parent1, parent2)
        # Mutation
        mutated_offspring = [mutate(route) for route in offspring]
        # Evaluate fitness of offspring
        offspring_fitness = [fitness(route, max_availability_window) for route in mutated_offspring]
        # Replace worst individuals in the population with offspring
        population = sorted(population, key=lambda route: fitness(route, max_availability_window))
        population[-len(mutated_offspring):] = mutated_offspring
    # Return the best route found
    return max(population, key=lambda route: fitness(route, max_availability_window))

# Parameters for the genetic algorithm
population_size = 50
generations = 100
max_availability_window = 1000  # Example value, adjust as needed

# Run the genetic algorithm and find the best route
best_route = genetic_algorithm(cities, items, truck_types, len(truck_types), population_size, max_availability_window, generations)

# Print the best route
print("Best route:", best_route)
