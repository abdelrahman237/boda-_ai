"""
Genetic Operations - Task 3
Crossover, mutation, and selection operations for genetic algorithms
Extracted from genetic_ops.ipynb (simplified versions without GUI info)
"""

import random
import copy


def single_point_crossover(parent1, parent2):
    """
    Single-point crossover for schedules
    Returns child schedule
    """
    min_len = min(len(parent1), len(parent2))
    
    if min_len < 2:
        return copy.deepcopy(parent1 if random.random() < 0.5 else parent2)
    
    point = random.randint(1, min_len - 1)
    child_matches = []
    seen_pairs = set()
    
    # Take matches from parent1 up to crossover point
    for i in range(point):
        if i < len(parent1):
            match = copy.deepcopy(parent1[i])
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                child_matches.append(match)
    
    # Take matches from parent2 after crossover point
    for i in range(point, len(parent2)):
        if i < len(parent2):
            match = copy.deepcopy(parent2[i])
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                child_matches.append(match)
    
    # Fill remaining if needed
    if len(child_matches) < min_len:
        all_matches = []
        seen_pairs = set()
        for match in parent1 + parent2:
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_matches.append(copy.deepcopy(match))
        
        random.shuffle(all_matches)
        for match in all_matches:
            if len(child_matches) >= min_len:
                break
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in {f"{m.team1}_{m.team2}" for m in child_matches}:
                child_matches.append(match)
    
    child_matches.sort(key=lambda m: m.date)
    return child_matches


def two_point_crossover(parent1, parent2):
    """
    Two-point crossover for schedules
    Returns child schedule
    """
    min_len = min(len(parent1), len(parent2))
    
    if min_len < 3:
        return single_point_crossover(parent1, parent2)
    
    point1, point2 = sorted(random.sample(range(1, min_len), 2))
    child_matches = []
    seen_pairs = set()
    
    # Part from parent 1 (before point1)
    for i in range(point1):
        if i < len(parent1):
            match = copy.deepcopy(parent1[i])
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                child_matches.append(match)
    
    # Part from parent 2 (between point1 and point2)
    for i in range(point1, point2):
        if i < len(parent2):
            match = copy.deepcopy(parent2[i])
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                child_matches.append(match)
    
    # Part from parent 1 (after point2)
    for i in range(point2, len(parent1)):
        if i < len(parent1):
            match = copy.deepcopy(parent1[i])
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                child_matches.append(match)
    
    # Fill if needed
    if len(child_matches) < min_len:
        all_matches = []
        seen_pairs = set()
        for match in parent1 + parent2:
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_matches.append(copy.deepcopy(match))
        
        random.shuffle(all_matches)
        for match in all_matches:
            if len(child_matches) >= min_len:
                break
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in {f"{m.team1}_{m.team2}" for m in child_matches}:
                child_matches.append(match)
    
    child_matches.sort(key=lambda m: m.date)
    return child_matches


def uniform_crossover(parent1, parent2):
    """
    Uniform crossover: for each position, randomly choose from parent1 or parent2
    Returns child schedule
    """
    min_len = min(len(parent1), len(parent2))
    max_len = max(len(parent1), len(parent2))
    
    child_matches = []
    seen_pairs = set()
    
    # For first min_len matches
    for i in range(min_len):
        # Randomly select which parent to take from
        if random.random() < 0.5:
            source = parent1[i] if i < len(parent1) else None
        else:
            source = parent2[i] if i < len(parent2) else None
        
        if source:
            match = copy.deepcopy(source)
            pair_key = f"{match.team1}_{match.team2}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                child_matches.append(match)
    
    # For additional matches from longer parent
    if max_len > min_len:
        longer_parent = parent1 if len(parent1) > len(parent2) else parent2
        for i in range(min_len, max_len):
            if i < len(longer_parent):
                match = copy.deepcopy(longer_parent[i])
                pair_key = f"{match.team1}_{match.team2}"
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    child_matches.append(match)
    
    child_matches.sort(key=lambda m: m.date)
    return child_matches


def swap_mutation(schedule, mutation_rate=0.1):
    """
    Swap mutation: exchange two random matches in the schedule
    Returns mutated schedule
    """
    mutated = copy.deepcopy(schedule)
    
    if len(mutated) < 2:
        return mutated
    
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    
    return mutated


def change_venue_mutation(schedule, venues, mutation_rate=0.1):
    """
    Change venue mutation: change venue of a random match
    Returns mutated schedule
    """
    mutated = copy.deepcopy(schedule)
    
    if not mutated or len(venues) < 2:
        return mutated
    
    if random.random() < mutation_rate:
        idx = random.randint(0, len(mutated) - 1)
        old_venue = mutated[idx].venue
        available_venues = [v for v in venues if v != old_venue]
        
        if available_venues:
            new_venue = random.choice(available_venues)
            mutated[idx].venue = new_venue
    
    return mutated


def change_time_mutation(schedule, match_times, mutation_rate=0.1):
    """
    Change time mutation: change time of a random match
    Returns mutated schedule
    """
    mutated = copy.deepcopy(schedule)
    
    if not mutated or len(match_times) < 2:
        return mutated
    
    if random.random() < mutation_rate:
        idx = random.randint(0, len(mutated) - 1)
        old_time = mutated[idx].time
        available_times = [t for t in match_times if t != old_time]
        
        if available_times:
            new_time = random.choice(available_times)
            mutated[idx].time = new_time
    
    return mutated


def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Tournament selection: select best individual from random tournament
    Returns selected schedule
    """
    tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_idx]


def roulette_wheel_selection(population, fitness_scores):
    """
    Roulette wheel selection: probability proportional to fitness
    Returns selected schedule
    """
    # Normalize fitness scores to positive values
    min_fitness = min(fitness_scores)
    adjusted_fitness = [f - min_fitness + 1 for f in fitness_scores]
    total_fitness = sum(adjusted_fitness)
    
    if total_fitness == 0:
        return random.choice(population)
    
    probabilities = [f / total_fitness for f in adjusted_fitness]
    r = random.random()
    cumulative = 0
    for i, prob in enumerate(probabilities):
        cumulative += prob
        if r <= cumulative:
            return population[i]
    return population[-1]


def rank_selection(population, fitness_scores):
    """
    Rank-based selection: selection based on rank rather than absolute fitness
    Returns selected schedule
    """
    # Create list of (index, fitness) pairs and sort by fitness
    indexed_fitness = list(enumerate(fitness_scores))
    indexed_fitness.sort(key=lambda x: x[1], reverse=True)
    
    # Assign ranks (higher fitness = higher rank)
    ranks = [len(population) - i for i in range(len(population))]
    
    # Calculate selection probabilities based on ranks
    total_rank = sum(ranks)
    probabilities = [r / total_rank for r in ranks]
    
    # Select based on probabilities
    r = random.random()
    cumulative = 0
    for i, prob in enumerate(probabilities):
        cumulative += prob
        if r <= cumulative:
            return population[indexed_fitness[i][0]]
    return population[indexed_fitness[-1][0]]


