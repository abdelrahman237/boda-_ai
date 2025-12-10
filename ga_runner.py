"""
Genetic Algorithm Engine for Sports Tournament Scheduling
Person 4 - GA Engine + Experiments

This module implements a complete genetic algorithm for optimizing
sports tournament schedules with selection, crossover, mutation, and elitism.
"""

import random
import copy
import json
import csv
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import sys
import os

# Import shared modules - try different paths
try:
    from data.models import Match
    from data.fitness import compute_fitness
    from data.teams_venues_times import teams, venues, match_times
except ImportError:
    # If direct import fails, add to path
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    if data_path not in sys.path:
        sys.path.insert(0, data_path)
    from models import Match
    from fitness import compute_fitness
    from teams_venues_times import teams, venues, match_times

# Tournament period
start_date = datetime(2025, 5, 1)
end_date = datetime(2026, 1, 31)

# Setup all dates
all_dates = []
current_date = start_date
while current_date <= end_date:
    all_dates.append(current_date)
    current_date += timedelta(days=1)


def generate_weekly_schedule(teams, venues, all_dates, match_times, min_rest_days=4):
    """
    Generate complete schedule: 18 teams × 17 opponents × 2 = 306 matches
    Each team plays every other team twice (home and away)
    """
    schedule = []
    last_played = {team: start_date - timedelta(days=min_rest_days) for team in teams}
    day_info = {d: {'count': 0, 'venues': set(), 'times': set()} for d in all_dates}
    
    # Generate all required pairings (home and away for each pair)
    pairings = []
    for t1 in teams:
        for t2 in teams:
            if t1 != t2:
                pairings.append((t1, t2))  # Home and away matches
    
    random.shuffle(pairings)
    matches_count = {(t1, t2): 0 for t1 in teams for t2 in teams if t1 != t2}
    required_matches = 2  # Each pair plays twice (home and away)
    
    # Try to schedule matches across all available dates
    for home, away in pairings:
        if matches_count[(home, away)] >= required_matches:
            continue
        
        # Find available date with rest day constraints
        possible_dates = [
            d for d in all_dates
            if (d - last_played[home]).days >= min_rest_days
            and (d - last_played[away]).days >= min_rest_days
            and day_info[d]['count'] < 2  # Max 2 matches per day
        ]
        
        if not possible_dates:
            # Try with relaxed constraints (only venue/time availability)
            possible_dates = [
                d for d in all_dates
                if day_info[d]['count'] < 2
            ]
        
        if not possible_dates:
            continue
        
        random.shuffle(possible_dates)
        scheduled = False
        
        for date in possible_dates:
            available_times = [t for t in match_times if t not in day_info[date]['times']]
            available_venues = [v for v in venues if v not in day_info[date]['venues']]
            
            if available_times and available_venues:
                time = random.choice(available_times)
                venue = random.choice(available_venues)
                match = Match(home, away, date, time, venue)
                schedule.append(match)
                last_played[home] = date
                last_played[away] = date
                matches_count[(home, away)] += 1
                day_info[date]['count'] += 1
                day_info[date]['venues'].add(venue)
                day_info[date]['times'].add(time)
                scheduled = True
                break
        
        # If still couldn't schedule, try any available slot
        if not scheduled and matches_count[(home, away)] < required_matches:
            for date in all_dates:
                if day_info[date]['count'] < 2:
                    available_times = [t for t in match_times if t not in day_info[date]['times']]
                    available_venues = [v for v in venues if v not in day_info[date]['venues']]
                    if available_times and available_venues:
                        time = random.choice(available_times)
                        venue = random.choice(available_venues)
                        match = Match(home, away, date, time, venue)
                        schedule.append(match)
                        last_played[home] = date
                        last_played[away] = date
                        matches_count[(home, away)] += 1
                        day_info[date]['count'] += 1
                        day_info[date]['venues'].add(venue)
                        day_info[date]['times'].add(time)
                        break

    schedule.sort(key=lambda m: m.date)
    return schedule


# ==================== VALIDATION FUNCTIONS ====================

def validate_schedule(schedule, teams, min_rest_days=4):
    """
    Validate schedule constraints
    
    Returns:
    --------
    tuple: (is_valid: bool, errors: List[str])
    """
    errors = []
    
    # Check for duplicate matches (same teams, same date)
    seen_matches = {}
    for match in schedule:
        key = (match.team1, match.team2, match.date)
        if key in seen_matches:
            errors.append(f"Duplicate match: {match}")
        seen_matches[key] = True
    
    # Check for self-play
    for match in schedule:
        if match.team1 == match.team2:
            errors.append(f"Team playing itself: {match}")
    
    # Check rest days
    last_played = {}
    for match in schedule:
        for team in [match.team1, match.team2]:
            if team in last_played:
                delta = (match.date - last_played[team]).days
                if delta < min_rest_days:
                    errors.append(f"Rest violation: {team} played {delta} days apart")
            last_played[team] = match.date
    
    # Check venue conflicts
    venue_slots = {}
    for match in schedule:
        key = (match.date, match.time, match.venue)
        if key in venue_slots:
            errors.append(f"Venue conflict: {match}")
        venue_slots[key] = True
    
    return len(errors) == 0, errors


def repair_schedule(schedule, teams, venues, match_times, all_dates, min_rest_days=4):
    """
    Repair invalid schedule by removing duplicates and fixing conflicts
    
    Returns:
    --------
    List[Match]: Repaired schedule
    """
    # Remove duplicates and self-play matches
    seen = set()
    valid_schedule = []
    for match in schedule:
        key = (match.team1, match.team2, match.date)
        if key not in seen and match.team1 != match.team2:
            seen.add(key)
            valid_schedule.append(match)
    
    # Remove venue conflicts (keep first occurrence)
    venue_slots = {}
    repaired = []
    for match in valid_schedule:
        key = (match.date, match.time, match.venue)
        if key not in venue_slots:
            venue_slots[key] = True
            repaired.append(match)
        else:
            # Try to fix by changing venue or time
            available_venues = [v for v in venues if v != match.venue]
            available_times = [t for t in match_times if t != match.time]
            
            fixed = False
            for new_venue in available_venues:
                new_key = (match.date, match.time, new_venue)
                if new_key not in venue_slots:
                    match.venue = new_venue
                    venue_slots[new_key] = True
                    repaired.append(match)
                    fixed = True
                    break
            
            if not fixed:
                for new_time in available_times:
                    new_key = (match.date, new_time, match.venue)
                    if new_key not in venue_slots:
                        match.time = new_time
                        venue_slots[new_key] = True
                        repaired.append(match)
                        fixed = True
                        break
    
    return repaired


# ==================== SELECTION METHODS ====================

def tournament_selection(population, fitness_scores, tournament_size=3):
    """Tournament selection: select best from random tournament"""
    tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_idx]


def roulette_wheel_selection(population, fitness_scores):
    """Roulette wheel selection: probability proportional to fitness"""
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
    """Rank-based selection: selection based on rank rather than absolute fitness"""
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


# ==================== CROSSOVER METHODS ====================

def single_point_crossover(parent1, parent2):
    """Single-point crossover for schedules"""
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
    """Two-point crossover for schedules"""
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


# ==================== MUTATION METHODS ====================

def swap_mutation(schedule, mutation_rate=0.1):
    """Swap two random matches in the schedule"""
    mutated = copy.deepcopy(schedule)
    
    if len(mutated) < 2:
        return mutated
    
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    
    return mutated


def change_venue_mutation(schedule, venues, mutation_rate=0.1):
    """Change venue of a random match"""
    mutated = copy.deepcopy(schedule)
    
    if len(mutated) == 0:
        return mutated
    
    if random.random() < mutation_rate:
        idx = random.randint(0, len(mutated) - 1)
        old_venue = mutated[idx].venue
        new_venue = random.choice([v for v in venues if v != old_venue])
        mutated[idx].venue = new_venue
    
    return mutated


def change_time_mutation(schedule, match_times, mutation_rate=0.1):
    """Change time of a random match"""
    mutated = copy.deepcopy(schedule)
    
    if len(mutated) == 0:
        return mutated
    
    if random.random() < mutation_rate:
        idx = random.randint(0, len(mutated) - 1)
        old_time = mutated[idx].time
        new_time = random.choice([t for t in match_times if t != old_time])
        mutated[idx].time = new_time
    
    return mutated


def apply_mutation(schedule, venues, match_times, mutation_rate=0.1):
    """
    Apply mutation based on mutation_rate
    
    The mutation_rate determines the probability that a mutation will occur.
    If mutation occurs, a random mutation type is chosen.
    """
    mutated = copy.deepcopy(schedule)
    
    # Check if mutation should occur
    if random.random() >= mutation_rate:
        return mutated
    
    # Choose mutation type (can be weighted)
    mutation_type = random.choice(['swap', 'venue', 'time'])
    
    if mutation_type == 'swap' and len(mutated) >= 2:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    elif mutation_type == 'venue' and len(mutated) > 0:
        idx = random.randint(0, len(mutated) - 1)
        old_venue = mutated[idx].venue
        new_venue = random.choice([v for v in venues if v != old_venue])
        mutated[idx].venue = new_venue
    elif mutation_type == 'time' and len(mutated) > 0:
        idx = random.randint(0, len(mutated) - 1)
        old_time = mutated[idx].time
        new_time = random.choice([t for t in match_times if t != old_time])
        mutated[idx].time = new_time
    
    return mutated


# ==================== GENETIC ALGORITHM MAIN LOOP ====================

def run_genetic_algorithm(
    population_size=50,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elitism_count=2,
    selection_method='tournament',
    tournament_size=3,
    crossover_method='single_point',
    verbose=True
):
    """
    Run the complete genetic algorithm
    
    Parameters:
    -----------
    population_size : int
        Number of individuals in the population
    generations : int
        Number of generations to evolve
    mutation_rate : float
        Probability of mutation (0.0 to 1.0)
    crossover_rate : float
        Probability of crossover (0.0 to 1.0)
    elitism_count : int
        Number of best individuals to preserve
    selection_method : str
        'tournament', 'roulette', or 'rank'
    tournament_size : int
        Size of tournament for tournament selection
    crossover_method : str
        'single_point' or 'two_point'
    verbose : bool
        Print progress information
    
    Returns:
    --------
    best_schedule : List[Match]
        Best schedule found
    best_fitness : float
        Fitness of best schedule
    history : Dict
        Evolution history with fitness statistics
    """
    
    # Initialize population
    if verbose:
        print(f"Initializing population of {population_size} schedules...")
    
    population = [generate_weekly_schedule(teams, venues, all_dates, match_times) 
                  for _ in range(population_size)]
    
    # Evaluate initial population
    fitness_scores = [compute_fitness(schedule) for schedule in population]
    
    # Track history
    history = {
        'best_fitness': [],
        'avg_fitness': [],
        'worst_fitness': [],
        'generation': []
    }
    
    best_fitness = max(fitness_scores)
    best_schedule = copy.deepcopy(population[fitness_scores.index(best_fitness)])
    
    if verbose:
        print(f"Initial best fitness: {best_fitness:.2f}")
    
    # Evolution loop
    for generation in range(generations):
        # Record statistics
        history['generation'].append(generation)
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(sum(fitness_scores) / len(fitness_scores))
        history['worst_fitness'].append(min(fitness_scores))
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        elite_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], 
                               reverse=True)[:elitism_count]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(population[idx]))
        
        # Generate remaining population
        while len(new_population) < population_size:
            # Selection
            if selection_method == 'tournament':
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)
            elif selection_method == 'roulette':
                parent1 = roulette_wheel_selection(population, fitness_scores)
                parent2 = roulette_wheel_selection(population, fitness_scores)
            elif selection_method == 'rank':
                parent1 = rank_selection(population, fitness_scores)
                parent2 = rank_selection(population, fitness_scores)
            else:
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)
            
            # Crossover
            if random.random() < crossover_rate:
                if crossover_method == 'two_point':
                    child = two_point_crossover(parent1, parent2)
                else:
                    child = single_point_crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1 if random.random() < 0.5 else parent2)
            
            # Mutation
            child = apply_mutation(child, venues, match_times, mutation_rate)
            
            # Validation and repair
            is_valid, errors = validate_schedule(child, teams, min_rest_days=4)
            if not is_valid:
                child = repair_schedule(child, teams, venues, match_times, all_dates, min_rest_days=4)
                # If still invalid or lost too many matches, use parent
                if len(child) < len(parent1) * 0.8:  # Lost too many matches
                    child = copy.deepcopy(parent1)
            
            new_population.append(child)
        
        # Update population
        population = new_population
        fitness_scores = [compute_fitness(schedule) for schedule in population]
        
        # Update best
        current_best = max(fitness_scores)
        if current_best > best_fitness:
            best_fitness = current_best
            best_schedule = copy.deepcopy(population[fitness_scores.index(current_best)])
        
        if verbose and (generation + 1) % 10 == 0:
            print(f"Generation {generation + 1}/{generations}: "
                  f"Best={best_fitness:.2f}, "
                  f"Avg={history['avg_fitness'][-1]:.2f}, "
                  f"Worst={history['worst_fitness'][-1]:.2f}")
    
    if verbose:
        print(f"\nFinal best fitness: {best_fitness:.2f}")
    
    return best_schedule, best_fitness, history


# ==================== EXPERIMENT RUNNER ====================

def run_experiments(experiment_configs, num_runs=3):
    """
    Run multiple experiments with different parameter configurations
    
    Parameters:
    -----------
    experiment_configs : List[Dict]
        List of parameter configurations to test
    num_runs : int
        Number of runs per configuration for averaging
    
    Returns:
    --------
    results : List[Dict]
        Results for each experiment configuration
    """
    results = []
    
    print("=" * 80)
    print("GENETIC ALGORITHM EXPERIMENTS")
    print("=" * 80)
    
    for exp_idx, config in enumerate(experiment_configs):
        print(f"\n{'='*80}")
        print(f"Experiment {exp_idx + 1}/{len(experiment_configs)}")
        print(f"Configuration: {config}")
        print(f"{'='*80}\n")
        
        run_results = []
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}...", end=" ")
            
            best_schedule, best_fitness, history = run_genetic_algorithm(
                population_size=config.get('population_size', 50),
                generations=config.get('generations', 100),
                mutation_rate=config.get('mutation_rate', 0.1),
                crossover_rate=config.get('crossover_rate', 0.8),
                elitism_count=config.get('elitism_count', 2),
                selection_method=config.get('selection_method', 'tournament'),
                tournament_size=config.get('tournament_size', 3),
                crossover_method=config.get('crossover_method', 'single_point'),
                verbose=False
            )
            
            run_results.append({
                'best_fitness': best_fitness,
                'final_avg_fitness': history['avg_fitness'][-1],
                'final_worst_fitness': history['worst_fitness'][-1],
                'history': history
            })
            
            print(f"Best fitness: {best_fitness:.2f}")
        
        # Calculate statistics
        best_fitnesses = [r['best_fitness'] for r in run_results]
        avg_fitnesses = [r['final_avg_fitness'] for r in run_results]
        
        result = {
            'config': config,
            'runs': run_results,
            'statistics': {
                'best_fitness_mean': sum(best_fitnesses) / len(best_fitnesses),
                'best_fitness_std': (sum((x - sum(best_fitnesses)/len(best_fitnesses))**2 
                                     for x in best_fitnesses) / len(best_fitnesses))**0.5,
                'best_fitness_max': max(best_fitnesses),
                'best_fitness_min': min(best_fitnesses),
                'avg_fitness_mean': sum(avg_fitnesses) / len(avg_fitnesses),
            }
        }
        
        results.append(result)
        
        print(f"\nStatistics:")
        print(f"  Best Fitness - Mean: {result['statistics']['best_fitness_mean']:.2f}, "
              f"Std: {result['statistics']['best_fitness_std']:.2f}, "
              f"Max: {result['statistics']['best_fitness_max']:.2f}, "
              f"Min: {result['statistics']['best_fitness_min']:.2f}")
    
    return results


def save_results(results, filename='experiment_results.json'):
    """Save experiment results to JSON file"""
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_result = {
            'config': result['config'],
            'statistics': result['statistics'],
            'runs': []
        }
        # Only save final statistics for each run (not full history)
        for run in result['runs']:
            json_result['runs'].append({
                'best_fitness': run['best_fitness'],
                'final_avg_fitness': run['final_avg_fitness'],
                'final_worst_fitness': run['final_worst_fitness']
            })
        json_results.append(json_result)
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {filename}")


def save_results_csv(results, filename='experiment_results.csv'):
    """Save experiment results summary to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Experiment', 'Population Size', 'Generations', 'Mutation Rate',
            'Crossover Rate', 'Elitism Count', 'Selection Method',
            'Tournament Size', 'Crossover Method',
            'Best Fitness Mean', 'Best Fitness Std', 'Best Fitness Max', 'Best Fitness Min',
            'Avg Fitness Mean'
        ])
        
        for idx, result in enumerate(results):
            config = result['config']
            stats = result['statistics']
            writer.writerow([
                idx + 1,
                config.get('population_size', 'N/A'),
                config.get('generations', 'N/A'),
                config.get('mutation_rate', 'N/A'),
                config.get('crossover_rate', 'N/A'),
                config.get('elitism_count', 'N/A'),
                config.get('selection_method', 'N/A'),
                config.get('tournament_size', 'N/A'),
                config.get('crossover_method', 'N/A'),
                f"{stats['best_fitness_mean']:.2f}",
                f"{stats['best_fitness_std']:.2f}",
                f"{stats['best_fitness_max']:.2f}",
                f"{stats['best_fitness_min']:.2f}",
                f"{stats['avg_fitness_mean']:.2f}"
            ])
    
    print(f"Results summary saved to {filename}")


def compare_results(results):
    """Print a comparison of all experiment results"""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    
    # Sort by best fitness mean
    sorted_results = sorted(results, 
                           key=lambda x: x['statistics']['best_fitness_mean'], 
                           reverse=True)
    
    print(f"\n{'Exp':<5} {'Pop':<5} {'Gen':<5} {'Mut':<6} {'Sel':<12} {'Best Mean':<12} {'Best Max':<12} {'Best Min':<12}")
    print("-" * 80)
    
    for idx, result in enumerate(sorted_results):
        config = result['config']
        stats = result['statistics']
        print(f"{idx+1:<5} "
              f"{config.get('population_size', 'N/A'):<5} "
              f"{config.get('generations', 'N/A'):<5} "
              f"{config.get('mutation_rate', 'N/A'):<6.2f} "
              f"{config.get('selection_method', 'N/A'):<12} "
              f"{stats['best_fitness_mean']:<12.2f} "
              f"{stats['best_fitness_max']:<12.2f} "
              f"{stats['best_fitness_min']:<12.2f}")
    
    print("\n" + "=" * 80)
    print(f"Best Configuration: Experiment {sorted_results[0]['config']}")
    print(f"Best Fitness Achieved: {sorted_results[0]['statistics']['best_fitness_max']:.2f}")
    print("=" * 80)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Define experiment configurations (reduced from 11 to 6 experiments)
    experiments = [
        # Baseline
        {
            'name': 'Baseline',
            'population_size': 30,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test population sizes
        {
            'name': 'Large Population',
            'population_size': 100,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test mutation rates
        {
            'name': 'High Mutation',
            'population_size': 50,
            'generations': 50,
            'mutation_rate': 0.3,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test selection methods
        {
            'name': 'Roulette Selection',
            'population_size': 50,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'roulette',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test crossover methods
        {
            'name': 'Two-Point Crossover',
            'population_size': 50,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'two_point'
        },
        # Best combination
        {
            'name': 'Best Combination',
            'population_size': 100,
            'generations': 100,
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elitism_count': 5,
            'selection_method': 'tournament',
            'tournament_size': 5,
            'crossover_method': 'two_point'
        }
    ]
    
    # Run experiments (with fewer runs for faster execution)
    print("Starting experiments...")
    results = run_experiments(experiments, num_runs=3)
    
    # Compare and save results
    compare_results(results)
    save_results(results, 'experiment_results.json')
    save_results_csv(results, 'experiment_results.csv')
    
    print("\nExperiments completed!")

