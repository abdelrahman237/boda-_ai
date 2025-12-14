"""
Data module for Sports Tournament Scheduling
Contains shared models, fitness functions, and data

INTEGRATED MODULES:
- Task 1: schedule_generator (generate_weekly_schedule)
- Task 2: fitness (compute_fitness)
- Task 3: genetic_operations (crossover, mutation, selection)
"""

from .models import Match
from .fitness import compute_fitness, compute_fitness_verbose
from .teams_venues_times import teams, venues, match_times
from .schedule_generator import generate_weekly_schedule  # Task 1
from .genetic_operations import (  # Task 3
    single_point_crossover,
    two_point_crossover,
    uniform_crossover,
    swap_mutation,
    change_venue_mutation,
    change_time_mutation,
    tournament_selection,
    roulette_wheel_selection,
    rank_selection
)

__all__ = [
    # Models
    'Match',
    # Fitness (Task 2)
    'compute_fitness',
    'compute_fitness_verbose',
    # Data
    'teams',
    'venues',
    'match_times',
    # Schedule Generator (Task 1)
    'generate_weekly_schedule',
    # Genetic Operations (Task 3)
    'single_point_crossover',
    'two_point_crossover',
    'uniform_crossover',
    'swap_mutation',
    'change_venue_mutation',
    'change_time_mutation',
    'tournament_selection',
    'roulette_wheel_selection',
    'rank_selection',
]

