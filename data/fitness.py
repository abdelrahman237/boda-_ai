"""
Fitness Function for Sports Tournament Scheduling
Calculates fitness score for tournament schedules
"""


def compute_fitness(schedule, min_rest_days=4, weights=None):
    """
    Calculate fitness score for a schedule (higher is better)
    
    Parameters:
    -----------
    schedule : List[Match]
        List of matches in the schedule
    min_rest_days : int
        Minimum rest days required between matches for a team
    weights : dict, optional
        Weights for different penalty types
        
    Returns:
    --------
    float
        Fitness score (0-100, higher is better)
    """
    if weights is None:
        weights = {
            'venue_conflict': 1.0, 
            'rest_violation': 0.7, 
            'repeated_opponent': 0.0, 
            'time_balance': 0.3
        }

    penalty = 0
    venue_day_time = {}
    last_played = {}
    matches_per_pair = {}
    team_times = {}

    for match in schedule:
        # Check venue conflicts (same venue, date, time)
        key = (match.date, match.time, match.venue)
        if key in venue_day_time:
            penalty += weights['venue_conflict']
        else:
            venue_day_time[key] = True

        # Check rest day violations
        for team in [match.team1, match.team2]:
            if team in last_played:
                delta_days = (match.date - last_played[team]).days
                if delta_days < min_rest_days:
                    penalty += weights['rest_violation']
            last_played[team] = match.date

        # Track repeated opponents (for round-robin tracking)
        pair = tuple(sorted([match.team1, match.team2]))
        matches_per_pair[pair] = matches_per_pair.get(pair, 0) + 1

        # Track time distribution per team
        for team in [match.team1, match.team2]:
            if team not in team_times:
                team_times[team] = {}
            team_times[team][match.time] = team_times[team].get(match.time, 0) + 1

    # Penalty for unbalanced time distribution
    for team, times in team_times.items():
        if len(times) > 1:
            diff = abs(times.get('17:00', 0) - times.get('20:00', 0))
            penalty += diff * weights['time_balance']

    fitness = max(0, 100 - penalty)
    return fitness


def compute_fitness_verbose(schedule, min_rest_days=4, weights=None):
    """
    Calculate fitness score with detailed breakdown (for debugging)
    
    Returns:
    --------
    tuple
        (fitness_score, breakdown_dict)
    """
    if weights is None:
        weights = {
            'venue_conflict': 1.0, 
            'rest_violation': 0.7, 
            'repeated_opponent': 0.0, 
            'time_balance': 0.3
        }

    penalty = 0
    venue_day_time = {}
    last_played = {}
    matches_per_pair = {}
    team_times = {}

    venue_conflict_count = 0
    rest_violation_count = 0
    repeated_opponents_count = 0
    time_balance_penalty = 0

    for match in schedule:
        # Check venue conflicts
        key = (match.date, match.time, match.venue)
        if key in venue_day_time:
            penalty += weights['venue_conflict']
            venue_conflict_count += 1
        else:
            venue_day_time[key] = True

        # Check rest day violations
        for team in [match.team1, match.team2]:
            if team in last_played:
                delta_days = (match.date - last_played[team]).days
                if delta_days < min_rest_days:
                    penalty += weights['rest_violation']
                    rest_violation_count += 1
            last_played[team] = match.date

        # Track repeated opponents
        pair = tuple(sorted([match.team1, match.team2]))
        if pair in matches_per_pair:
            penalty += weights['repeated_opponent']
            repeated_opponents_count += 1
        matches_per_pair[pair] = matches_per_pair.get(pair, 0) + 1

        # Track time distribution
        for team in [match.team1, match.team2]:
            if team not in team_times:
                team_times[team] = {}
            team_times[team][match.time] = team_times[team].get(match.time, 0) + 1

    # Penalty for unbalanced time distribution
    for team, times in team_times.items():
        if len(times) > 1:
            diff = abs(times.get('17:00', 0) - times.get('20:00', 0))
            penalty += diff * weights['time_balance']
            time_balance_penalty += diff

    fitness = max(0, 100 - penalty)

    breakdown = {
        'fitness_score': fitness,
        'venue_conflicts': venue_conflict_count,
        'rest_violations': rest_violation_count,
        'repeated_opponents': repeated_opponents_count,
        'time_balance_penalty': time_balance_penalty,
        'total_penalty': penalty
    }

    return fitness, breakdown

