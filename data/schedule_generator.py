"""
Schedule Generator - Task 1
Generates tournament schedules for sports leagues
Extracted from schedule_model.ipynb
"""

import random
from datetime import datetime, timedelta
from .models import Match


def generate_weekly_schedule(teams, venues, all_dates, match_times, min_rest_days=4, start_date=None):
    """
    Generate complete schedule: 18 teams × 17 opponents = 306 matches
    Each team plays every other team (home and away handled separately)
    
    Parameters:
    -----------
    teams : List[str]
        List of team names
    venues : List[str]
        List of venue names
    all_dates : List[datetime]
        List of all available dates
    match_times : List[str]
        List of available match times
    min_rest_days : int
        Minimum rest days between matches for a team
    start_date : datetime, optional
        Start date for the tournament (used for last_played initialization)
        
    Returns:
    --------
    List[Match]
        Complete schedule of matches
    """
    if start_date is None:
        start_date = all_dates[0] if all_dates else datetime(2025, 5, 1)
    
    # Use start_date for last_played initialization
    
    schedule = []
    last_played = {team: start_date - timedelta(days=min_rest_days) for team in teams}
    day_info = {d: {'count': 0, 'venues': set(), 'times': set()} for d in all_dates}
    pairings = [(t1, t2) for t1 in teams for t2 in teams if t1 != t2]
    random.shuffle(pairings)
    matches_count = {(t1, t2): 0 for t1 in teams for t2 in teams if t1 != t2}
    week_dates = all_dates[::7]

    for week_start in week_dates:
        used_teams = set()
        for home, away in pairings:
            if matches_count[(home, away)] >= 1:
                continue
            if home in used_teams or away in used_teams:
                continue
            possible_days = [d for d in all_dates if week_start <= d < week_start + timedelta(days=7)
                             and (d - last_played[home]).days >= min_rest_days
                             and (d - last_played[away]).days >= min_rest_days
                             and day_info[d]['count'] < 2]
            if not possible_days:
                continue
            random.shuffle(possible_days)
            for date in possible_days:
                available_times = [t for t in match_times if t not in day_info[date]['times']]
                available_venues = [v for v in venues if v not in day_info[date]['venues']]
                if not available_times or not available_venues:
                    continue
                time = random.choice(available_times)
                venue = random.choice(available_venues)
                match = Match(home, away, date, time, venue)
                schedule.append(match)
                last_played[home] = date
                last_played[away] = date
                matches_count[(home, away)] += 1
                used_teams.update([home, away])
                day_info[date]['count'] += 1
                day_info[date]['venues'].add(venue)
                day_info[date]['times'].add(time)
                break

    schedule.sort(key=lambda m: m.date)
    return schedule

