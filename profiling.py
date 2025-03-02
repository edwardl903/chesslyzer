import pstats

# Load the profile results file
profile = pstats.Stats('profile_results.prof')

# Sort by cumulative time
profile.sort_stats('cumtime').print_stats(50)  # Print top 10 functions

# python -m cProfile -o profile_results.prof testing.py EdwardL903 2022
# python profiling.py
