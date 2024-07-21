def density_calc (pedestrian, square_meter):
    # Calculate the density of people per square meter
    if pedestrian == 0 :
        return 'A'

    density = square_meter / pedestrian
    
    # Define the density ranges for each level of service
    density_ranges = {
        'A': (1.9, 10000),
        'B': (1.6, 1.9),
        'C': (1.1, 1.6),
        'D': (0.7, 1.1),
        'E': (0.5, 0.7),
        'F': (0, 0.5)
    }
    
    # Determine the level of service based on the density
    for level, (min_density, max_density) in density_ranges.items():
        if (min_density is None and density <= max_density) or \
           (max_density is None and density > min_density) or \
           (min_density <= density <= max_density):
            return level

    # If density does not fall into any category, return an error message
    return "Invalid density value"