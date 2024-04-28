def density_calc (pedestrian, square_meter):
    # Calculate the density of people per square meter
    density = pedestrian / square_meter
    
    # Define the density ranges for each level of service
    density_ranges = {
        'A': (0, 0.308),
        'B': (0.308, 0.431),
        'C': (0.431, 0.718),
        'D': (0.718, 1.076),
        'E': (1.076, 2.153),
        'F': (2.153, 10000)
    }
    
    # Determine the level of service based on the density
    for level, (min_density, max_density) in density_ranges.items():
        if (min_density is None and density <= max_density) or \
           (max_density is None and density > min_density) or \
           (min_density <= density <= max_density):
            return level

    # If density does not fall into any category, return an error message
    return "Invalid density value"