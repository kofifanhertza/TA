def density_calc (pedestrian, square_meter):
    density = pedestrian / square_meter

    if density <= 0.08 : 
        return "A"
    elif 0.08 < density <= 0.27 :
        return "B" 
    elif 0.27 < density <= 0.45 :
        return "C" 
    elif 0.45 < density <= 0.69 :
        return "D" 
    elif 0.69 < density <= 1.66 :
        return "E" 
    else :
        return "F"