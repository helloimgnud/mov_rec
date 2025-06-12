def map_age(age):
    if age < 18:
        return 0
    elif 18 <= age <= 24:
        return 1/6
    elif 25 <= age <= 34:
        return 2/6
    elif 35 <= age <= 44:
        return 3/6
    elif 45 <= age <= 49:
        return 4/6
    elif 50 <= age <= 55:
        return 5/6
    else:  # age >= 56
        return 1

def map_occupation(occupation):
    return occupation / 20