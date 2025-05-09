def generate_tom_sequence(n=26):
    toms = []
    for i in range(n):
        if i == 0:
            toms.append(f"-ztom → atom (2.8 trillion years old)")
        else:
            toms.append(f"ztom+{i-1} → atom+{i}")
    return toms


def display_tom_sequence():
    toms = generate_tom_sequence()
    for step in toms:
        print(step)


if __name__ == "__main__":
    display_tom_sequence()