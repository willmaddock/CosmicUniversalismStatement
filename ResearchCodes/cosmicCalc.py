import math

# Constants
STATE_MAP = {
    "-ttom": -65536,
    "-utom": -16,
    "-vtom": -4,
    "-wtom": -2,
    "-xtom": -1,
    "-ytom": 0,
    "-ztom": 1,
    "atom": 2,
    "btom": 4,
    "ctom": 16,
    "dtom": 65536
}

MAX_SAFE_EXPONENT = 1000  # Practical computation limit


def calculate_geometric_series(start_exp, end_exp):
    """
    Calculate the exact sum of 2^n from start_exp to end_exp using geometric series formula.
    Handles large exponents and provides both numerical and symbolic representations.
    """
    if start_exp > end_exp:
        return 0, [], "Invalid range: start must be ≤ end"

    # Handle large exponents symbolically
    if end_exp > MAX_SAFE_EXPONENT or start_exp < -MAX_SAFE_EXPONENT:
        symbolic = f"2^{end_exp + 1} - 2^{start_exp}"
        return math.inf, [], f"Σ2ⁿ = {symbolic} (symbolic representation)"

    # Exact calculation for manageable exponents
    sum_total = (2 ** (end_exp + 1)) - (2 ** start_exp)
    terms = []

    if end_exp - start_exp <= 10:  # Show steps only for small ranges
        terms = [f"2^{n} = {2 ** n}" for n in range(start_exp, end_exp + 1)]

    return sum_total, terms, f"Σ2ⁿ = {sum_total}"


def display_state_help():
    """Show available states and their meanings"""
    print("\nState Hierarchy (from smallest to largest):")
    for state, exp in sorted(STATE_MAP.items(), key=lambda x: x[1]):
        print(f"{state:>6} : 2^{exp}")


def get_user_range():
    """Get valid input range with enhanced error handling"""
    while True:
        display_state_help()
        print("\nEnter states from the list above")
        start = input("Start state (e.g., -vtom, atom): ").strip()
        end = input("End state (e.g., btom, dtom): ").strip()

        if start not in STATE_MAP or end not in STATE_MAP:
            print("Invalid state name. Please use from the list above.")
            continue

        return STATE_MAP[start], STATE_MAP[end]


def main_menu():
    """Enhanced user interface with better navigation"""
    while True:
        print("\n=== CUME Calculator ===")
        print("1. Calculate full spectrum (-ttom to dtom)")
        print("2. Calculate core range (-vtom to ctom)")
        print("3. Custom range")
        print("4. Understanding the States")
        print("5. Exit")

        choice = input("Choose (1-5): ")

        if choice == "1":
            start, end = -65536, 65536
        elif choice == "2":
            start, end = -4, 16
        elif choice == "3":
            start, end = get_user_range()
        elif choice == "4":
            display_state_help()
            continue
        elif choice == "5":
            print("Exiting cosmic calculation...")
            break
        else:
            print("Invalid choice")
            continue

        total, terms, desc = calculate_geometric_series(start, end)
        print(f"\nResult for range {start} to {end}:")
        print(desc)
        if terms:
            print("\nStep-by-step breakdown:")
            print("\n".join(terms))


if __name__ == "__main__":
    main_menu()