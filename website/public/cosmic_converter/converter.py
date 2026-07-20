"""
Cosmic Universalism Time Converter v1.0.9 — Final Version

This module converts between CU-Time (trillions of CU-years), NASA Time (billions of CU-years),
and Gregorian dates.

For the NASA branch, the conversion formula is:
    Gregorian = BASE_DATE_NASA − ((NASA_REFERENCE − input) × SECONDS_PER_YEAR)
so that an input of 13,796,999,982.79 (NASA Time, in billions) yields approximately
"2007-10-15 00:00:00 UTC" (which in America/Denver appears as roughly "2007-10-14 18:00:00 MDT").

The NASA Time is then converted to CU-Time using:
    (BASE_CU + CU_OFFSET) / NASA_REFERENCE

The CU branch uses BASE_DATE_UTC.

Constants:
  • BASE_CU and CU_OFFSET anchor the CU-Time scale.
  • SECONDS_PER_YEAR is based on 365.2425 days.
  • BASE_DATE_NASA is tuned for the NASA branch.
  • Adjust these constants further for extra precision if needed.
"""

from decimal import Decimal
from datetime import datetime, timedelta
import pytz
import math

# Constants (v1.0.9)
BASE_CU = Decimal('3079913911800.94954834')   # CU-Time anchor (4 BCE)
CU_OFFSET = Decimal('335739.82')               # Calibration offset for CU-Time
SECONDS_PER_YEAR = Decimal('31557600')         # 365.2425 days × 86400 seconds
COSMIC_LIFESPAN = Decimal('13.8e9')              # CU universe age (years)
NASA_LIFESPAN = Decimal('13.797e9')              # NASA universe age (years)
CONVERGENCE_YEAR = Decimal('2029')               # Convergence year constant

# For the CU-Time branch ("cosmic" branch)
BASE_DATE_UTC = datetime(2025, 5, 16, 18, 30, 0, tzinfo=pytz.UTC)

# We choose BASE_DATE_NASA so that the forward conversion yields:
#   Gregorian = BASE_DATE_NASA – ((NASA_REFERENCE – input) × SECONDS_PER_YEAR)
# For input = 13,796,999,982.79, Δ ≈ 17.21 billions,
# Total seconds = 17.21 × 31557600 ≈ 543,106,296 sec (~6286 days).
# Then, BASE_DATE_NASA is set to:
#   "2007-10-15 00:00:00 UTC" + 543,106,296 sec ≈ December 30, 2024, 04:48:00 UTC.
BASE_DATE_NASA = datetime(2024, 12, 30, 4, 48, 0, tzinfo=pytz.UTC)

NASA_REFERENCE = Decimal('13797000000')  # NASA Time reference (billions)

def cu_to_gregorian(cu_time, timezone="UTC", scales=None, calibration="cu", compression_model="logarithmic"):
    """
    Convert a given CU-Time or NASA Time value to a Gregorian date.

    For NASA calibration, the input (in billions) uses BASE_DATE_NASA.
    Then NASA Time is converted to CU-Time using the scaling factor:
      (BASE_CU + CU_OFFSET) / NASA_REFERENCE.
    """
    cu_decimal = Decimal(str(cu_time))
    # Determine whether to treat the input as NASA Time or CU-Time.
    is_nasa = (calibration.lower() == "nasa") or (cu_decimal >= Decimal('1e9') and cu_decimal < Decimal('1e11'))

    if is_nasa:
        # NASA branch:
        years_from_reference = NASA_REFERENCE - cu_decimal  # in billions
        total_seconds = years_from_reference * SECONDS_PER_YEAR
        gregorian_time = BASE_DATE_NASA - timedelta(seconds=float(total_seconds))
        # Convert NASA Time to its CU-Time equivalent.
        cu_time_in_cu = cu_decimal * ((BASE_CU + CU_OFFSET) / NASA_REFERENCE)
        nasa_time = cu_decimal
    else:
        # CU-Time branch:
        ratio = COSMIC_LIFESPAN / CONVERGENCE_YEAR
        cu_diff = cu_decimal - (BASE_CU + CU_OFFSET)
        gregorian_seconds = (cu_diff * SECONDS_PER_YEAR) / ratio
        if compression_model.lower() == "polynomial" and gregorian_seconds < (Decimal(-31557600) * Decimal('1000')):
            gregorian_seconds *= Decimal('1.0001')
        gregorian_time = BASE_DATE_UTC + timedelta(seconds=int(gregorian_seconds))
        cu_time_in_cu = cu_decimal
        nasa_time = cu_time_in_cu * Decimal('0.9997826')

    tz = pytz.timezone(timezone) if timezone != "UTC" else pytz.UTC
    formatted_time = gregorian_time.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S %Z')

    output_lines = [f"Gregorian: {formatted_time}"]
    if scales:
        if "geological" in scales:
            output_lines.append("Geological: Holocene")
        if "cosmic" in scales:
            output_lines.append("Cosmic: Dec-31")
        if "patterns" in scales:
            digits = str(cu_time).replace('.', '')
            digit_freq = {str(i): digits.count(str(i)) for i in range(10)}
            entropy = -sum((count / len(digits)) * math.log2(count / len(digits))
                           for count in digit_freq.values() if count)
            note = "High 9s suggest medieval bias" if digit_freq.get('9', 0) > 3 else "Balanced digits"
            output_lines.append(f"Patterns: {{ 'digit_frequency': {digit_freq}, 'entropy': {entropy:.1f}, 'note': '{note}' }}")

    header = (f"║ {nasa_time:,.2f} NASA Time ║") if is_nasa else (f"║ {cu_time_in_cu:,.2f} CU-Time ║")
    border = "╔" + "═" * 45 + "╗"
    footer = "╚" + "═" * 45 + "╝"

    return '\n'.join([border, header, footer] + output_lines)

def gregorian_to_cu(gregorian_time_str, timezone="UTC", lifespan="CU", compression_model="logarithmic"):
    """
    Convert a Gregorian time (as a string) to CU-Time or NASA Time.

    For the NASA branch:
      - Uses BASE_DATE_NASA as the reference.
      - Computes:
            years_from_reference = - (total_seconds / SECONDS_PER_YEAR)
        and calculates NASA Time as:
            NASA_REFERENCE - years_from_reference.
      - Then converts NASA Time to CU-Time using:
            (BASE_CU + CU_OFFSET) / NASA_REFERENCE.

    The CU-Time branch uses the cosmic scaling ratio directly.
    """
    try:
        try:
            parsed_time = datetime.strptime(gregorian_time_str, '%Y-%m-%d %H:%M:%S')
            parsed_time = pytz.timezone(timezone).localize(parsed_time)
        except ValueError:
            parsed_time = datetime.strptime(gregorian_time_str, '%Y-%m-%dT%H:%M:%S%z')

        if lifespan.upper() == "NASA":
            time_diff = parsed_time - BASE_DATE_NASA
        else:
            time_diff = parsed_time - BASE_DATE_UTC

        total_seconds = Decimal(time_diff.total_seconds())

        if lifespan.upper() == "NASA":
            years_from_reference = -total_seconds / SECONDS_PER_YEAR
            nasa_time = NASA_REFERENCE - years_from_reference
            cu_time = nasa_time * ((BASE_CU + CU_OFFSET) / NASA_REFERENCE)
        else:
            ratio = COSMIC_LIFESPAN / CONVERGENCE_YEAR
            cu_diff = (total_seconds * ratio) / SECONDS_PER_YEAR
            if compression_model.lower() == "polynomial" and total_seconds < (Decimal(-31557600) * Decimal('1000')):
                cu_diff /= Decimal('1.0001')
            cu_time = BASE_CU + CU_OFFSET + cu_diff
            nasa_time = cu_time * Decimal('0.9997826')

        header = (f"║ {nasa_time:,.2f} NASA Time ║") if lifespan.upper() == "NASA" else (f"║ {cu_time:,.2f} CU-Time ║")
        border = "╔" + "═" * 45 + "╗"
        footer = "╚" + "═" * 45 + "╝"

        return '\n'.join([
            border,
            header,
            footer,
            f"{lifespan.upper()} Time: {(nasa_time if lifespan.upper()=='NASA' else cu_time):,.2f} CU-years",
            f"{'NASA' if lifespan.upper()=='CU' else 'CU'}-Time: {(cu_time if lifespan.upper()=='NASA' else nasa_time):,.2f} CU-years",
            f"Gregorian: {parsed_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        ])
    except Exception as e:
        return f"Conversion error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    print("=== cu_to_gregorian (NASA branch) ===")
    result1 = cu_to_gregorian(
        13796999982.79,
        timezone="America/Denver",
        scales=["gregorian", "geological", "cosmic"],
        calibration="nasa"
    )
    print(result1)

    print("\n=== gregorian_to_cu (NASA branch) ===")
    result2 = gregorian_to_cu("2007-10-15 00:00:00", timezone="UTC", lifespan="NASA")
    print(result2)
