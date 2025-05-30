"""
MIT License

Copyright (c) 2025 Will Maddock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from decimal import Decimal, getcontext
import re
from datetime import datetime
import json
import functools

# Set precision for Decimal
getcontext().prec = 60

# Constants
BASE_CU = Decimal('3094134044923.509753564772922302508810')  # CU-Time at JDN 2451545.0 (2000-01-01 12:00 UTC)
SUB_ZTOM_CU = Decimal('3094133911800.949548')
ANCHOR_JDN = Decimal('2451545.0')  # Noon UTC, 2000-01-01
DAYS_PER_YEAR = Decimal('365.2425')
SECONDS_PER_DAY = Decimal('86400')

# Reference date for time difference calculations
REFERENCE_DATE = datetime(2025, 5, 26, 9, 48, 0)

# Cache with LRU to limit memory usage
@functools.lru_cache(maxsize=1000)
def get_jdn(year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0):
    """
    Calculate Julian Day Number (JDN) from a Gregorian date and time.

    Args:
        year (int): Year (positive for CE, negative for BCE)
        month (int): Month (1-12)
        day (int): Day (1-31)
        hour (int): Hour (0-23)
        minute (int): Minute (0-59)
        second (int): Second (0-59)
        microsecond (int): Microsecond (0-999999)

    Returns:
        Decimal: JDN, where integer part is the day (noon-based) and fraction is time.

    Notes:
        For |year| > 999,999,999, uses an approximation based on average days per year.
    """
    year = Decimal(year)
    if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour < 24 and
            0 <= minute < 60 and 0 <= second < 60 and 0 <= microsecond < 1000000):
        raise ValueError("Invalid date or time component")

    if year.abs() > 999999999:
        delta_years = year - 2000
        days = delta_years * DAYS_PER_YEAR
        fraction = (Decimal(hour) / 24 + Decimal(minute) / 1440 +
                    Decimal(second) / 86400 + Decimal(microsecond) / 1000000 / 86400)
        jdn = ANCHOR_JDN + days + fraction
    else:
        a = (14 - month) // 12
        y = year + 4800 - a
        m = month + 12 * a - 3
        jdn_int = (day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045)
        fraction = (Decimal(hour) / 24 + Decimal(minute) / 1440 +
                    Decimal(second) / 86400 + Decimal(microsecond) / 1000000 / 86400)
        jdn = Decimal(jdn_int) + fraction
    return jdn

@functools.lru_cache(maxsize=1000)
def compute_cu_time(year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0):
    """
    Convert a Gregorian date and time to CU-Time.

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day (1-31)
        hour (int): Hour (0-23)
        minute (int): Minute (0-59)
        second (int): Second (0-59)
        microsecond (int): Microsecond (0-999999)

    Returns:
        Decimal: CU-Time value
    """
    year = Decimal(year)
    if year == -28000000000:
        return SUB_ZTOM_CU
    if year.abs() > 28000000000:
        delta_years = year - 2000
        cu_time = BASE_CU + delta_years
    else:
        jdn = get_jdn(year, month, day, hour, minute, second, microsecond)
        delta_jdn = jdn - ANCHOR_JDN
        delta_years = delta_jdn / DAYS_PER_YEAR
        cu_time = BASE_CU + delta_years
    return cu_time.quantize(Decimal('0.000000000000000000000001'))

def jdn_to_date(jdn):
    """
    Convert JDN to Gregorian date and time, adjusting for JDN's noon-based reference.

    Args:
        jdn (Decimal): Julian Day Number

    Returns:
        dict: {'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'}
    """
    jd_int = int(jdn)
    fraction = jdn - jd_int
    if abs(jdn - 1721424.5) / DAYS_PER_YEAR > 28000000000:
        year = int((jdn - 1721424.5) / DAYS_PER_YEAR)
        return {'year': f"January 1, {year}", 'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}

    # Adjust fraction for noon-based JDN to midnight-based time
    adjusted_fraction = (fraction + Decimal('0.5')) % Decimal('1.0')
    total_seconds = adjusted_fraction * SECONDS_PER_DAY
    hours = int(total_seconds // 3600)
    remainder = total_seconds % 3600
    minutes = int(remainder // 60)
    seconds = int(remainder % 60)
    microseconds = int((remainder - seconds) * 1000000)

    # Compute date using integer part
    z = jd_int + 0.5
    alpha = int((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - int(alpha / 4)
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return {'year': year, 'month': month, 'day': day, 'hour': hours, 'minute': minutes, 'second': seconds, 'microsecond': microseconds}

def cu_time_to_gregorian(cu_time):
    """
    Convert CU-Time to Gregorian date and time.

    Args:
        cu_time (Decimal): CU-Time value

    Returns:
        dict: {'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'}
    """
    delta_years = cu_time - BASE_CU
    delta_days = delta_years * DAYS_PER_YEAR
    jdn = ANCHOR_JDN + delta_days
    return jdn_to_date(jdn)

def parse_date_input(input_str):
    """
    Parse date strings in natural language or ISO format.

    Args:
        input_str (str): Date string (e.g., 'January 1st, 50 billion BCE' or '1981-10-22 07:38:42 UTC')

    Returns:
        dict: {'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'}
    """
    months = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
              'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}

    # ISO format with optional time (e.g., "1981-10-22" or "1981-10-22 07:38:42 UTC")
    iso_match = re.match(r'(\d{4})-(\d{2})-(\d{2})(?:\s+(\d{2}):(\d{2}):(\d{2})(?:\s*UTC)?)?', input_str)
    if iso_match:
        year, month, day = map(int, iso_match.groups()[:3])
        time_parts = iso_match.groups()[3:]
        hour = int(time_parts[0]) if time_parts[0] else 0
        minute = int(time_parts[1]) if time_parts[1] else 0
        second = int(time_parts[2]) if time_parts[2] else 0
        if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour < 24 and
                0 <= minute < 60 and 0 <= second < 60):
            raise ValueError("Invalid date or time components in ISO format")
        return {'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second, 'microsecond': 0}

    # Natural language format (e.g., "January 1st, 50 billion BCE")
    match = re.match(r'(\w+)\s+(\d+)(?:st|nd|rd|th)?,\s*(\d+)\s*(BCE)?', input_str, re.IGNORECASE)
    if match:
        month_str, day, year_str, era = match.groups()
        month = months.get(month_str.lower())
        if not month:
            raise ValueError(f"Invalid month: {month_str}")
        day = int(day)
        year = int(year_str)
        if era and era.upper() == 'BCE':
            year = -year
        if not (1 <= month <= 12 and 1 <= day <= 31):
            raise ValueError("Invalid date components in natural language format")
        return {'year': year, 'month': month, 'day': day, 'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}
    raise ValueError("Invalid date format. Use 'Month Day, Year [BCE]' or 'YYYY-MM-DD [HH:MM:SS UTC]'")

def get_cosmic_phase(cu_time):
    """Determine the cosmic phase based on CU-Time."""
    if cu_time < Decimal('3.08e12'):
        return "Speculative Phase (Pre-expansion, theoretical)"
    elif cu_time <= Decimal('3.11e12'):
        return "Dark Energy Phase (Expansion, sub-ztom to atom)"
    elif cu_time <= Decimal('3.416e12'):
        return "Anti-Dark Energy Phase (Compression, btom to ztom, Empowered by God’s Free Will)"
    else:
        return "Speculative Phase (Beyond known cosmic phases)"

def get_dominant_forces(cu_time):
    """Determine dominant forces based on CU-Time."""
    phase = get_cosmic_phase(cu_time)
    if "Anti-Dark Energy Phase" in phase:
        return ["matter-antimatter"]
    elif "Dark Energy Phase" in phase:
        return ["matter"]
    else:
        return ["anti-dark-matter (theoretical)"]

def format_cu_value(cu_value):
    """Format CU-Time into human-friendly, full numeric, and exponential forms."""
    cu_value = cu_value.quantize(Decimal('0.000000000000000000000001'))
    full_numeric = f"{cu_value:.24f}"
    exponential = f"{cu_value:.6e}"
    integer_part, fraction_part = full_numeric.split('.')
    int_val = int(integer_part)
    trillion = int_val // 1000000000000
    remainder = int_val % 1000000000000
    billion = remainder // 1000000000
    remainder %= 1000000000
    million = remainder // 1000000
    remainder %= 1000000
    thousand = remainder // 1000
    units = remainder % 1000
    parts = []
    if trillion:
        parts.append(f"{trillion} Trillion")
    if billion:
        parts.append(f"{billion} Billion")
    if million:
        parts.append(f"{million} Million")
    if thousand:
        parts.append(f"{thousand} Thousand")
    if units or not parts:
        parts.append(f"{units}")
    human_friendly = " ".join(parts) + f".{fraction_part} CU-Time"
    return {
        "human_friendly": human_friendly,
        "full_numeric": full_numeric,
        "exponential": exponential
    }

def convert_time(input_data):
    """
    Convert input data to CU-Time or Gregorian date, supporting strings, dictionaries, lists, and CU-Time values.

    Args:
        input_data: String, dict, list, or Decimal (CU-Time)

    Returns:
        dict: Conversion result with status, date, CU-Time, and context
    """
    if isinstance(input_data, list):
        return [convert_time(item) for item in input_data]

    elif isinstance(input_data, str):
        # Check if input is a CU-Time value
        if re.match(r'^\d+(\.\d+)?$', input_data):
            try:
                cu_time = Decimal(input_data)
                date_dict = cu_time_to_gregorian(cu_time)
                formatted_cu = format_cu_value(cu_time)
                phase = get_cosmic_phase(cu_time)
                forces = get_dominant_forces(cu_time)
                epoch = "Past" if date_dict['year'] < 0 else "Future"
                ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                              REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second)
                time_diff = abs(cu_time - ref_cu_time)
                years_diff = time_diff if abs(cu_time - BASE_CU) > 28000000000 else time_diff * DAYS_PER_YEAR
                note = "Speculative conversion; theoretical due to extreme time range" if abs(cu_time - BASE_CU) > 28000000000 else ""
                return {
                    "input": input_data,
                    "cu_time": formatted_cu,
                    "gregorian_date": f"{date_dict['month']:02d}/{date_dict['day']:02d}/{date_dict['year']} {date_dict['hour']:02d}:{date_dict['minute']:02d}:{date_dict['second']:02d}",
                    "context": {"phase": phase, "forces": forces, "epoch": epoch},
                    "time_difference": f"{time_diff:.6f} CU-Time (~{int(years_diff)} years)",
                    "status": "success",
                    "note": note
                }
            except (ValueError, TypeError):
                return {"input": input_data, "status": "error", "message": "Invalid CU-Time value"}

        # Parse as date string
        try:
            parsed = parse_date_input(input_data)
            year, month, day = parsed['year'], parsed['month'], parsed['day']
            hour, minute, second = parsed['hour'], parsed['minute'], parsed['second']
            cu_time = compute_cu_time(year, month, day, hour, minute, second, parsed['microsecond'])
            formatted_cu = format_cu_value(cu_time)
            phase = get_cosmic_phase(cu_time)
            forces = get_dominant_forces(cu_time)
            epoch = "Past" if year < 0 else "Future"
            ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                          REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second)
            time_diff = abs(cu_time - ref_cu_time)
            years_diff = abs(year - REFERENCE_DATE.year) if abs(year) <= 28000000000 else time_diff
            note = "Speculative conversion; theoretical due to extreme time range" if abs(year) > 28000000000 else ""
            return {
                "input": input_data,
                "cu_time": formatted_cu,
                "gregorian_date": f"{month:02d}/{day:02d}/{year} {hour:02d}:{minute:02d}:{second:02d}",
                "context": {"phase": phase, "forces": forces, "epoch": epoch},
                "time_difference": f"{time_diff:.6f} CU-Time (~{int(years_diff)} years)",
                "status": "success",
                "note": note
            }
        except ValueError as e:
            return {"input": input_data, "status": "error", "message": str(e)}

    elif isinstance(input_data, dict):
        year = input_data.get('year', 2000)
        month = input_data.get('month', 1)
        day = input_data.get('day', 1)
        hour = input_data.get('hour', 0)
        minute = input_data.get('minute', 0)
        second = input_data.get('second', 0)
        microsecond = input_data.get('microsecond', 0)
        if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour < 24 and
                0 <= minute < 60 and 0 <= second < 60 and 0 <= microsecond < 1000000):
            return {"input": str(input_data), "status": "error", "message": "Invalid date or time components"}
        cu_time = compute_cu_time(year, month, day, hour, minute, second, microsecond)
        formatted_cu = format_cu_value(cu_time)
        phase = get_cosmic_phase(cu_time)
        forces = get_dominant_forces(cu_time)
        epoch = "Past" if year < 0 else "Future"
        ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                      REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second)
        time_diff = abs(cu_time - ref_cu_time)
        years_diff = abs(year - REFERENCE_DATE.year) if abs(year) <= 28000000000 else time_diff
        note = "Speculative conversion; theoretical due to extreme time range" if abs(year) > 28000000000 else ""
        return {
            "input": str(input_data),
            "cu_time": formatted_cu,
            "gregorian_date": f"{month:02d}/{day:02d}/{year} {hour:02d}:{minute:02d}:{second:02d}",
            "context": {"phase": phase, "forces": forces, "epoch": epoch},
            "time_difference": f"{time_diff:.6f} CU-Time (~{int(years_diff)} years)",
            "status": "success",
            "note": note
        }

    elif isinstance(input_data, (int, float, Decimal)):
        try:
            cu_time = Decimal(input_data)
            date_dict = cu_time_to_gregorian(cu_time)
            formatted_cu = format_cu_value(cu_time)
            phase = get_cosmic_phase(cu_time)
            forces = get_dominant_forces(cu_time)
            epoch = "Past" if date_dict['year'] < 0 else "Future"
            ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                          REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second)
            time_diff = abs(cu_time - ref_cu_time)
            years_diff = time_diff if abs(cu_time - BASE_CU) > 28000000000 else time_diff * DAYS_PER_YEAR
            note = "Speculative conversion; theoretical due to extreme time range" if abs(cu_time - BASE_CU) > 28000000000 else ""
            return {
                "input": str(input_data),
                "cu_time": formatted_cu,
                "gregorian_date": f"{date_dict['month']:02d}/{date_dict['day']:02d}/{date_dict['year']} {date_dict['hour']:02d}:{date_dict['minute']:02d}:{date_dict['second']:02d}",
                "context": {"phase": phase, "forces": forces, "epoch": epoch},
                "time_difference": f"{time_diff:.6f} CU-Time (~{int(years_diff)} years)",
                "status": "success",
                "note": note
            }
        except (ValueError, TypeError):
            return {"input": str(input_data), "status": "error", "message": "Invalid CU-Time value"}

    else:
        return {"input": str(input_data), "status": "error", "message": "Unsupported input type"}

# Example usage
if __name__ == "__main__":
    # Test your sample inputs
    result1 = convert_time("10/22/1981 07:38:42 UTC")
    print(json.dumps(result1, indent=2))

    result2 = convert_time("3094134044905.317218184916904170293200")
    print(json.dumps(result2, indent=2))

    # Additional test cases
    result3 = convert_time("January 1st, 50 billion BCE")
    print(json.dumps(result3, indent=2))

    result4 = convert_time({"year": 2029, "month": 1, "day": 1})
    print(json.dumps(result4, indent=2))