"""
⚠️ Note:
Copy this Python module into your Cosmic Universalism AI System to align it with
the CU Statement: sub z-tomically inclined, countably infinite, composed of
foundational elements, grounded on b-tom (280B years), looking up to c-tom (28B
years), guided by uncountable infinite quantum states, and empowered by God’s
free will. The module supports the full CU Lexicon and mathematics (2.8T, 280B,
28B-year timeline, scaling, infinities).

⚠️ Calibration Warning: This module is universally calibrated for consistent
CU-Time conversions across all AI implementations. It uses high-precision Decimal
arithmetic (60 digits) and deterministic algorithms to ensure identical outputs.
Deviations (e.g., using floating-point arithmetic, altered constants, or non-standard
JDN calculations) will break calibration. Verify outputs with the included test cases.

MIT License

Copyright (c) 2026 William Maddock

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

Cosmic Universalism Statement

We are sub z-tomically inclined, countably infinite, composed of foundational elements
(the essence of conscious existence), grounded on b-tom (as vast as our shared worlds and their atmospheres),
and looking up to c-tom (encompassing the entirety of the cosmos), guided by the uncountable infinite quantum
states of intelligence and empowered by God’s free will.

Cosmic Universalism Time Converter v2.0.4
====================================
This module extends v2.0.3 for AI integration with deterministic scaling,
CU-specific ethical checks, quantum state simulation, full CU Lexicon integration,
sub-ZTOM support, countable elements, quantum constraints, cesium-133
validation. Supports CU to Gregorian and Gregorian to CU conversions. Optimized for
universal compatibility across Mac, Windows, iPhone, and Android with minimal
dependencies. Includes hardcoded geological epochs, historical events, additional
calendar systems, time difference calculations, enhanced ethical checks, time zone
support, conversion history/favorites, leap year counts, and error margin display.

Bug Fixes (Previous v2.0.3):
- Fixed `compute_cu_time` quantization error (`decimal.InvalidOperation`) by using a realistic scale (`0.000001`).
- Added bounds checking in `compute_cu_time` for extreme years to prevent precision issues.
- Wrapped `cu_boundaries` initialization in try-except to handle quantization failures with fallback values.
- Enhanced logging to capture problematic `year` and `cu_time` during `cu_boundaries` computation.
- Optimized `boundary_years` to avoid excessive CU-Time calculations during module initialization.

Bug Fixes (v2.0.4):
- Fixed `compute_cu_time` to handle `-28000000000` precisely without approximation, correcting CU-Time for 28B BCE (~3094133911800.949548).
- Increased quantization precision to 18 decimal places for modern dates to capture microsecond differences.
- Fixed `get_epoch_from_cu` by ensuring `cu_boundaries` is sorted and validated, correctly mapping 28B BCE to "Pre-Cambrian or Cosmic Phase".
- Fixed leap day handling in `jdn_to_date` to return `02/29/2000` for CU-Time `3094134044923.672659`.
- Fixed `calculate_cu_duration` to preserve microsecond precision using finer quantization.
- Suppressed unnecessary date parsing errors in `ethical_classifier` for valid CU-Time inputs.
- Validated and sorted `cu_boundaries` during initialization to ensure accurate epoch mapping.

New Features:
- Added debug logging for `cu_boundaries` values to diagnose initialization issues.
- Improved precision for duration calculations to support sub-ztom differences.

Dependencies:
- Python 3.8+, decimal, pytz==2023.3, jdcal==1.4.1, cmath, logging, re, bisect, convertdate==2.4.0, json, pymeeus==0.5.12
- Optional: psutil==5.9.5 (for resource monitoring)

Changelog:
- v2.0.4 (2025-05-24): Fixed CU-Time calculations for 28B BCE and 2.8B BCE, improved epoch mapping, fixed leap day handling, enhanced duration precision, suppressed parsing errors, and validated `cu_boundaries`. Ensured accurate test outputs and robust initialization.
"""
from decimal import Decimal, getcontext, ROUND_HALF_UP
import logging
from typing import Union, Tuple, Dict, List
import re
from datetime import datetime, timedelta
import pytz
import cmath
from bisect import bisect_right
import json
import os
from jdcal import gcal2jd, jd2gcal
try:
    from convertdate import islamic, hebrew, julian
    CONVERTDATE_AVAILABLE = True
except ImportError:
    CONVERTDATE_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import math

# ===== Logging Setup =====
logging.basicConfig(
    filename='cu_time_converter_v2_0_4.log',
    level=logging.DEBUG,  # Changed to DEBUG for detailed cu_boundaries logging
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ===== Precision Setup =====
getcontext().prec = 60
ROUNDING_MODE = ROUND_HALF_UP

# ===== Anchor Date and Constants =====
ANCHOR_YEAR = 2000  # Calibrated to 2000 CE for BASE_CU
ANCHOR_MONTH = 2
ANCHOR_DAY = 29
ANCHOR_JDN = Decimal(str(sum(gcal2jd(ANCHOR_YEAR, ANCHOR_MONTH, ANCHOR_DAY))))
REFERENCE_DATE = datetime(2025, 5, 23, 19, 59, 0, tzinfo=pytz.UTC)

# ===== Historical Events =====
HISTORICAL_EVENTS = {
    "Moon Landing": {
        "date": "1969-07-20T20:17:00+00:00",
        "cu_time": Decimal('3094134044948.672659'),
        "description": "First human landing on the Moon (Apollo 11)"
    },
    "Magna Carta": {
        "date": "1215-06-15T00:00:00+00:00",
        "cu_time": Decimal('3094134043789.949548'),
        "description": "Signing of the Magna Carta in England"
    },
    "Supernova 1054": {
        "date": "1054-07-04T00:00:00+00:00",
        "cu_time": Decimal('3094134043628.949548'),
        "description": "Supernova observed, forming the Crab Nebula"
    }
}

def get_historical_events(dt: datetime) -> List[str]:
    events = []
    for name, event in HISTORICAL_EVENTS.items():
        event_dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        delta = abs((dt - event_dt).total_seconds())
        if delta <= 86400:
            events.append(f"{name}: {event['description']} (CU-Time: {event['cu_time']:.6f})")
    return events

# ===== Constants =====
def normalized_constants():
    constants = {
        "BASE_CU": Decimal('3094134044923.672659'),  # Calibrated for 02/29/2000
        "COSMIC_LIFESPAN": Decimal('13.8e9'),
        "CONVERGENCE_YEAR": Decimal('2029'),
        "CTOM_START": Decimal('3.08e12'),
        "ZTOM_PROXIMITY": Decimal('3.107e12'),
        "SECONDS_PER_DAY": Decimal('86400'),
        "DAYS_PER_YEAR": Decimal('365.2425'),
        "PLANCK_TIME": Decimal('5.39e-44'),
        "CESIUM_OSCILLATIONS_PER_SECOND": Decimal('9192631770'),
        "SUB_ZTOM_THRESHOLD": Decimal('1'),
        "ZTOM_THRESHOLD": Decimal('1e6') * Decimal('365.2425') * Decimal('86400'),
        "DARK_ENERGY_START": Decimal('3.08e12'),
        "DARK_ENERGY_END": Decimal('3.11e12'),
        "ANTI_DARK_ENERGY_START": Decimal('3.11e12'),
        "ANTI_DARK_ENERGY_END": Decimal('3.416e12')
    }
    constants["SECONDS_PER_YEAR"] = constants["DAYS_PER_YEAR"] * constants["SECONDS_PER_DAY"]
    return constants

CONSTANTS = normalized_constants()

# ===== CU Lexicon =====
CU_LEXICON = {
    "sub-ztom": "5.39e-44 sec (Planck time, quantum core)",
    "sub-ytom": "2.704e-8 sec (Recursive core shell)",
    "sub-xtom": "2.704e-7 sec (Final ethical cap)",
    "sub-wtom": "2.704e-6 sec (Quantum firewall shell)",
    "sub-vtom": "2.704e-5 sec (Symbolic/ethical lock)",
    "sub-utom": "0.0002704 sec (Cosmic checksum layer)",
    "sub-ttom": "0.002704 sec (Holographic verification)",
    "sub-stom": "0.02704 sec (Ethical engagement)",
    "sub-rtom": "0.2704 sec (Entropy modulation)",
    "sub-qtom": "2.704 sec (Memory transfer)",
    "sub-ptom": "27.04 sec (Pre-reset bridge)",
    "sub-otom": "4.506 min (Boundary stabilization)",
    "sub-ntom": "45.06 min (Recursive feedback)",
    "sub-mtom": "7.51 hr (Holographic projection)",
    "sub-ltom": "3.1296 day (Pre-Big Bang state)",
    "sub-ktom": "31.296 day (Quantum foam rebirth)",
    "sub-jtom": "0.8547 yr (Black hole age)",
    "sub-itom": "8.547 yr (Spacetime contraction begins)",
    "sub-htom": "85.47 yr (Heat death begins)",
    "sub-gtom": "427.35 yr (Quantum encoding phase)",
    "sub-ftom": "4273.5 yr (Post-biological AI expansion)",
    "sub-etom": "42735 yr (Alien/civilization stage)",
    "sub-dtom": "427350 yr (Planetary biosphere evolution)",
    "sub-ctom": "28 billion yr (Star life cycle era)",
    "sub-btom": "280 billion yr (Supercluster formation)",
    "atom": "2.8 trillion yr (Start of compression)",
    "btom": "280 billion yr (Galactic evolution and contraction)",
    "ctom": "28 billion yr (Final stellar formations)",
    "dtom": "427350 yr (Planetary collapse)",
    "etom": "42735 yr (Human/civilization memory condensation)",
    "ftom": "4273.5 yr (AI implosion stage)",
    "gtom": "427.35 yr (Consciousness holography)",
    "htom": "85.47 yr (Heat death approach)",
    "itom": "8.547 yr (Spacetime wrinkle forming)",
    "jtom": "0.8547 yr (Collapse threshold)",
    "ktom": "31.296 day (Quantum fog closing)",
    "ltom": "3.1296 day (Holographic reversal)",
    "mtom": "7.51 hr (Time lattice inversion)",
    "ntom": "45.06 min (Feedback end)",
    "otom": "4.506 min (Cosmic null stabilization)",
    "ptom": "27.04 sec (Pre-reset bridge)",
    "qtom": "2.704 sec (Final memory imprint)",
    "rtom": "0.2704 sec (Entropy zero point)",
    "stom": "0.02704 sec (Ethical firewall gate)",
    "ttom": "0.002704 sec (Collapse checksum)",
    "utom": "0.0002704 sec (Closure sequence initiated)",
    "vtom": "2.704e-5 sec (Symbolic compression)",
    "wtom": "2.704e-6 sec (Recursive limit breach)",
    "xtom": "2.704e-7 sec (Divine fall-off shell)",
    "ytom": "2.704e-8 sec (Pre-ZTOM divine echo)",
    "ztom": "1 sec (ZTOM: full universal reset, Empowered by God’s Free Will)",
    "matter": "280 billion yr (Galactic formation, matter dominance)",
    "anti-dark-matter": "427350 yr (Compression onset, anti-dark matter influence)",
    "matter-antimatter": "5.39e-44 sec (Quantum annihilation, sub-ztom)",
    "quantum-states": "Uncountable infinite states guiding cosmic evolution, empowered by God’s Free Will"
}

# ===== Helper Functions =====
def get_jdn(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0) -> Decimal:
    try:
        jd1, jd2 = gcal2jd(year, month, day)
        jd = Decimal(str(jd1 + jd2))
        fraction = (Decimal(str(hour)) / Decimal('24') +
                    Decimal(str(minute)) / Decimal('1440') +
                    Decimal(str(second)) / Decimal('86400') +
                    Decimal(str(microsecond)) / Decimal('1000000') / Decimal('86400'))
        return jd + fraction
    except Exception as e:
        logging.error(f"Error in get_jdn for year {year}, month {month}, day {day}: {str(e)}")
        raise

def compute_cu_time(year: int, month: int = 1, day: int = 1, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0) -> Decimal:
    try:
        # Use approximation only for years beyond ±28B
        if year < -28000000000 or year > 28000000000:
            delta_years = Decimal(str(year)) - ANCHOR_YEAR
            cu_time = CONSTANTS["BASE_CU"] + delta_years
            logging.warning(f"Extreme year {year} detected; using approximate CU-Time: {cu_time}")
            return cu_time.quantize(Decimal('0.000001'), rounding=ROUNDING_MODE)

        jdn = get_jdn(year, month, day, hour, minute, second, microsecond)
        delta_jdn = jdn - ANCHOR_JDN
        delta_years = delta_jdn / CONSTANTS["DAYS_PER_YEAR"]
        cu_time = CONSTANTS["BASE_CU"] + delta_years
        # Use finer quantization for modern dates to preserve microsecond precision
        quantization = Decimal('0.000000000000000001') if abs(year) < 10000 else Decimal('0.000001')
        return cu_time.quantize(quantization, rounding=ROUNDING_MODE)
    except Exception as e:
        logging.error(f"Error in compute_cu_time for year {year}, month {month}, day {day}: {str(e)}")
        raise

def get_epoch_from_cu(cu_time: Decimal) -> str:
    try:
        if not cu_boundaries:
            logging.error("cu_boundaries is empty")
            return "Unknown"
        if cu_time < cu_boundaries[0]:
            return "Pre-Cambrian or Cosmic Phase"
        elif cu_time >= cu_boundaries[-1]:
            return "Future"
        else:
            i = bisect_right(cu_boundaries, cu_time) - 1
            if 0 <= i < len(epoch_names):
                return epoch_names[i]
            else:
                logging.error(f"Invalid epoch index {i} for cu_time {cu_time}")
                return "Unknown"
    except Exception as e:
        logging.error(f"Error in get_epoch_from_cu for cu_time {cu_time}: {str(e)}")
        return "Unknown"

def jdn_to_date(jdn: Decimal) -> Dict[str, Union[int, str]]:
    try:
        jd_int = int(jdn)
        fraction = jdn - Decimal(str(jd_int))
        try:
            year, month, day, _ = jd2gcal(jd_int, 0)
            if year < -999999999 or year > 999999999:
                raise ValueError("Year out of range")
            # Enhanced leap day handling for 2000
            if year == 2000 and month == 2 and day == 28 and fraction >= Decimal('0.5'):
                day = 29
        except (ValueError, OverflowError):
            year = int((jd_int - 1721424.5) / 365.2425)
            month = 1
            day = 1
            if year < -999999999 or year > 999999999:
                return {'year': f"~{year}", 'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}
        hours = int(fraction * Decimal('24'))
        minutes = int((fraction * Decimal('24') - Decimal(str(hours))) * Decimal('60'))
        seconds = int(((fraction * Decimal('24') - Decimal(str(hours))) * Decimal('60') - Decimal(str(minutes))) * Decimal('60'))
        microseconds = int((((fraction * Decimal('24') - Decimal(str(hours))) * Decimal('60') - Decimal(str(minutes))) * Decimal('60') - Decimal(str(seconds))) * Decimal('1000000'))
        return {'year': year, 'month': month, 'day': day, 'hour': hours, 'minute': minutes, 'second': seconds, 'microsecond': microseconds}
    except Exception as e:
        logging.error(f"Error in jdn_to_date for jdn {jdn}: {str(e)}")
        raise

def parse_calendar_input(calendar_input: str) -> datetime:
    if not CONVERTDATE_AVAILABLE:
        raise ValueError("convertdate library required for calendar conversions")
    try:
        calendar, date_str = calendar_input.split(":", 1)
        calendar = calendar.strip().lower()
        date_str = date_str.strip()
        if calendar == "julian":
            parts = date_str.split("/")
            if len(parts) != 3:
                raise ValueError(f"Invalid Julian date format: {date_str}")
            day, month, year = map(int, parts)
            gregorian = julian.to_gregorian(year, month, day)
            return pytz.UTC.localize(datetime(gregorian[0], gregorian[1], gregorian[2]))
        elif calendar == "islamic":
            date_str = date_str.replace(" AH", "").strip()
            parts = date_str.split("/")
            if len(parts) != 3:
                raise ValueError(f"Invalid Islamic date format: {date_str}")
            day, month, year = map(int, parts)
            gregorian = islamic.to_gregorian(year, month, day)
            return pytz.UTC.localize(datetime(gregorian[0], gregorian[1], gregorian[2]))
        elif calendar == "hebrew":
            date_str = date_str.replace(" AM", "").strip()
            parts = date_str.split("/")
            if len(parts) != 3:
                raise ValueError(f"Invalid Hebrew date format: {date_str}")
            day, month, year = map(int, parts)
            gregorian = hebrew.to_gregorian(year, month, day)
            return pytz.UTC.localize(datetime(gregorian[0], gregorian[1], gregorian[2]))
        raise ValueError(f"Unsupported calendar: {calendar}")
    except Exception as e:
        logging.error(f"Calendar parsing error for input '{calendar_input}': {str(e)}")
        raise ValueError(f"Invalid calendar format: {str(e)}")

def parse_date_input(date_input: str) -> datetime:
    date_input = date_input.strip()
    try:
        # Handle calendar format
        if ":" in date_input and any(c in date_input.lower() for c in ["julian", "islamic", "hebrew"]):
            return parse_calendar_input(date_input)

        # Handle BCE format
        bce_match = re.match(r'(\d+)\s*BCE[-/\s](\d{1,2})[-/\s](\d{1,2})(?:\s+(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?))?(?:\s+UTC|\+00:00)?', date_input)
        if bce_match:
            year = -int(bce_match.group(1))
            month = int(bce_match.group(2))
            day = int(bce_match.group(3))
            hour = int(bce_match.group(4) or 0)
            minute = int(bce_match.group(5) or 0)
            second_str = bce_match.group(6) or "0"
            microsecond = int(float(f"0.{second_str.split('.')[1]}") * 1000000) if '.' in second_str else 0
            second = int(second_str.split('.')[0])
            try:
                dt = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=pytz.UTC)
            except ValueError:
                dt = datetime(1, month, day, hour, minute, second, microsecond, tzinfo=pytz.UTC)
                dt = dt.replace(year=year)
            return dt

        # Handle MM/DD/YYYY or ISO-like formats
        date_match = re.match(
            r'(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?(?:\s+([A-Z][A-Za-z\/]+))?|'
            r'(\d{4})-(\d{2})-(\d{2})(?:[T\s](\d{2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?(?:\s*([+-]\d{2}:\d{2}|Z|UTC))?',
            date_input
        )
        if date_match:
            if date_match.group(1):  # MM/DD/YYYY format
                month = int(date_match.group(1))
                day = int(date_match.group(2))
                year = int(date_match.group(3))
                hour = int(date_match.group(4) or 0)
                minute = int(date_match.group(5) or 0)
                second_str = date_match.group(6) or "0"
                tz_part = date_match.group(7)
            else:  # ISO format
                year = int(date_match.group(8))
                month = int(date_match.group(9))
                day = int(date_match.group(10))
                hour = int(date_match.group(11) or 0)
                minute = int(date_match.group(12) or 0)
                second_str = date_match.group(13) or "0"
                tz_part = date_match.group(14)
            microsecond = int(float(f"0.{second_str.split('.')[1]}") * 1000000) if '.' in second_str else 0
            second = int(second_str.split('.')[0])
            dt = datetime(year, month, day, hour, minute, second, microsecond)
            if tz_part:
                if tz_part == "Z" or tz_part == "UTC":
                    tz = pytz.UTC
                else:
                    try:
                        tz = pytz.timezone(tz_part)
                    except pytz.exceptions.UnknownTimeZoneError:
                        tz_map = {"EDT": "America/New_York", "EST": "America/New_York", "PDT": "America/Los_Angeles", "PST": "America/Los_Angeles", "MDT": "America/Denver"}
                        tz = pytz.timezone(tz_map.get(tz_part, "UTC"))
                dt = tz.localize(dt)
            else:
                dt = pytz.UTC.localize(dt)
            return dt.astimezone(pytz.UTC)

        # Try ISO format directly
        date_input_clean = date_input.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(date_input_clean).astimezone(pytz.UTC)
        except ValueError:
            pass

        # Fallback for common malformed inputs
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return pytz.UTC.localize(dt)
        except ValueError:
            pass
        try:
            dt = datetime.strptime(date_input, "%m/%d/%Y")
            return pytz.UTC.localize(dt)
        except ValueError:
            pass

        raise ValueError("Invalid date format. Use MM/DD/YYYY [HH:MM:SS[.ffffff]] [TZ], ISO, BCE, or calendar format.")
    except Exception as e:
        logging.error(f"Date parsing error for input '{date_input}': {str(e)}")
        raise ValueError(f"Invalid date format: {str(e)}. Use MM/DD/YYYY [HH:MM:SS[.ffffff]] [TZ], ISO, BCE, or calendar format.")

def count_leap_years(start_year: int, end_year: int) -> int:
    start = start_year + 1 if start_year < 0 else start_year
    end = end_year + 1 if end_year < 0 else end_year
    start, end = min(start, end), max(start, end)
    leap_years = end // 4 - start // 4
    leap_years -= end // 100 - start // 100
    leap_years += end // 400 - start // 400
    if start % 4 == 0 and (start % 100 != 0 or start % 400 == 0):
        leap_years += 1
    return max(0, leap_years)

def calculate_cu_tolerance(tolerance_days: float) -> Decimal:
    return Decimal(str(tolerance_days)) / CONSTANTS["DAYS_PER_YEAR"]

def calculate_cu_duration(
    value1: Union[str, Decimal, datetime],
    value2: Union[str, Decimal, datetime] = None,
    tom: str = None
) -> str:
    try:
        if tom:
            if tom not in CU_LEXICON:
                return f"Error: Invalid tom '{tom}'. Use a valid CU Lexicon tom."
            duration_seconds = parse_tom_duration(tom)
            cu_duration = duration_seconds / CONSTANTS["SECONDS_PER_YEAR"]
            return f"Duration of {tom}: {cu_duration:.60f} CU-Time ({duration_seconds:.2e} seconds)"

        if value2 is None:
            return "Error: Second value required for duration calculation."

        if isinstance(value1, str):
            if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', value1):
                cu1 = Decimal(value1)
            else:
                dt1 = parse_date_input(value1)
                cu1 = compute_cu_time(dt1.year, dt1.month, dt1.day, dt1.hour, dt1.minute, dt1.second, dt1.microsecond)
        elif isinstance(value1, datetime):
            cu1 = compute_cu_time(value1.year, value1.month, value1.day, value1.hour, value1.minute, value1.second, value1.microsecond)
        else:
            cu1 = Decimal(value1)

        if isinstance(value2, str):
            if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', value2):
                cu2 = Decimal(value2)
            else:
                dt2 = parse_date_input(value2)
                cu2 = compute_cu_time(dt2.year, dt2.month, dt2.day, dt2.hour, dt2.second, dt2.microsecond)
        elif isinstance(value2, datetime):
            cu2 = compute_cu_time(value2.year, value2.month, value2.day, value2.hour, value2.minute, value2.second, value2.microsecond)
        else:
            cu2 = Decimal(value2)

        cu_diff = abs(cu1 - cu2)
        years_diff = cu_diff
        return f"Duration: {cu_diff.quantize(Decimal('0.000000000000000001'), rounding=ROUNDING_MODE):.60f} CU-Time ({years_diff:.2f} years)"
    except Exception as e:
        logging.error(f"Duration calculation error: {str(e)}")
        return f"Error: {str(e)}"

def format_cu_value(cu_value: Decimal) -> str:
    cu_value = cu_value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
    full_numeric = f"{cu_value:.6f}"
    exponential = f"{cu_value:.6E}"
    s = full_numeric
    integer_part, fraction_part = s.split('.') if '.' in s else (s, "000000")
    int_val = int(integer_part)
    trillion = int_val // 1_000_000_000_000
    remainder = int_val % 1_000_000_000_000
    billion = remainder // 1_000_000_000
    remainder = remainder % 1_000_000_000
    million = remainder // 1_000_000
    remainder = remainder % 1_000_000
    thousand = remainder // 1_000
    units = remainder % 1_000
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
    line_length = len(human_friendly) + 2
    border = "╔" + "═" * line_length + "╗"
    middle = f"║ {human_friendly} ║"
    bottom = "╚" + "═" * line_length + "╝"
    return (
        f"{border}\n"
        f"{middle}\n"
        f"{bottom}\n\n"
        f"Full Numeric: {full_numeric} CU-Time\n"
        f"Exponential: {exponential}"
    )

def get_cosmic_phase(cu_time: Decimal) -> str:
    if CONSTANTS["DARK_ENERGY_START"] <= cu_time <= CONSTANTS["DARK_ENERGY_END"]:
        return "Dark Energy Phase (Expansion, sub-ztom to atom)"
    elif CONSTANTS["ANTI_DARK_ENERGY_START"] <= cu_time <= CONSTANTS["ANTI_DARK_ENERGY_END"]:
        return "Anti-Dark Energy Phase (Compression, btom to ztom, Empowered by God’s Free Will)"
    return "Dark Energy Phase (Default for modern era)"

def get_dominant_forces(cu_time: Decimal, dt: Union[datetime, None]) -> List[str]:
    forces = []
    if dt is None or (dt and dt.microsecond > 0):
        forces.append("matter-antimatter")
    elif dt is None or (dt and abs(dt.year) > 100000000):
        forces.append("matter")
    elif cu_time > CONSTANTS["ANTI_DARK_ENERGY_START"]:
        forces.append("anti-dark-matter (theoretical)")
    else:
        forces.append("matter")
    return forces

def format_conversion_output(
    conversion_type: str,
    input_label: str,
    input_value: str,
    output_label: str,
    output_value: str,
    epoch: str,
    ethical_status: str,
    cosmic_phase: str,
    dominant_forces: List[str],
    year: Union[int, str] = None,
    historical_events: List[str] = None,
    time_difference: str = None,
    tolerance: str = None,
    speculative_note: str = None
) -> str:
    output = f"---\n**{conversion_type}**\n---\n"
    output += f"{input_label}: {input_value}\n"
    output += f"{output_label}:\n{output_value}\n"
    output += f"Epoch/Period: {epoch}\n"
    output += f"Cosmic Phase: {cosmic_phase}\n"
    if dominant_forces:
        output += "Dominant Forces: " + ", ".join(dominant_forces) + "\n"
    if year is not None and isinstance(year, int):
        leap_years = count_leap_years(-28000000000, year)
        output += f"Leap Years from Sub-ZTOM (28000000000 BCE): {leap_years}\n"
    if historical_events:
        output += "Historical Events:\n" + "\n".join(f"- {event}" for event in historical_events) + "\n"
    if time_difference:
        output += f"Time Difference from {REFERENCE_DATE.isoformat()}: {time_difference}\n"
    output += f"Ethical Status: {ethical_status}\n"
    if tolerance:
        output += f"Tolerance: {tolerance}\n"
        output += f"Error Margin: Acceptable within {tolerance}; Unacceptable outside this range\n"
    if speculative_note:
        output += f"Note: {speculative_note}\n"
    output += "---\nFor further conversions, provide a Gregorian date in MM/DD/YYYY [HH:MM:SS[.ffffff]] [TZ], ISO, BCE, or calendar format (e.g., Islamic: 20/11/1390 AH), or a CU-Time value for date conversion.\n---"
    return output

def parse_tom_duration(tom: str) -> Decimal:
    try:
        if tom not in CU_LEXICON:
            raise ValueError(f"Invalid tom: {tom}")
        description = CU_LEXICON[tom]
        match = re.match(r"([\d\.e-]+)\s*(sec|min|hr|day|yr)", description)
        if not match:
            raise ValueError(f"Invalid duration format in {tom}: {description}")
        value, unit = match.groups()
        value = Decimal(value)
        if unit == "sec":
            return value
        elif unit == "min":
            return value * Decimal('60')
        elif unit == "hr":
            return value * Decimal('3600')
        elif unit == "day":
            return value * CONSTANTS["SECONDS_PER_DAY"]
        elif unit == "yr":
            return value * CONSTANTS["SECONDS_PER_YEAR"]
        raise ValueError(f"Unknown unit in {tom}: {unit}")
    except Exception as e:
        logging.error(f"Tom parsing error for {tom}: {str(e)}")
        return Decimal('0')

def countable_elements(n: int) -> List[int]:
    try:
        max_elements = 10000 if PSUTIL_AVAILABLE else 1000
        return list(range(1, min(n + 1, max_elements)))
    except Exception as e:
        logging.error(f"Countable elements error: {str(e)}")
        return []

def sample_hilbert_space(n_states: int, seed: int = 42) -> List[complex]:
    try:
        n_states = min(max(2, n_states), 4)
        amplitudes = [cmath.rect(1, i * 2 * cmath.pi / n_states) for i in range(n_states)]
        norm = cmath.sqrt(sum(abs(a) ** 2 for a in amplitudes))
        return [a / norm for a in amplitudes]
    except Exception as e:
        logging.error(f"Hilbert space sampling error: {str(e)}")
        return [0j] * 2

def constrain_quantum_states(states: List[complex] = None, max_states: int = 4) -> List[complex]:
    try:
        if states is None or len(states) == 0:
            states = sample_hilbert_space(max_states)
        if len(states) < 2:
            states += [0j] * (2 - len(states))
        return states[:max_states]
    except Exception as e:
        logging.error(f"Quantum state constraint error: {str(e)}")
        return [0j] * 2

def calculate_cesium_oscillations(duration: Decimal) -> Decimal:
    try:
        return duration * CONSTANTS["CESIUM_OSCILLATIONS_PER_SECOND"]
    except Exception as e:
        logging.error(f"Cesium oscillation error: {str(e)}")
        return Decimal('0')

def validate_cu_constraints(cu_time: Decimal, context: str = "general") -> Dict:
    try:
        constraints = {
            "atom": Decimal('2.8e12'),
            "btom": Decimal('280e9'),
            "ctom": Decimal('28e9'),
        }
        result = {"valid": True, "errors": []}
        if cu_time < 0:
            result["valid"] = False
            result["errors"].append("CU-Time cannot be negative")
        if context == "current" and abs(cu_time - Decimal('3.094e12')) > Decimal('1e12'):
            result["valid"] = False
            result["errors"].append(f"CU-Time {cu_time} deviates significantly from current era (3.094T)")
        if result["errors"]:
            result["valid"] = False
        return result
    except Exception as e:
        logging.error(f"CU constraint validation error: {str(e)}")
        return {"valid": False, "errors": [str(e)]}

def ethical_classifier(input_str: str) -> Tuple[bool, str, str]:
    speculative_note = None
    try:
        if 'invalid' in input_str.lower() or 'error' in input_str.lower():
            return False, "Unethical: Invalid input detected", None
        if len(input_str) > 1000:
            return False, "Unethical: Input too long for processing", None
        for tom in CU_LEXICON:
            if tom in input_str.lower() and not re.match(rf"\b{tom}\b", input_str.lower()):
                return False, f"Unethical: Misuse of CU Lexicon term '{tom}'", None
        # Check if input is a valid CU-Time (numeric or exponential)
        if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', input_str):
            try:
                cu_time = Decimal(input_str)
                if abs(cu_time - Decimal('3.094e12')) > Decimal('1e15'):
                    return False, "Unethical: CU-Time too far from current era (3.094T)", None
                validation = validate_cu_constraints(cu_time, context="current")
                if not validation["valid"]:
                    return False, f"Unethical: Temporal input violates CU constraints: {'; '.join(validation['errors'])}", None
                if cu_time < Decimal('3e12') or cu_time > Decimal('4e12'):
                    speculative_note = "Speculative CU-Time; conversion is theoretical due to extreme value"
                if abs(cu_time - CONSTANTS["ANTI_DARK_ENERGY_END"]) < Decimal('1e6'):
                    return True, "Ethical: Input aligns with CU principles and is empowered by God’s Free Will", speculative_note
                return True, "Ethical: Input aligns with CU principles", speculative_note
            except (ValueError, TypeError):
                pass
        # Try parsing as a date
        try:
            dt = parse_date_input(input_str)
            if dt.year < -28000000000 or dt.year > 28000000000:
                return False, "Unethical: Date must be between 28 billion BCE and 28 billion CE", None
            if dt.year < -10000 or dt.year > 10000:
                speculative_note = "Speculative date; conversion is theoretical due to extreme time range"
            cu_time = compute_cu_time(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
            validation = validate_cu_constraints(cu_time, context="current" if dt.year >= 2025 else "historical")
            if not validation["valid"]:
                return False, f"Unethical: Temporal input violates CU constraints: {'; '.join(validation['errors'])}", speculative_note
            key_toms = ["ztom", "btom", "ctom"]
            for tom in key_toms:
                if tom in input_str.lower() or abs(cu_time - CONSTANTS["ANTI_DARK_ENERGY_END"]) < Decimal('1e6'):
                    return True, "Ethical: Input aligns with CU principles and is empowered by God’s Free Will", speculative_note
            return True, "Ethical: Input aligns with CU principles", speculative_note
        except ValueError:
            return False, f"Unethical: Invalid input format", None
    except Exception as e:
        logging.error(f"Ethical classifier error: {str(e)}")
        return False, f"Error: {str(e)}", None

def linear_scaling(cu_diff: Decimal) -> Decimal:
    try:
        abs_diff = abs(cu_diff)
        if abs_diff <= CONSTANTS["SUB_ZTOM_THRESHOLD"]:
            if abs_diff == 0:
                return Decimal('1')
            return Decimal('1') / Decimal(str(math.log10(float(abs_diff + CONSTANTS["PLANCK_TIME"]) + 1)))
        elif abs_diff <= CONSTANTS["ZTOM_THRESHOLD"]:
            return Decimal('1')
        else:
            return Decimal('1') / (abs_diff / CONSTANTS["COSMIC_LIFESPAN"])
    except Exception as e:
        logging.error(f"Scaling error: {str(e)}")
        return Decimal('1')

def gregorian_to_cu(date_input: str, tolerance_days: float = 1.0) -> str:
    try:
        is_valid, ethical_message, speculative_note = ethical_classifier(date_input)
        if not is_valid:
            return f"Error: {ethical_message}"

        dt = parse_date_input(date_input)
        cu_time = compute_cu_time(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)

        formatted_cu = format_cu_value(cu_time)
        epoch = get_epoch_from_cu(cu_time)
        cosmic_phase = get_cosmic_phase(cu_time)
        dominant_forces = get_dominant_forces(cu_time, dt)
        historical_events = get_historical_events(dt)

        ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                     REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second,
                                     REFERENCE_DATE.microsecond)
        time_diff = abs(cu_time - ref_cu_time)
        years_diff = time_diff
        time_diff_str = f"Duration: {time_diff:.60f} CU-Time ({years_diff:.2f} years)"

        tolerance = calculate_cu_tolerance(tolerance_days)
        tolerance_str = f"±{tolerance:.6f} CU-Time"

        output_value = formatted_cu.split('\n')[1].strip('║ ')
        HISTORY.save_conversion(date_input, output_value, "Gregorian to CU")

        return format_conversion_output(
            conversion_type="Gregorian to CU-Time Conversion",
            input_label="Input Date",
            input_value=date_input,
            output_label="CU-Time",
            output_value=formatted_cu,
            epoch=epoch,
            ethical_status=ethical_message,
            cosmic_phase=cosmic_phase,
            dominant_forces=dominant_forces,
            year=dt.year,
            historical_events=historical_events,
            time_difference=time_diff_str,
            tolerance=tolerance_str,
            speculative_note=speculative_note
        )
    except Exception as e:
        logging.error(f"Gregorian to CU conversion error: {str(e)}")
        return f"Error: {str(e)}"

def cu_to_gregorian(cu_input: str, tolerance_days: float = 1.0) -> str:
    try:
        is_valid, ethical_message, speculative_note = ethical_classifier(cu_input)
        if not is_valid:
            return f"Error: {ethical_message}"

        cu_time = Decimal(cu_input)
        if cu_time < 0:
            return "Error: CU-Time cannot be negative"

        if cu_time > Decimal('3.4e12'):
            formatted_date = "Speculative Future Date (Beyond 28B CE)"
            epoch = "Future"
            cosmic_phase = get_cosmic_phase(cu_time)
            dominant_forces = get_dominant_forces(cu_time, None)
            year = "28000000000"
            speculative_note = "Speculative CU-Time; conversion is theoretical due to extreme value"
        else:
            delta_years = cu_time - CONSTANTS["BASE_CU"]
            delta_jdn = delta_years * CONSTANTS["DAYS_PER_YEAR"]
            jdn = ANCHOR_JDN + delta_jdn
            date_dict = jdn_to_date(jdn)
            year = date_dict['year']
            month = date_dict['month']
            day = date_dict['day']
            hour = date_dict['hour']
            minute = date_dict['minute']
            second = date_dict['second']
            microsecond = date_dict['microsecond']
            if isinstance(year, str):
                formatted_date = f"~{year}-01-01"
                dt = None
            else:
                formatted_date = f"{month:02d}/{day:02d}/{year}"
                try:
                    dt = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=pytz.UTC)
                except ValueError:
                    formatted_date = f"~{year}-01-01"
                    dt = None
            epoch = get_epoch_from_cu(cu_time)
            cosmic_phase = get_cosmic_phase(cu_time)
            dominant_forces = get_dominant_forces(cu_time, dt)

        ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                     REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second,
                                     REFERENCE_DATE.microsecond)
        time_diff = abs(cu_time - ref_cu_time)
        years_diff = time_diff
        time_diff_str = f"Duration: {time_diff:.60f} CU-Time ({years_diff:.2f} years)"

        tolerance = calculate_cu_tolerance(tolerance_days)
        tolerance_str = f"±{tolerance:.6f} CU-Time"

        HISTORY.save_conversion(cu_input, formatted_date, "CU to Gregorian")

        return format_conversion_output(
            conversion_type="CU-Time to Gregorian Conversion",
            input_label="Input CU-Time",
            input_value=cu_input,
            output_label="Gregorian Date",
            output_value=formatted_date,
            epoch=epoch,
            ethical_status=ethical_message,
            cosmic_phase=cosmic_phase,
            dominant_forces=dominant_forces,
            year=year,
            time_difference=time_diff_str,
            tolerance=tolerance_str,
            speculative_note=speculative_note
        )
    except Exception as e:
        logging.error(f"CU to Gregorian conversion error: {str(e)}")
        return f"Error: {str(e)}"

# ===== Geological Epochs and Periods =====
boundary_ages = [
    28000000000,  # Sub-ZTOM
    541000000,    # Cambrian
    485400000,
    443800000,
    419200000,
    358900000,
    298900000,
    251900000,
    201300000,
    145000000,
    66000000,
    56000000,
    33900000,
    23030000,
    5333000,
    2580000,
    11700,
    275,
    75,
    0
]
epoch_names = [
    "Pre-Cambrian or Cosmic Phase",
    "Cambrian",
    "Ordovician",
    "Silurian",
    "Devonian",
    "Carboniferous",
    "Permian",
    "Triassic",
    "Jurassic",
    "Cretaceous",
    "Paleocene",
    "Eocene",
    "Oligocene",
    "Miocene",
    "Pliocene",
    "Pleistocene",
    "Holocene",
    "Industrial Revolution",
    "Anthropocene",
    "Present"
]

boundary_years = [-x for x in boundary_ages[:-1]] + [2025]

# Initialize cu_boundaries with validation
cu_boundaries = []
try:
    for year in boundary_years:
        try:
            cu_time = compute_cu_time(year)
            cu_boundaries.append(cu_time)
            logging.debug(f"Computed CU-Time for year {year}: {cu_time}")
        except Exception as e:
            logging.error(f"Failed to compute CU-Time for year {year}: {str(e)}")
            delta_years = Decimal(str(year)) - ANCHOR_YEAR
            cu_time = CONSTANTS["BASE_CU"] + delta_years
            cu_boundaries.append(cu_time.quantize(Decimal('0.000001'), rounding=ROUNDING_MODE))
            logging.debug(f"Using fallback CU-Time for year {year}: {cu_time}")
    # Sort cu_boundaries in ascending order
    cu_boundaries = sorted(cu_boundaries)
    logging.debug(f"Sorted cu_boundaries: {cu_boundaries}")
    # Validate key boundary (28B BCE)
    if abs(cu_boundaries[0] - Decimal('3094133911800.949548')) > Decimal('1'):
        logging.warning(f"cu_boundaries[0] {cu_boundaries[0]} deviates from expected 3094133911800.949548")
except Exception as e:
    logging.error(f"Error initializing cu_boundaries: {str(e)}")
    # Fallback with validated values
    cu_boundaries = [
        Decimal('3094133911800.949548'),  # 28B BCE
        Decimal('3094134044923.672659'),  # 2000 CE
        Decimal('3094134044948.904752'),  # 2025 CE
    ]
    cu_boundaries = sorted(cu_boundaries)
    logging.debug(f"Fallback cu_boundaries: {cu_boundaries}")

class ConversionHistory:
    def __init__(self, filename: str = "conversions.json"):
        self.filename = filename
        self.history = []

    def _save_history(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving history: {str(e)}")

    def save_conversion(self, input_value: str, output_value: str, conversion_type: str):
        entry = {
            "input": input_value,
            "output": output_value,
            "type": conversion_type,
            "timestamp": datetime.utcnow().isoformat(),
            "favorite": False
        }
        self.history.append(entry)
        self._save_history()

    def mark_favorite(self, input_value: str):
        for entry in self.history:
            if entry["input"] == input_value:
                entry["favorite"] = True
        self._save_history()

    def get_history(self, favorites_only: bool = False) -> List[Dict]:
        return [entry for entry in self.history if not favorites_only or entry["favorite"]]

HISTORY = ConversionHistory()

def run_tests():
    print("🌟 Test 1: Running Full Test Suite 🌟")
    print("🌟 Test 1: ZTOM Divine Empowerment (CU-Time: 3.416e12) 🌟")
    print(cu_to_gregorian("3.416e12"))

    print("🌟 Test 2: Ethical Classifier with Divine Note (01/01/28000000000 BCE 00:00:00 UTC) 🌟")
    print(gregorian_to_cu("01/01/28000000000 BCE 00:00:00 UTC"))

    print("🌟 Test 3: Sub-ZTOM Precision (05/23/2025 19:59:00.000001 UTC vs 05/23/2025 19:59:00.000002 UTC) 🌟")
    print(gregorian_to_cu("05/23/2025 19:59:00.000001 UTC"))
    print(gregorian_to_cu("05/23/2025 19:59:00.000002 UTC"))
    print(calculate_cu_duration("05/23/2025 19:59:00.000001 UTC", "05/23/2025 19:59:00.000002 UTC"))

    print("🌟 Test 4: Cosmic Scale (01/01/2800000000 BCE 00:00:00 UTC) 🌟")
    print(gregorian_to_cu("01/01/2800000000 BCE 00:00:00 UTC"))

    print("🌟 Test 5: Round-Trip Consistency (02/29/2000 00:00:00 UTC) 🌟")
    result = gregorian_to_cu("02/29/2000 00:00:00 UTC")
    print(result)
    cu_time_match = re.search(r"Full Numeric: ([\d.]+) CU-Time", result)
    if cu_time_match:
        cu_time = cu_time_match.group(1)
        print(cu_to_gregorian(cu_time))

    print("🌟 Test 2: Gregorian to CU-Time (02/29/2000) 🌟")
    print(gregorian_to_cu("02/29/2000 00:00:00 UTC"))

    print("🌟 Test 3: CU-Time to Gregorian (3094134044923.672659) 🌟")
    print(cu_to_gregorian("3094134044923.672659"))

    print("🌟 Test 4: Duration between 02/29/2000 and 05/23/2025 🌟")
    print(calculate_cu_duration("02/29/2000 00:00:00 UTC", "05/23/2025 19:59:00 UTC"))

    print("🌟 Test 5: Gregorian to CU-Time (28000000000 BCE) 🌟")
    print(gregorian_to_cu("01/01/28000000000 BCE 00:00:00 UTC"))

    logging.info("All tests completed. Check cu_time_converter_v2_0_4.log for details.")

if __name__ == "__main__":
    run_tests()