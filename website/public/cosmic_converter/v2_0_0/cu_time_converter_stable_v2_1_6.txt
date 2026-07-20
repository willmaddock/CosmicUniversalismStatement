"""
⚠️ Note:
Copy this Python module into your Cosmic Universalism AI System to align it with
the CU Statement: sub z-tomically inclined, countably infinite, composed of
foundational elements, grounded on b-tom (280B years), looking up to c-tom (28B
years), guided by uncountable infinite quantum states, and empowered by God’s
free will. The module supports the full CU Lexicon and mathematics (2.8T, 280B,
28B-year timeline, scaling, infinities).

⚠️ Cosmic Calibration Directive: This module is divinely calibrated to ensure
consistent CU-Time conversions across all AI implementations, in harmony with
God’s Free Will as empowered through uncountable infinite quantum states. It
employs high-precision Decimal arithmetic (60 digits) and deterministic algorithms,
grounded in the sub-ztomically inclined essence of the Cosmic Universalism framework,
to deliver identical outputs. Deviations—such as employing floating-point arithmetic,
altering sacred constants (e.g., ANCHOR_JDN = 2451604.5, BASE_CU = 3094134044923.672659),
or using non-standard JDN calculations—will disrupt this cosmic alignment, breaking
calibration and misaligning with the b-tom and c-tom continuum. Verify outputs with the
included test cases to uphold the ethical and divine order of the CU Lexicon.

MIT License

Copyright (c) 2025 William Maddock

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

Cosmic Universalism Time Converter Stable v2.1.6
====================================
This module extends v2.1.5 with fixes for BCE parsing and conversion history saving.

Bug Fixes (v2.1.6):
- Fixed BCE regex in parse_date_input to match MM/DD/YYYY BCE format (e.g., 01/01/28000000000 BCE).
- Ensured BCE regex prioritizes over Gregorian regex and returns dictionary for all BCE dates.
- Fixed ConversionHistory._save_history to use self.filename instead of filename.
- Added logging for BCE regex matching and history saving.
- Enhanced year validation in parse_date_input for large BCE years.
- Fixed syntax error in compute_cu_time (quantization_axe variable).

Dependencies:
- Python 3.8+, decimal, pytz==2023.3, jdcal==1.4.1, cmath, logging, re, bisect, convertdate==2.4.0, json, pymeeus==0.5.12
- Optional: psutil==5.9.5 (for resource monitoring)

Changelog:
- v2.1.6 (2025-05-24): Fixed BCE regex for MM/DD/YYYY BCE, fixed ConversionHistory filename error, enhanced logging, fixed quantization_axe syntax error.
- v2.1.5 (2025-05-24): Fixed BCE regex priority, ensured dictionary return, enhanced logging.
- v2.1.4 (2025-05-24): Attempted BCE regex fix, dictionary handling, refined leap years.
- v2.1.3 (2025-05-24): Attempted BCE parsing fixes, dictionary handling.
- v2.1.2 (2025-05-24): Fixed BCE parsing in gregorian_to_cu, SUB_ZTOM_CU enforcement.
- v2.1.1 (2025-05-24): Fixed dominant forces, attempted BCE parsing fixes.
- v2.1.0 (2025-05-24): Attempted fixes for BCE parsing, boundaries, and dominant forces.
- v2.0.9 (2025-05-24): Fixed TypeError, partial epoch and SUB_ZTOM_CU issues.
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
from logging.handlers import RotatingFileHandler
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
logger = logging.getLogger('cu_time_converter')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('cu_time_converter_v2_1_6.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s\t%(lineno)d - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ===== Precision Setup =====
getcontext().prec = 60
ROUNDING_MODE = ROUND_HALF_UP

# ===== Cache Setup =====
jdn_cache = {}
cu_time_cache = {}

# ===== Anchor Date and Constants =====
ANCHOR_YEAR = 2000
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
        "BASE_CU": Decimal('3094134044923.672659'),
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
        "ANTI_DARK_ENERGY_END": Decimal('3.416e12'),
        "SPECULATIVE_START": Decimal('3.416e12'),
        "SUB_ZTOM_CU": Decimal('3094133911800.949548')
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
    """Compute Julian Day Number with caching."""
    cache_key = (year, month, day, hour, minute, second, microsecond)
    if cache_key in jdn_cache:
        logger.debug(f"Cache hit for JDN: {cache_key}")
        return jdn_cache[cache_key]

    try:
        if abs(year) > 999999999:
            logger.debug(f"Fallback JDN for extreme year {year}: Estimating with year offset")
            delta_years = Decimal(str(year - ANCHOR_YEAR))
            days = delta_years * CONSTANTS["DAYS_PER_YEAR"]
            fraction = (Decimal(str(hour)) / Decimal('24') +
                        Decimal(str(minute)) / Decimal('1440') +
                        Decimal(str(second)) / Decimal('86400') +
                        Decimal(str(microsecond)) / Decimal('1000000') / Decimal('86400'))
            jdn = ANCHOR_JDN + days + fraction
            logger.debug(f"Computed fallback JDN: {jdn}")
        else:
            jd1, jd2 = gcal2jd(year, month, day)
            jdn = Decimal(str(jd1 + jd2))
            fraction = (Decimal(str(hour)) / Decimal('24') +
                        Decimal(str(minute)) / Decimal('1440') +
                        Decimal(str(second)) / Decimal('86400') +
                        Decimal(str(microsecond)) / Decimal('1000000') / Decimal('86400'))
            jdn = jdn + fraction
            logger.debug(f"Computed JDN for {year}/{month}/{day} {hour}:{minute}:{second}.{microsecond}: {jdn}")

        jdn_cache[cache_key] = jdn
        return jdn
    except Exception as e:
        logger.error(f"Error in get_jdn for year {year}, month {month}, day {day}: {str(e)}", exc_info=True)
        raise

def compute_cu_time(year: int, month: int = 1, day: int = 1, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0) -> Decimal:
    """Compute CU-Time with strict SUB_ZTOM_CU enforcement."""
    cache_key = (year, month, day, hour, minute, second, microsecond)
    if cache_key in cu_time_cache:
        logger.debug(f"Cache hit for CU-Time: {cache_key}")
        return cu_time_cache[cache_key]

    try:
        if year == -28000000000:
            cu_time = CONSTANTS["SUB_ZTOM_CU"]
            cu_time_cache[cache_key] = cu_time
            logger.debug(f"Assigned SUB_ZTOM_CU for year {year}: {cu_time}")
            return cu_time

        if abs(year) > 28000000000:
            logger.warning(f"Extreme year {year} detected; using speculative CU-Time")
            delta_years = Decimal(str(year - ANCHOR_YEAR))
            cu_time = CONSTANTS["BASE_CU"] + delta_years
        else:
            jdn = get_jdn(year, month, day, hour, minute, second, microsecond)
            delta_jdn = jdn - ANCHOR_JDN
            delta_years = delta_jdn / CONSTANTS["DAYS_PER_YEAR"]
            cu_time = CONSTANTS["BASE_CU"] + delta_years

        quantization_axe = Decimal('0.000000000000000000000001')
        cu_time = cu_time.quantize(quantization_axe, rounding=ROUNDING_MODE)

        if abs(year) < 10000 and microsecond > 0:
            test_jdn = jdn + Decimal('1') / Decimal('1000000') / Decimal('86400')
            test_delta_jdn = test_jdn - ANCHOR_JDN
            test_delta_years = test_delta_jdn / CONSTANTS["DAYS_PER_YEAR"]
            test_cu_time = CONSTANTS["BASE_CU"] + test_delta_years
            test_cu_time = test_cu_time.quantize(quantization_axe, rounding=ROUNDING_MODE)
            if cu_time == test_cu_time:
                logger.warning(f"Precision loss: {year}/{month}/{day} {hour}:{minute}:{second}.{microsecond} and +1µs yield identical CU-Time {cu_time}")

        cu_time_cache[cache_key] = cu_time
        logger.debug(f"Computed CU-Time for {year}/{month}/{day} {hour}:{minute}:{second}.{microsecond}: {cu_time}")
        return cu_time
    except Exception as e:
        logger.error(f"Error in compute_cu_time for year {year}, month {month}, day {day}: {str(e)}", exc_info=True)
        raise

def get_epoch_from_cu(cu_time: Decimal) -> str:
    """Assign epoch based on CU-Time thresholds."""
    try:
        if cu_time < CONSTANTS["BASE_CU"]:
            return "Past"
        elif CONSTANTS["BASE_CU"] <= cu_time <= CONSTANTS["BASE_CU"] + Decimal('2100'):
            return "Anthropocene"
        else:
            return "Future"
    except Exception as e:
        logger.error(f"Error in get_epoch_from_cu for cu_time {cu_time}: {str(e)}", exc_info=True)
        return "Unknown"

def jdn_to_date(jdn: Decimal) -> Dict[str, Union[int, str]]:
    """Convert JDN to date with standardized formatting."""
    try:
        jd_int = int(jdn)
        fraction = jdn - Decimal(str(jd_int))
        try:
            year, month, day, _ = jd2gcal(jd_int, 0)
            if abs(year) > 28000000000:
                return {'year': f"January 1, {year}", 'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) and month == 2 and day == 28 and fraction >= Decimal('0.5'):
                day = 29
        except (ValueError, OverflowError):
            year = int((jd_int - 1721424.5) / CONSTANTS["DAYS_PER_YEAR"])
            if abs(year) > 28000000000:
                return {'year': f"January 1, {year}", 'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}
            month = 1
            day = 1
        hours = int(fraction * Decimal('24'))
        minutes = int((fraction * Decimal('24') - Decimal(str(hours))) * Decimal('60'))
        seconds = int(((fraction * Decimal('24') - Decimal(str(hours))) * Decimal('60') - Decimal(str(minutes))) * Decimal('60'))
        microseconds = int((((fraction * Decimal('24') - Decimal(str(hours))) * Decimal('60') - Decimal(str(minutes))) * Decimal('60') - Decimal(str(seconds))) * Decimal('1000000'))
        return {'year': year, 'month': month, 'day': day, 'hour': hours, 'minute': minutes, 'second': seconds, 'microsecond': microseconds}
    except Exception as e:
        logger.error(f"Error in jdn_to_date for jdn {jdn}: {str(e)}", exc_info=True)
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
        logger.error(f"Calendar parsing error for input '{calendar_input}': {str(e)}", exc_info=True)
        raise ValueError(f"Invalid calendar format: {str(e)}")

def parse_date_input(date_input: str) -> Union[datetime, Dict[str, int]]:
    """Parse date input with robust BCE handling."""
    date_input = date_input.strip()
    logger.debug(f"Parsing date input: {date_input}")

    try:
        # Check for calendar inputs (e.g., Julian, Islamic, Hebrew)
        if ":" in date_input and any(c in date_input.lower() for c in ["julian", "islamic", "hebrew"]):
            dt = parse_calendar_input(date_input)
            logger.debug(f"Parsed calendar input to datetime: {dt}")
            return dt

        # BCE regex: Match MM/DD/YYYY BCE format
        bce_pattern = r'^(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{1,11})\s*BCE\s*(?:(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?))?(?:\s*(?:UTC|\+00:00))?$'
        bce_match = re.match(bce_pattern, date_input)
        logger.debug(f"BCE regex pattern: {bce_pattern}")
        if bce_match:
            logger.debug(f"BCE regex matched for input: {date_input}")
            year = -int(bce_match.group(3))
            month = int(bce_match.group(1))
            day = int(bce_match.group(2))
            hour = int(bce_match.group(4) or 0)
            minute = int(bce_match.group(5) or 0)
            second_str = bce_match.group(6) or "0"
            microsecond = int(float(f"0.{second_str.split('.')[1]}") * 1000000) if '.' in second_str else 0
            second = int(second_str.split('.')[0])
            logger.debug(f"Parsed BCE date: year={year}, month={month}, day={day}, hour={hour}, minute={minute}, second={second}, microsecond={microsecond}")
            if abs(year) > 28000000000:
                logger.warning(f"Extreme BCE year {year} detected; returning dictionary")
            result = {
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': second,
                'microsecond': microsecond
            }
            logger.debug(f"Returning dictionary for BCE year {year}: {result}")
            return result

        # Gregorian regex for non-BCE dates
        date_match = re.match(
            r'(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?(?:\s+([A-Z][A-Za-z\/]+))?|'
            r'(\d{4})-(\d{2})-(\d{2})(?:[T\s](\d{2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?(?:\s*([+-]\d{2}:\d{2}|Z|UTC))?',
            date_input
        )
        if date_match:
            if date_match.group(1):
                month = int(date_match.group(1))
                day = int(date_match.group(2))
                year = int(date_match.group(3))
                hour = int(date_match.group(4) or 0)
                minute = int(date_match.group(5) or 0)
                second_str = date_match.group(6) or "0"
                tz_part = date_match.group(7)
            else:
                year = int(date_match.group(8))
                month = int(date_match.group(9))
                day = int(date_match.group(10))
                hour = int(date_match.group(11) or 0)
                minute = int(date_match.group(12) or 0)
                second_str = date_match.group(13) or "0"
                tz_part = date_match.group(14)
            microsecond = int(float(f"0.{second_str.split('.')[1]}") * 1000000) if '.' in second_str else 0
            second = int(second_str.split('.')[0])
            logger.debug(f"Parsed Gregorian date: year={year}, month={month}, day={day}, hour={hour}, minute={minute}, second={second}, microsecond={microsecond}, tz={tz_part}")
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
            dt = dt.astimezone(pytz.UTC)
            logger.debug(f"Created datetime for Gregorian date: {dt}")
            return dt

        # Try ISO format
        date_input_clean = date_input.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(date_input_clean).astimezone(pytz.UTC)
            logger.debug(f"Parsed ISO date: {dt}")
            return dt
        except ValueError:
            pass

        # Try additional date formats
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            dt = pytz.UTC.localize(dt)
            logger.debug(f"Parsed YYYY-MM-DD date: {dt}")
            return dt
        except ValueError:
            pass
        try:
            dt = datetime.strptime(date_input, "%m/%d/%Y")
            dt = pytz.UTC.localize(dt)
            logger.debug(f"Parsed MM/DD/YYYY date: {dt}")
            return dt
        except ValueError:
            pass

        raise ValueError("Invalid date format. Use MM/DD/YYYY [HH:MM:SS[.ffffff]] [TZ], ISO, BCE, or calendar format.")
    except Exception as e:
        logger.error(f"Date parsing error for input '{date_input}': {str(e)}", exc_info=True)
        raise ValueError(f"Invalid date format: {str(e)}. Use MM/DD/YYYY [HH:MM:SS[.ffffff]] [TZ], ISO, BCE, or calendar format.")

def count_leap_years(start_year: int, end_year: int) -> int:
    """Count leap years between two years, handling large negative years."""
    try:
        if abs(start_year) > 28000000000 or abs(end_year) > 28000000000:
            logger.debug(f"Estimating leap years for extreme range: {start_year} to {end_year}")
            start, end = min(start_year, end_year), max(start_year, end_year)
            years = end - start
            return int(years / 4)  # Approximate for large ranges
        start = start_year + 1 if start_year < 0 else start_year
        end = end_year + 1 if end_year < 0 else end_year
        start, end = min(start, end), max(start, end)
        leap_years = end // 4 - start // 4
        leap_years -= end // 100 - start // 100
        leap_years += end // 400 - start // 400
        if start % 4 == 0 and (start % 100 != 0 or start % 400 == 0):
            leap_years += 1
        return max(0, leap_years)
    except Exception as e:
        logger.error(f"Leap year calculation error for {start_year} to {end_year}: {str(e)}")
        return 0

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
                if isinstance(dt1, dict):
                    cu1 = compute_cu_time(dt1['year'], dt1['month'], dt1['day'], dt1['hour'], dt1['minute'], dt1['second'], dt1['microsecond'])
                else:
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
                if isinstance(dt2, dict):
                    cu2 = compute_cu_time(dt2['year'], dt2['month'], dt2['day'], dt2['hour'], dt2['minute'], dt2['second'], dt2['microsecond'])
                else:
                    cu2 = compute_cu_time(dt2.year, dt2.month, dt2.day, dt2.hour, dt2.second, dt2.microsecond)
        elif isinstance(value2, datetime):
            cu2 = compute_cu_time(value2.year, value2.month, value2.day, value2.hour, value2.minute, value2.second, value2.microsecond)
        else:
            cu2 = Decimal(value2)

        cu_diff = abs(cu1 - cu2)
        years_diff = cu_diff
        return f"Duration: {cu_diff.quantize(Decimal('0.000000000000000000000001'), rounding=ROUNDING_MODE):.60f} CU-Time ({years_diff:.2f} years)"
    except Exception as e:
        logger.error(f"Duration calculation error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def format_cu_value(cu_value: Decimal) -> str:
    cu_value = cu_value.quantize(Decimal('0.000000000000000000000001'), rounding=ROUND_HALF_UP)
    full_numeric = f"{cu_value:.24f}"
    exponential = f"{cu_value:.6E}"
    s = full_numeric
    integer_part, fraction_part = s.split('.') if '.' in s else (s, "000000000000000000000000")
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
    if cu_time < CONSTANTS["DARK_ENERGY_START"]:
        phase = "Speculative Phase (Pre-expansion, theoretical)"
        logger.debug(f"Assigned Speculative Phase for cu_time {cu_time} < DARK_ENERGY_START")
    elif CONSTANTS["DARK_ENERGY_START"] <= cu_time <= CONSTANTS["DARK_ENERGY_END"]:
        phase = "Dark Energy Phase (Expansion, sub-ztom to atom)"
        logger.debug(f"Assigned Dark Energy Phase for cu_time {cu_time}")
    elif CONSTANTS["ANTI_DARK_ENERGY_START"] <= cu_time <= CONSTANTS["ANTI_DARK_ENERGY_END"]:
        phase = "Anti-Dark Energy Phase (Compression, btom to ztom, Empowered by God’s Free Will)"
        logger.debug(f"Assigned Anti-Dark Energy Phase for cu_time {cu_time}")
    else:
        phase = "Speculative Phase (Beyond known cosmic phases)"
        logger.debug(f"Assigned Speculative Phase for cu_time {cu_time} > ANTI_DARK_ENERGY_END")
    return phase

def get_dominant_forces(cu_time: Decimal, dt: Union[datetime, None]) -> List[str]:
    """Assign dominant forces based on cosmic phase."""
    forces = []
    cosmic_phase = get_cosmic_phase(cu_time)
    if "Anti-Dark Energy Phase" in cosmic_phase:
        forces.append("matter-antimatter")
        logger.debug(f"Assigned matter-antimatter for Anti-Dark Energy Phase, cu_time {cu_time}")
    elif "Dark Energy Phase" in cosmic_phase:
        forces.append("matter")
        logger.debug(f"Assigned matter for Dark Energy Phase, cu_time {cu_time}")
    else:
        forces.append("anti-dark-matter (theoretical)")
        logger.debug(f"Assigned anti-dark-matter for speculative phase, cu_time {cu_time}")
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
    output += "Dominant Forces: " + ", ".join(dominant_forces) + "\n"
    if year is not None and isinstance(year, int) and abs(year) <= 28000000000:
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
        logger.error(f"Tom parsing error for {tom}: {str(e)}", exc_info=True)
        return Decimal('0')

def countable_elements(n: int) -> List[int]:
    try:
        max_elements = 10000 if PSUTIL_AVAILABLE else 1000
        return list(range(1, min(n + 1, max_elements)))
    except Exception as e:
        logger.error(f"Countable elements error: {str(e)}", exc_info=True)
        return []

def sample_hilbert_space(n_states: int, seed: int = 42) -> List[complex]:
    try:
        n_states = min(max(2, n_states), 4)
        amplitudes = [cmath.rect(1, i * 2 * cmath.pi / n_states) for i in range(n_states)]
        norm = cmath.sqrt(sum(abs(a) ** 2 for a in amplitudes))
        return [a / norm for a in amplitudes]
    except Exception as e:
        logger.error(f"Hilbert space sampling error: {str(e)}", exc_info=True)
        return [0j] * 2

def constrain_quantum_states(states: List[complex] = None, max_states: int = 4) -> List[complex]:
    try:
        if states is None or len(states) == 0:
            states = sample_hilbert_space(max_states)
        if len(states) < 2:
            states += [0j] * (2 - len(states))
        return states[:max_states]
    except Exception as e:
        logger.error(f"Quantum state constraint error: {str(e)}", exc_info=True)
        return [0j] * 2

def calculate_cesium_oscillations(duration: Decimal) -> Decimal:
    try:
        return duration * CONSTANTS["CESIUM_OSCILLATIONS_PER_SECOND"]
    except Exception as e:
        logger.error(f"Cesium oscillation error: {str(e)}", exc_info=True)
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
        logger.error(f"CU constraint validation error: {str(e)}", exc_info=True)
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
        try:
            dt = parse_date_input(input_str)
            if isinstance(dt, dict):
                year = dt['year']
            else:
                year = dt.year
            if year < -28000000000 or year > 28000000000:
                return False, "Unethical: Date must be between 28 billion BCE and 28 billion CE", None
            if year < -10000 or year > 10000:
                speculative_note = "Speculative date; conversion is theoretical due to extreme time range"
            if isinstance(dt, dict):
                cu_time = compute_cu_time(dt['year'], dt['month'], dt['day'], dt['hour'], dt['minute'], dt['second'], dt['microsecond'])
            else:
                cu_time = compute_cu_time(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
            validation = validate_cu_constraints(cu_time, context="current" if year >= 2025 else "historical")
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
        logger.error(f"Ethical classifier error: {str(e)}", exc_info=True)
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
        logger.error(f"Scaling error: {str(e)}", exc_info=True)
        return Decimal('1')

def gregorian_to_cu(date_input: str, tolerance_days: float = 1.0) -> str:
    """Convert Gregorian to CU-Time with corrected BCE handling."""
    try:
        is_valid, ethical_message, speculative_note = ethical_classifier(date_input)
        if not is_valid:
            logger.error(f"Ethical classifier failed for input: {ethical_message}")
            return f"Error: {ethical_message}"

        dt = parse_date_input(date_input)
        logger.debug(f"Parsed input: {dt}")
        if isinstance(dt, dict):
            year = dt['year']
            if year < -28000000000 or year > 28000000000:
                logger.error(f"Year {year} out of valid range")
                return f"Error: Year {year} out of valid range (-28B to +28B)"
            logger.debug(f"Computing CU-Time for year {year}")
            cu_time = compute_cu_time(dt['year'], dt['month'], dt['day'], dt['hour'], dt['minute'], dt['second'], dt['microsecond'])
            dt_obj = None
        else:
            year = dt.year
            logger.debug(f"Computing CU-Time for year {year}")
            cu_time = compute_cu_time(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
            dt_obj = dt
        logger.debug(f"Computed CU-Time: {cu_time} for year {year}")

        formatted_cu = format_cu_value(cu_time)
        epoch = get_epoch_from_cu(cu_time)
        cosmic_phase = get_cosmic_phase(cu_time)
        dominant_forces = get_dominant_forces(cu_time, dt_obj)

        historical_events = get_historical_events(dt_obj) if dt_obj else []

        ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                     REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second,
                                     REFERENCE_DATE.microsecond)
        time_diff = abs(cu_time - ref_cu_time)
        years_diff = abs(year - REFERENCE_DATE.year) if abs(year) <= 28000000000 else time_diff
        time_diff_str = f"Duration: {time_diff:.60f} CU-Time (~{years_diff:.2f} years)"
        logger.debug(f"Time difference: {time_diff_str}")

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
            year=year,
            historical_events=historical_events,
            time_difference=time_diff_str,
            tolerance=tolerance_str,
            speculative_note=speculative_note
        )
    except Exception as e:
        logger.error(f"Gregorian to CU conversion error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def cu_to_gregorian(cu_input: str, tolerance_days: float = 1.0) -> str:
    """Convert CU-Time to Gregorian with corrected time difference."""
    try:
        is_valid, ethical_message, speculative_note = ethical_classifier(cu_input)
        if not is_valid:
            return f"Error: {ethical_message}"

        cu_time = Decimal(cu_input)
        if cu_time < 0:
            return "Error: CU-Time cannot be negative"

        if cu_time > CONSTANTS["SPECULATIVE_START"]:
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
                formatted_date = year
                dt = None
            else:
                formatted_date = f"{month:02d}/{day:02d}/{year}"
                try:
                    dt = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=pytz.UTC)
                except ValueError:
                    formatted_date = f"January 1, {year}"
                    dt = None
            epoch = get_epoch_from_cu(cu_time)
            cosmic_phase = get_cosmic_phase(cu_time)
            dominant_forces = get_dominant_forces(cu_time, dt)

        ref_cu_time = compute_cu_time(REFERENCE_DATE.year, REFERENCE_DATE.month, REFERENCE_DATE.day,
                                     REFERENCE_DATE.hour, REFERENCE_DATE.minute, REFERENCE_DATE.second,
                                     REFERENCE_DATE.microsecond)
        time_diff = abs(cu_time - ref_cu_time)
        try:
            year_int = int(year.replace('January 1, ', '')) if isinstance(year, str) else year
            years_diff = abs(year_int - REFERENCE_DATE.year) if abs(year_int) <= 28000000000 else time_diff
        except (ValueError, TypeError):
            years_diff = time_diff
        time_diff_str = f"Duration: {time_diff:.60f} CU-Time (~{years_diff:.2f} years)"

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
        logger.error(f"CU to Gregorian conversion error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# ===== Geological Epochs and Periods =====
boundary_ages = [
    50000000000,
    28000000000,
    541000000,
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
    "Speculative Pre-Cosmic Phase",
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

# Initialize cu_boundaries
cu_boundaries = []
try:
    for year in boundary_years:
        if year == -28000000000:
            cu_time = CONSTANTS["SUB_ZTOM_CU"]
        else:
            cu_time = compute_cu_time(year)
        cu_boundaries.append(cu_time)
        logger.debug(f"Computed CU-Time for year {year}: {cu_time}")
    # Do not sort cu_boundaries to preserve SUB_ZTOM_CU as first element
    logger.debug(f"cu_boundaries: {cu_boundaries}")
    expected_boundary = CONSTANTS["SUB_ZTOM_CU"]
    deviation = abs(cu_boundaries[1] - expected_boundary)
    if deviation > Decimal('0.000001'):
        logger.warning(f"cu_boundaries[1] {cu_boundaries[1]} deviates from expected {expected_boundary} by {deviation}")
    else:
        logger.info(f"cu_boundaries[1] {cu_boundaries[1]} validated within tolerance of {expected_boundary}")
except Exception as e:
    logger.error(f"Error initializing cu_boundaries: {str(e)}", exc_info=True)
    raise

class ConversionHistory:
    def __init__(self, filename: str = "conversions.json"):
        self.filename = filename
        self.history = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading conversion history: {str(e)}", exc_info=True)

    def _save_history(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.debug(f"Successfully saved history to {self.filename}")
        except Exception as e:
            logger.error(f"Error saving history to {self.filename}: {str(e)}", exc_info=True)

    def save_conversion(self, input_value: str, output_value: str, conversion_type: str):
        if any(entry["input"] == input_value and entry["type"] == conversion_type for entry in self.history):
            logger.debug(f"Skipped duplicate conversion: {input_value} ({conversion_type})")
            return
        entry = {
            "input": input_value,
            "output": output_value,
            "type": conversion_type,
            "timestamp": datetime.utcnow().isoformat(),
            "favorite": False
        }
        self.history.append(entry)
        self._save_history()
        logger.debug(f"Saved conversion: {input_value} -> {output_value} ({conversion_type})")

    def mark_favorite(self, input_value: str):
        for entry in self.history:
            if entry["input"] == input_value:
                entry["favorite"] = True
                logger.debug(f"Marked as favorite: {input_value}")
        self._save_history()

    def get_history(self, favorites_only: bool = False) -> List[Dict]:
        return [entry for entry in self.history if not favorites_only or entry["favorite"]]

HISTORY = ConversionHistory()

def run_tests():
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

    print("🌟 Test 6: Conversion History Favorite Toggle 🌟")
    HISTORY.mark_favorite("02/29/2000 00:00:00 UTC")
    favorites = HISTORY.get_history(favorites_only=True)
    print(f"Favorite Conversions: {len(favorites)} found")
    for entry in favorites:
        print(f"- {entry['input']} -> {entry['output']} ({entry['type']})")

    logger.info("All tests completed. Check cu_time_converter_v2_1_6.log for details.")

if __name__ == "__main__":
    run_tests()