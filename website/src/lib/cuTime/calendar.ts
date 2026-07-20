import type Decimal from 'decimal.js';
import { CuDecimal, toCuDecimal } from './decimal';
import { failure, type CuTimeResult } from './errors';
import type { CanonicalGregorianUtc, CivilGregorianUtc, Era } from './types';

export const MIN_ASTRONOMICAL_YEAR = toCuDecimal('-9999999999998');
export const MAX_ASTRONOMICAL_YEAR = toCuDecimal('9999999999999');
export const MAX_CIVIL_YEAR = toCuDecimal('9999999999999');

export function civilToAstronomicalYear(civilYear: Decimal, era: Era): Decimal {
  return era === 'BCE' ? toCuDecimal(1).minus(civilYear) : toCuDecimal(civilYear);
}

export function astronomicalToCivilYear(
  astronomicalYear: Decimal,
): { civilYear: string; era: Era } {
  if (astronomicalYear.greaterThanOrEqualTo(1)) {
    return { civilYear: astronomicalYear.toFixed(0), era: 'CE' };
  }

  return {
    civilYear: toCuDecimal(1).minus(astronomicalYear).toFixed(0),
    era: 'BCE',
  };
}

export function isGregorianLeapYear(astronomicalYear: Decimal): boolean {
  return (
    astronomicalYear.modulo(400).isZero() ||
    (astronomicalYear.modulo(4).isZero() && !astronomicalYear.modulo(100).isZero())
  );
}

export function daysInGregorianMonth(astronomicalYear: Decimal, month: number): number {
  if (month === 2) return isGregorianLeapYear(astronomicalYear) ? 29 : 28;
  if ([4, 6, 9, 11].includes(month)) return 30;
  return 31;
}

export function gregorianUtcToJdn(input: CanonicalGregorianUtc): Decimal {
  const month = toCuDecimal(input.month);
  const a = toCuDecimal(14).minus(month).dividedBy(12).floor();
  const y = input.astronomicalYear.plus(4800).minus(a);
  const m = month.plus(a.times(12)).minus(3);

  const jdnInteger = toCuDecimal(input.day)
    .plus(m.times(153).plus(2).dividedBy(5).floor())
    .plus(y.times(365))
    .plus(y.dividedBy(4).floor())
    .minus(y.dividedBy(100).floor())
    .plus(y.dividedBy(400).floor())
    .minus(32045);

  const secondsSinceMidnight = toCuDecimal(input.hour)
    .times(3600)
    .plus(toCuDecimal(input.minute).times(60))
    .plus(input.second);

  return jdnInteger.minus('0.5').plus(secondsSinceMidnight.dividedBy(86400));
}

function incrementGregorianDate(
  astronomicalYear: Decimal,
  month: number,
  day: number,
): { astronomicalYear: Decimal; month: number; day: number } {
  if (day < daysInGregorianMonth(astronomicalYear, month)) {
    return { astronomicalYear, month, day: day + 1 };
  }
  if (month < 12) {
    return { astronomicalYear, month: month + 1, day: 1 };
  }
  return { astronomicalYear: astronomicalYear.plus(1), month: 1, day: 1 };
}

export function jdnToGregorianUtc(jdn: Decimal): CuTimeResult<CivilGregorianUtc> {
  if (!jdn.isFinite()) return failure('NONFINITE_JDN', 'JDN must be finite.', 'jdn');

  const calendarJdn = jdn.toDecimalPlaces(24, CuDecimal.ROUND_HALF_UP);
  const jd = calendarJdn.plus('0.5');
  const z = jd.floor();
  const f = jd.minus(z);
  const alpha = z.minus('1867216.25').dividedBy('36524.25').floor();
  const a = z.plus(1).plus(alpha).minus(alpha.dividedBy(4).floor());
  const b = a.plus(1524);
  const c = b.minus('122.1').dividedBy('365.25').floor();
  const d = toCuDecimal('365.25').times(c).floor();
  const e = b.minus(d).dividedBy('30.6001').floor();
  const dayWithFraction = b
    .minus(d)
    .minus(toCuDecimal('30.6001').times(e).floor())
    .plus(f);

  let day = dayWithFraction.floor().toNumber();
  let month = (e.lessThan(14) ? e.minus(1) : e.minus(13)).toNumber();
  let astronomicalYear = month > 2 ? c.minus(4716) : c.minus(4715);

  const dayFraction = dayWithFraction.minus(dayWithFraction.floor());
  const totalSeconds = dayFraction
    .times(86400)
    .toDecimalPlaces(24, CuDecimal.ROUND_HALF_UP);
  let hour = totalSeconds.dividedBy(3600).floor().toNumber();
  let remaining = totalSeconds.minus(toCuDecimal(hour).times(3600));
  let minute = remaining.dividedBy(60).floor().toNumber();
  remaining = remaining.minus(toCuDecimal(minute).times(60));
  let second = remaining.toDecimalPlaces(0, CuDecimal.ROUND_HALF_UP).toNumber();

  if (second >= 60) {
    second -= 60;
    minute += 1;
  }
  if (minute >= 60) {
    minute -= 60;
    hour += 1;
  }
  if (hour >= 24) {
    hour -= 24;
    ({ astronomicalYear, month, day } = incrementGregorianDate(
      astronomicalYear,
      month,
      day,
    ));
  }

  if (
    astronomicalYear.lessThan(MIN_ASTRONOMICAL_YEAR) ||
    astronomicalYear.greaterThan(MAX_ASTRONOMICAL_YEAR)
  ) {
    return failure(
      'REVERSE_RESULT_OUT_OF_RANGE',
      'The converted Gregorian year is outside the supported range.',
      'cuTime',
    );
  }

  if (
    !Number.isInteger(month) ||
    month < 1 ||
    month > 12 ||
    !Number.isInteger(day) ||
    day < 1 ||
    day > daysInGregorianMonth(astronomicalYear, month) ||
    hour < 0 ||
    hour > 23 ||
    minute < 0 ||
    minute > 59 ||
    second < 0 ||
    second > 59
  ) {
    return failure(
      'STRUCTURALLY_INVALID_RESULT',
      'The reverse calendar calculation produced invalid components.',
    );
  }

  const civil = astronomicalToCivilYear(astronomicalYear);
  return {
    ok: true,
    value: { ...civil, month, day, hour, minute, second },
  };
}
