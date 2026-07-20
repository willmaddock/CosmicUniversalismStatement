import type Decimal from 'decimal.js';
import {
  MAX_CIVIL_YEAR,
  civilToAstronomicalYear,
  daysInGregorianMonth,
} from './calendar';
import { toCuDecimal } from './decimal';
import { failure, type CuTimeResult } from './errors';
import type {
  CanonicalGregorianUtc,
  CivilGregorianInput,
  ParsedCuTimeInput,
  UtcTimeInput,
} from './types';

const CIVIL_YEAR_PATTERN = /^\d{1,13}$/;
const CU_TIME_PATTERN = /^-?\d+(\.\d+)?$/;

function validateTime(
  time: Partial<UtcTimeInput> | null | undefined,
): CuTimeResult<UtcTimeInput> {
  if (time == null || Object.keys(time).length === 0) {
    return { ok: true, value: { hour: 0, minute: 0, second: 0 } };
  }

  if (time.hour == null || time.minute == null || time.second == null) {
    return failure('PARTIAL_TIME', 'Supply hour, minute, and second together.', 'time');
  }

  for (const [field, value] of Object.entries(time)) {
    if (!Number.isInteger(value)) {
      return failure(
        'FRACTIONAL_TIME_NOT_SUPPORTED',
        'Fractional time values are not supported.',
        field,
      );
    }
  }

  if (time.hour < 0 || time.hour > 23) {
    return failure('HOUR_OUT_OF_RANGE', 'Hour must be between 0 and 23.', 'hour');
  }
  if (time.minute < 0 || time.minute > 59) {
    return failure('MINUTE_OUT_OF_RANGE', 'Minute must be between 0 and 59.', 'minute');
  }
  if (time.second === 60) {
    return failure('LEAP_SECOND_NOT_SUPPORTED', 'Leap seconds are not supported.', 'second');
  }
  if (time.second < 0 || time.second > 59) {
    return failure('SECOND_OUT_OF_RANGE', 'Second must be between 0 and 59.', 'second');
  }

  return {
    ok: true,
    value: { hour: time.hour, minute: time.minute, second: time.second },
  };
}

export function validateCivilGregorianInput(
  input: CivilGregorianInput,
): CuTimeResult<CanonicalGregorianUtc> {
  if (!input.civilYear) return failure('YEAR_REQUIRED', 'Year is required.', 'civilYear');
  if (!CIVIL_YEAR_PATTERN.test(input.civilYear)) {
    return failure(
      'MALFORMED_NUMERIC_TEXT',
      'Year must contain 1 to 13 base-10 digits.',
      'civilYear',
    );
  }

  const civilYear = toCuDecimal(input.civilYear);
  if (civilYear.isZero()) {
    return failure('YEAR_ZERO_NOT_ALLOWED', 'Civil year zero is not allowed.', 'civilYear');
  }
  if (civilYear.greaterThan(MAX_CIVIL_YEAR)) {
    return failure('YEAR_OUT_OF_RANGE', 'Year exceeds the supported range.', 'civilYear');
  }
  if (input.era !== 'CE' && input.era !== 'BCE') {
    return failure('ERA_INVALID', 'Era must be CE or BCE.', 'era');
  }
  if (!Number.isInteger(input.month) || input.month < 1 || input.month > 12) {
    return failure('MONTH_OUT_OF_RANGE', 'Month must be between 1 and 12.', 'month');
  }
  if (!Number.isInteger(input.day) || input.day < 1 || input.day > 31) {
    return failure('DAY_OUT_OF_RANGE', 'Day must be between 1 and 31.', 'day');
  }

  const astronomicalYear = civilToAstronomicalYear(civilYear, input.era);
  const maximumDay = daysInGregorianMonth(astronomicalYear, input.month);
  if (input.day > maximumDay) {
    const code = input.month === 2 && input.day === 29 ? 'INVALID_LEAP_DAY' : 'INVALID_DAY_FOR_MONTH';
    return failure(code, `Day is invalid for the selected month and year.`, 'day', {
      maximumDay: String(maximumDay),
    });
  }

  const time = validateTime(input.time);
  if (!time.ok) return time;

  return {
    ok: true,
    value: {
      astronomicalYear,
      month: input.month,
      day: input.day,
      ...time.value,
    },
  };
}

export function parseCuTimeInput(raw: string): CuTimeResult<ParsedCuTimeInput> {
  const lexical = raw.trim();
  if (!lexical) return failure('CU_TIME_REQUIRED', 'CU-Time is required.', 'cuTime');
  if (!CU_TIME_PATTERN.test(lexical)) {
    return failure('INVALID_CU_TIME_SYNTAX', 'CU-Time must be a plain decimal number.', 'cuTime');
  }

  let value: Decimal;
  try {
    value = toCuDecimal(lexical);
  } catch {
    return failure('INVALID_CU_TIME_SYNTAX', 'CU-Time must be a plain decimal number.', 'cuTime');
  }
  if (!value.isFinite()) return failure('NONFINITE_CU_INPUT', 'CU-Time must be finite.', 'cuTime');

  return {
    ok: true,
    value: {
      raw,
      lexical,
      value: value.isZero() ? toCuDecimal('0') : value,
    },
  };
}
