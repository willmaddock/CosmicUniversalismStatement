import type { CuTimeError } from './errors';
import { failure } from './errors';
import {
  convertGregorianToCuTime,
  parseAndConvertCuTimeToGregorian,
} from './converter';
import type { CivilGregorianInput, CivilGregorianUtc, Era } from './types';

export interface CuTimeDisplayResult {
  inputCuTime: string;
  inputCuTimeExponential: string;
  gregorianUtc: string;
}

export type CuTimeDisplayOutcome =
  | { ok: true; value: CuTimeDisplayResult }
  | { ok: false; error: CuTimeError };

export interface GregorianDisplayInput {
  civilYear: string;
  era: string;
  month: string;
  day: string;
  hour: string;
  minute: string;
  second: string;
}

export interface GregorianToCuTimeDisplayResult {
  gregorianUtc: string;
  cuTime: string;
  cuTimeExponential: string;
}

export type GregorianToCuTimeDisplayOutcome =
  | { ok: true; value: GregorianToCuTimeDisplayResult }
  | { ok: false; error: CuTimeError };

function padComponent(value: number): string {
  return String(value).padStart(2, '0');
}

export function formatCivilGregorianUtc(value: CivilGregorianUtc): string {
  const year = value.civilYear.padStart(4, '0');
  return [
    `${padComponent(value.month)}/${padComponent(value.day)}/${year} ${value.era}`,
    `${padComponent(value.hour)}:${padComponent(value.minute)}:${padComponent(value.second)} UTC`,
  ].join(' ');
}

export function convertCuTimeForDisplay(raw: string): CuTimeDisplayOutcome {
  const converted = parseAndConvertCuTimeToGregorian(raw);
  if (!converted.ok) return converted;

  return {
    ok: true,
    value: {
      inputCuTime: converted.value.inputCuTime.toString(),
      inputCuTimeExponential: converted.value.inputCuTime.toExponential(6),
      gregorianUtc: formatCivilGregorianUtc(converted.value),
    },
  };
}

function parseRequiredInteger(
  raw: string,
  field: 'month' | 'day',
): ReturnType<typeof failure> | number {
  const lexical = raw.trim();
  if (!lexical) {
    return failure(
      field === 'month' ? 'MONTH_REQUIRED' : 'DAY_REQUIRED',
      `${field === 'month' ? 'Month' : 'Day'} is required.`,
      field,
    );
  }
  if (!/^\d+$/.test(lexical)) {
    return failure('MALFORMED_NUMERIC_TEXT', `${field === 'month' ? 'Month' : 'Day'} must be a whole number.`, field);
  }
  return Number(lexical);
}

function parseOptionalTime(input: GregorianDisplayInput):
  | ReturnType<typeof failure>
  | CivilGregorianInput['time'] {
  const fields = [
    ['hour', input.hour],
    ['minute', input.minute],
    ['second', input.second],
  ] as const;
  const supplied = fields.filter(([, raw]) => raw.trim() !== '');

  if (supplied.length === 0) return null;
  if (supplied.length !== fields.length) {
    return failure('PARTIAL_TIME', 'Supply hour, minute, and second together.', 'time');
  }

  const parsed: Record<string, number> = {};
  for (const [field, raw] of fields) {
    const lexical = raw.trim();
    if (!/^\d+$/.test(lexical)) {
      const code = lexical.includes('.') ? 'FRACTIONAL_TIME_NOT_SUPPORTED' : 'MALFORMED_NUMERIC_TEXT';
      return failure(code, lexical.includes('.')
        ? 'Fractional time values are not supported.'
        : `${field[0].toUpperCase()}${field.slice(1)} must be a whole number.`, field);
    }
    parsed[field] = Number(lexical);
  }

  return {
    hour: parsed.hour,
    minute: parsed.minute,
    second: parsed.second,
  };
}

export function convertGregorianForDisplay(
  raw: GregorianDisplayInput,
): GregorianToCuTimeDisplayOutcome {
  const month = parseRequiredInteger(raw.month, 'month');
  if (typeof month !== 'number') return month;
  const day = parseRequiredInteger(raw.day, 'day');
  if (typeof day !== 'number') return day;
  const time = parseOptionalTime(raw);
  if (time && 'ok' in time) return time;

  const input: CivilGregorianInput = {
    civilYear: raw.civilYear.trim(),
    era: raw.era as Era,
    month,
    day,
    time,
  };
  const converted = convertGregorianToCuTime(input);
  if (!converted.ok) return converted;

  const validatedTime = time ?? { hour: 0, minute: 0, second: 0 };
  return {
    ok: true,
    value: {
      gregorianUtc: formatCivilGregorianUtc({
        civilYear: input.civilYear,
        era: input.era,
        month: input.month,
        day: input.day,
        ...validatedTime,
      }),
      cuTime: converted.value.cuTime.toFixed(),
      cuTimeExponential: converted.value.cuTime.toExponential(6),
    },
  };
}
