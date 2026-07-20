import type { CuTimeError } from './errors';
import { parseAndConvertCuTimeToGregorian } from './converter';
import type { CivilGregorianUtc } from './types';

export interface CuTimeDisplayResult {
  inputCuTime: string;
  gregorianUtc: string;
}

export type CuTimeDisplayOutcome =
  | { ok: true; value: CuTimeDisplayResult }
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
      gregorianUtc: formatCivilGregorianUtc(converted.value),
    },
  };
}
