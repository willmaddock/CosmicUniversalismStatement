import { describe, expect, it } from 'vitest';
import {
  convertCuTimeForDisplay,
  convertGregorianForDisplay,
  formatCivilGregorianUtc,
} from '..';

const VALID_INPUT = {
  civilYear: '2000',
  era: 'CE',
  month: '1',
  day: '1',
  hour: '0',
  minute: '0',
  second: '0',
};

describe('UI adapter regression coverage', () => {
  it('preserves extended civil years without truncation', () => {
    expect(formatCivilGregorianUtc({
      civilYear: '9999999999999',
      era: 'CE',
      month: 12,
      day: 31,
      hour: 23,
      minute: 59,
      second: 59,
    })).toBe('12/31/9999999999999 CE 23:59:59 UTC');
  });

  it.each([
    [{ ...VALID_INPUT, month: '' }, 'MONTH_REQUIRED', 'month'],
    [{ ...VALID_INPUT, day: '' }, 'DAY_REQUIRED', 'day'],
    [{ ...VALID_INPUT, month: '1.5' }, 'MALFORMED_NUMERIC_TEXT', 'month'],
    [{ ...VALID_INPUT, day: 'abc' }, 'MALFORMED_NUMERIC_TEXT', 'day'],
    [{ ...VALID_INPUT, second: '0.5' }, 'FRACTIONAL_TIME_NOT_SUPPORTED', 'second'],
    [{ ...VALID_INPUT, era: 'AD' }, 'ERA_INVALID', 'era'],
  ])('maps display input to stable field errors %#', (input, code, field) => {
    expect(convertGregorianForDisplay(input)).toMatchObject({
      ok: false,
      error: { code, field },
    });
  });

  it('trims Gregorian field text and returns canonical UTC context', () => {
    expect(convertGregorianForDisplay({
      civilYear: ' 2000 ',
      era: 'CE',
      month: ' 1 ',
      day: ' 1 ',
      hour: ' 0 ',
      minute: ' 0 ',
      second: ' 0 ',
    })).toMatchObject({
      ok: true,
      value: { gregorianUtc: '01/01/2000 CE 00:00:00 UTC' },
    });
  });

  it('never exposes lexical negative zero in target numeric output', () => {
    const positiveZero = convertCuTimeForDisplay('0');
    const negativeZero = convertCuTimeForDisplay('-0');

    expect(positiveZero.ok).toBe(true);
    expect(negativeZero.ok).toBe(true);
    if (!positiveZero.ok || !negativeZero.ok) return;
    expect(negativeZero.value.inputCuTime).toBe('0');
    expect(negativeZero.value).toEqual(positiveZero.value);
  });

  it('adds a six-place scientific companion without changing the exact value', () => {
    const result = convertGregorianForDisplay(VALID_INPUT);

    expect(result).toMatchObject({
      ok: true,
      value: {
        cuTime: '3094134044923.50975356477292230250881',
        cuTimeExponential: '3.094134e+12',
      },
    });
  });
});
