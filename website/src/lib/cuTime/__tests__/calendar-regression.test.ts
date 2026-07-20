import { describe, expect, it } from 'vitest';
import {
  convertGregorianToCuTime,
  isGregorianLeapYear,
  parseAndConvertCuTimeToGregorian,
  toCuDecimal,
  validateCivilGregorianInput,
} from '..';

describe('calendar regression coverage', () => {
  it('keeps October 1582 continuous under the proleptic Gregorian policy', () => {
    const dates = [4, 5, 14, 15].map((day) =>
      convertGregorianToCuTime({
        civilYear: '1582',
        era: 'CE',
        month: 10,
        day,
      }),
    );

    for (const result of dates) expect(result.ok).toBe(true);
    if (dates.some((result) => !result.ok)) return;

    const [october4, october5, october14, october15] = dates.map((result) => {
      if (!result.ok) throw new Error('Expected a valid proleptic Gregorian date.');
      return result.value.jdn;
    });

    expect(october5.minus(october4).toString()).toBe('1');
    expect(october14.minus(october5).toString()).toBe('9');
    expect(october15.minus(october14).toString()).toBe('1');
  });

  it('applies leap divisibility to negative astronomical years', () => {
    expect(isGregorianLeapYear(toCuDecimal('-1'))).toBe(false);
    expect(isGregorianLeapYear(toCuDecimal('-4'))).toBe(true);
    expect(isGregorianLeapYear(toCuDecimal('-100'))).toBe(false);
    expect(isGregorianLeapYear(toCuDecimal('-400'))).toBe(true);
  });

  it.each([
    [4, 31],
    [6, 31],
    [9, 31],
    [11, 31],
  ])('rejects day %2$i for 30-day month %1$i', (month, day) => {
    const result = validateCivilGregorianInput({
      civilYear: '2025',
      era: 'CE',
      month,
      day,
    });

    expect(result).toMatchObject({
      ok: false,
      error: {
        code: 'INVALID_DAY_FOR_MONTH',
        field: 'day',
        details: { maximumDay: '30' },
      },
    });
  });

  it('round-trips both sides of the BCE/CE boundary without public year zero', () => {
    const inputs = [
      { civilYear: '1', era: 'BCE' as const, month: 12, day: 31 },
      { civilYear: '1', era: 'CE' as const, month: 1, day: 1 },
      { civilYear: '2', era: 'BCE' as const, month: 1, day: 1 },
    ];

    for (const input of inputs) {
      const forward = convertGregorianToCuTime({
        ...input,
        time: { hour: 23, minute: 59, second: 59 },
      });
      expect(forward.ok).toBe(true);
      if (!forward.ok) continue;

      const reverse = parseAndConvertCuTimeToGregorian(forward.value.cuTime.toString());
      expect(reverse).toMatchObject({
        ok: true,
        value: { ...input, hour: 23, minute: 59, second: 59 },
      });
      if (reverse.ok) expect(reverse.value.civilYear).not.toBe('0');
    }
  });
});
