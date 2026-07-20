import { describe, expect, it } from 'vitest';
import {
  astronomicalToCivilYear,
  civilToAstronomicalYear,
  isGregorianLeapYear,
  toCuDecimal,
  validateCivilGregorianInput,
} from '..';

describe('proleptic Gregorian calendar policy', () => {
  it('maps civil BCE and CE years without a public year zero', () => {
    expect(civilToAstronomicalYear(toCuDecimal('1'), 'CE').toString()).toBe('1');
    expect(civilToAstronomicalYear(toCuDecimal('1'), 'BCE').toString()).toBe('0');
    expect(civilToAstronomicalYear(toCuDecimal('2'), 'BCE').toString()).toBe('-1');
    expect(astronomicalToCivilYear(toCuDecimal('0'))).toEqual({ civilYear: '1', era: 'BCE' });
    expect(astronomicalToCivilYear(toCuDecimal('-1'))).toEqual({ civilYear: '2', era: 'BCE' });
  });

  it('applies Gregorian leap rules to positive, zero, and negative years', () => {
    expect(isGregorianLeapYear(toCuDecimal('1900'))).toBe(false);
    expect(isGregorianLeapYear(toCuDecimal('2000'))).toBe(true);
    expect(isGregorianLeapYear(toCuDecimal('0'))).toBe(true);
    expect(isGregorianLeapYear(toCuDecimal('-400'))).toBe(true);
  });

  it('rejects invalid civil dates and year zero', () => {
    const invalidLeapDay = validateCivilGregorianInput({
      civilYear: '1900', era: 'CE', month: 2, day: 29,
    });
    const invalidMonthDay = validateCivilGregorianInput({
      civilYear: '2025', era: 'CE', month: 4, day: 31,
    });
    const yearZero = validateCivilGregorianInput({
      civilYear: '0000', era: 'CE', month: 1, day: 1,
    });

    expect(invalidLeapDay).toMatchObject({ ok: false, error: { code: 'INVALID_LEAP_DAY' } });
    expect(invalidMonthDay).toMatchObject({ ok: false, error: { code: 'INVALID_DAY_FOR_MONTH' } });
    expect(yearZero).toMatchObject({ ok: false, error: { code: 'YEAR_ZERO_NOT_ALLOWED' } });
  });

  it('defaults an omitted time to UTC midnight and rejects partial time', () => {
    const omitted = validateCivilGregorianInput({
      civilYear: '2000', era: 'CE', month: 1, day: 1,
    });
    const partial = validateCivilGregorianInput({
      civilYear: '2000', era: 'CE', month: 1, day: 1, time: { hour: 12 },
    });

    expect(omitted).toMatchObject({
      ok: true,
      value: { hour: 0, minute: 0, second: 0 },
    });
    expect(partial).toMatchObject({ ok: false, error: { code: 'PARTIAL_TIME' } });
  });
});
