import { describe, expect, it } from 'vitest';
import {
  convertGregorianToCuTime,
  parseAndConvertCuTimeToGregorian,
} from '..';

const ANCHOR_CU_TIME = '3094134044923.50975356477292230250881';

describe('Gregorian-to-CU public adapter', () => {
  it('exposes the approved forward anchor values', () => {
    const result = convertGregorianToCuTime({
      civilYear: '2000',
      era: 'CE',
      month: 1,
      day: 1,
    });

    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(result.value.jdn.toString()).toBe('2451544.5');
    expect(result.value.deltaJdn.toString()).toBe('0');
    expect(result.value.deltaYears.toString()).toBe('0');
    expect(result.value.nasaCuTime.toString()).toBe('3094213000000');
    expect(result.value.cuTime.toString()).toBe(ANCHOR_CU_TIME);
    expect(result.value.yearsSinceBigBang.toString()).toBe('13786999981.453');
  });

  it('defaults an omitted time to UTC midnight through existing validation', () => {
    const omitted = convertGregorianToCuTime({
      civilYear: '2000', era: 'CE', month: 1, day: 1,
    });
    const midnight = convertGregorianToCuTime({
      civilYear: '2000', era: 'CE', month: 1, day: 1,
      time: { hour: 0, minute: 0, second: 0 },
    });

    expect(omitted.ok).toBe(true);
    expect(midnight.ok).toBe(true);
    if (!omitted.ok || !midnight.ok) return;
    expect(omitted.value.cuTime.toString()).toBe(midnight.value.cuTime.toString());
  });

  it('returns existing structured errors for invalid calendar boundaries', () => {
    const invalidLeapDay = convertGregorianToCuTime({
      civilYear: '1900', era: 'CE', month: 2, day: 29,
    });
    const yearZero = convertGregorianToCuTime({
      civilYear: '0000', era: 'CE', month: 1, day: 1,
    });
    const partialTime = convertGregorianToCuTime({
      civilYear: '2000', era: 'CE', month: 1, day: 1,
      time: { hour: 12 },
    });
    const impossibleMonthDay = convertGregorianToCuTime({
      civilYear: '2025', era: 'CE', month: 4, day: 31,
    });
    const aboveMaximumYear = convertGregorianToCuTime({
      civilYear: '10000000000000', era: 'CE', month: 1, day: 1,
    });

    expect(invalidLeapDay).toMatchObject({
      ok: false,
      error: { code: 'INVALID_LEAP_DAY', field: 'day' },
    });
    expect(yearZero).toMatchObject({
      ok: false,
      error: { code: 'YEAR_ZERO_NOT_ALLOWED', field: 'civilYear' },
    });
    expect(partialTime).toMatchObject({
      ok: false,
      error: { code: 'PARTIAL_TIME', field: 'time' },
    });
    expect(impossibleMonthDay).toMatchObject({
      ok: false,
      error: { code: 'INVALID_DAY_FOR_MONTH', field: 'day' },
    });
    expect(aboveMaximumYear).toMatchObject({
      ok: false,
      error: { code: 'MALFORMED_NUMERIC_TEXT', field: 'civilYear' },
    });
  });

  it('accepts the supported civil-year endpoints', () => {
    const maximumCe = convertGregorianToCuTime({
      civilYear: '9999999999999', era: 'CE', month: 12, day: 31,
    });
    const maximumBce = convertGregorianToCuTime({
      civilYear: '9999999999999', era: 'BCE', month: 1, day: 1,
    });

    expect(maximumCe.ok).toBe(true);
    expect(maximumBce.ok).toBe(true);
  });

  it('round-trips CE and BCE inputs through the existing reverse path', () => {
    const cases = [
      { civilYear: '2000', era: 'CE' as const, month: 2, day: 29 },
      { civilYear: '1', era: 'BCE' as const, month: 2, day: 29 },
    ];

    for (const input of cases) {
      const forward = convertGregorianToCuTime({
        ...input,
        time: { hour: 12, minute: 34, second: 56 },
      });
      expect(forward.ok).toBe(true);
      if (!forward.ok) continue;

      const reverse = parseAndConvertCuTimeToGregorian(forward.value.cuTime.toString());
      expect(reverse).toMatchObject({
        ok: true,
        value: {
          ...input,
          hour: 12,
          minute: 34,
          second: 56,
        },
      });
    }
  });
});
