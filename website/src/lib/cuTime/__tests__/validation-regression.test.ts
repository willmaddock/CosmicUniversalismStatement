import { describe, expect, it } from 'vitest';
import { parseCuTimeInput, validateCivilGregorianInput } from '..';

describe('validation regression coverage', () => {
  it.each([
    [{ civilYear: '', era: 'CE' as const, month: 1, day: 1 }, 'YEAR_REQUIRED', 'civilYear'],
    [{ civilYear: '2000', era: '' as 'CE', month: 1, day: 1 }, 'ERA_INVALID', 'era'],
    [{ civilYear: '2000', era: 'CE' as const, month: 0, day: 1 }, 'MONTH_OUT_OF_RANGE', 'month'],
    [{ civilYear: '2000', era: 'CE' as const, month: 13, day: 1 }, 'MONTH_OUT_OF_RANGE', 'month'],
    [{ civilYear: '2000', era: 'CE' as const, month: 1, day: 0 }, 'DAY_OUT_OF_RANGE', 'day'],
    [{ civilYear: '2000', era: 'CE' as const, month: 1, day: 32 }, 'DAY_OUT_OF_RANGE', 'day'],
  ])('returns stable field errors for invalid civil input %#', (input, code, field) => {
    expect(validateCivilGregorianInput(input)).toMatchObject({
      ok: false,
      error: { code, field },
    });
  });

  it.each([
    [{ hour: -1, minute: 0, second: 0 }, 'HOUR_OUT_OF_RANGE', 'hour'],
    [{ hour: 24, minute: 0, second: 0 }, 'HOUR_OUT_OF_RANGE', 'hour'],
    [{ hour: 0, minute: -1, second: 0 }, 'MINUTE_OUT_OF_RANGE', 'minute'],
    [{ hour: 0, minute: 60, second: 0 }, 'MINUTE_OUT_OF_RANGE', 'minute'],
    [{ hour: 0, minute: 0, second: 60 }, 'LEAP_SECOND_NOT_SUPPORTED', 'second'],
    [{ hour: 0, minute: 0, second: 1.5 }, 'FRACTIONAL_TIME_NOT_SUPPORTED', 'second'],
  ])('returns stable errors for invalid time %#', (time, code, field) => {
    expect(validateCivilGregorianInput({
      civilYear: '2000',
      era: 'CE',
      month: 1,
      day: 1,
      time,
    })).toMatchObject({ ok: false, error: { code, field } });
  });

  it.each(['+1', '.5', '1.', '1e3', '1E3', 'Infinity', 'NaN', '1 2'])(
    'rejects unsupported CU-Time lexical form %j',
    (raw) => {
      expect(parseCuTimeInput(raw)).toMatchObject({
        ok: false,
        error: { code: 'INVALID_CU_TIME_SYNTAX', field: 'cuTime' },
      });
    },
  );

  it('preserves raw CU-Time text while trimming only the validated lexical form', () => {
    const result = parseCuTimeInput(' 0001.2500 ');

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.raw).toBe(' 0001.2500 ');
    expect(result.value.lexical).toBe('0001.2500');
    expect(result.value.value.toString()).toBe('1.25');
  });
});
