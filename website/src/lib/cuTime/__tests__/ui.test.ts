import { describe, expect, it } from 'vitest';
import { convertCuTimeForDisplay, formatCivilGregorianUtc } from '..';

const ANCHOR_CU_TIME = '3094134044923.50975356477292230250881';

describe('CU-Time interface adapter', () => {
  it('formats the approved reverse anchor as canonical UTC', () => {
    expect(convertCuTimeForDisplay(ANCHOR_CU_TIME)).toEqual({
      ok: true,
      value: {
        inputCuTime: ANCHOR_CU_TIME,
        gregorianUtc: '01/01/2000 CE 00:00:00 UTC',
      },
    });
  });

  it('returns stable validation errors for empty and malformed input', () => {
    expect(convertCuTimeForDisplay('   ')).toMatchObject({
      ok: false,
      error: { code: 'CU_TIME_REQUIRED', field: 'cuTime' },
    });
    expect(convertCuTimeForDisplay('1e3')).toMatchObject({
      ok: false,
      error: { code: 'INVALID_CU_TIME_SYNTAX', field: 'cuTime' },
    });
  });

  it('trims input and canonicalizes mathematical negative zero', () => {
    const zero = convertCuTimeForDisplay(' -0 ');
    expect(zero).toMatchObject({
      ok: true,
      value: { inputCuTime: '0' },
    });
  });

  it('serializes BCE without exposing civil year zero', () => {
    expect(formatCivilGregorianUtc({
      civilYear: '1',
      era: 'BCE',
      month: 1,
      day: 1,
      hour: 0,
      minute: 0,
      second: 0,
    })).toBe('01/01/0001 BCE 00:00:00 UTC');
  });
});
