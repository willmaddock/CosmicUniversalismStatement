import { describe, expect, it } from 'vitest';
import {
  convertCuTimeForDisplay,
  convertGregorianForDisplay,
  formatCivilGregorianUtc,
} from '..';

const ANCHOR_CU_TIME = '3094134044923.50975356477292230250881';
const ANCHOR_REFERENCE = {
  observableUniverseAlignedCuCoordinate: '3094213000000',
  observableUniverseAlignedCuCoordinateExponential: '3.094213e+12',
  observableAge: {
    kind: 'elapsed',
    label: 'Derived Age Since the Big Bang',
    years: '13786999981.453',
  },
  coordinateToObservableAgeRatio: {
    kind: 'available',
    label: 'CU Coordinate-to-Observable-Age Ratio',
    summary: '224.430',
    exact: '224.429752967469255696500492159497688263016147400751467748004',
  },
};

describe('CU-Time interface adapter', () => {
  it('formats the approved reverse anchor as canonical UTC', () => {
    expect(convertCuTimeForDisplay(ANCHOR_CU_TIME)).toEqual({
      ok: true,
      value: {
        inputCuTime: ANCHOR_CU_TIME,
        inputCuTimeExponential: '3.094134e+12',
        gregorianUtc: '01/01/2000 CE 00:00:00 UTC',
        observableUniverseReference: ANCHOR_REFERENCE,
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

describe('Gregorian interface adapter', () => {
  const anchorInput = {
    civilYear: '2000',
    era: 'CE',
    month: '1',
    day: '1',
    hour: '0',
    minute: '0',
    second: '0',
  };

  it('formats the approved forward anchor with the observable-universe reference', () => {
    expect(convertGregorianForDisplay(anchorInput)).toEqual({
      ok: true,
      value: {
        gregorianUtc: '01/01/2000 CE 00:00:00 UTC',
        cuTime: ANCHOR_CU_TIME,
        cuTimeExponential: '3.094134e+12',
        observableUniverseReference: ANCHOR_REFERENCE,
      },
    });
  });

  it('normalizes a wholly omitted time to UTC midnight', () => {
    expect(convertGregorianForDisplay({
      ...anchorInput,
      hour: '', minute: '', second: '',
    })).toMatchObject({
      ok: true,
      value: { gregorianUtc: '01/01/2000 CE 00:00:00 UTC' },
    });
  });

  it('returns the core calendar error for an impossible civil date', () => {
    expect(convertGregorianForDisplay({
      ...anchorInput,
      civilYear: '1900', month: '2', day: '29',
    })).toMatchObject({
      ok: false,
      error: { code: 'INVALID_LEAP_DAY', field: 'day' },
    });
  });

  it('rejects year zero and partial time while retaining stable field errors', () => {
    expect(convertGregorianForDisplay({
      ...anchorInput,
      civilYear: '0000',
    })).toMatchObject({
      ok: false,
      error: { code: 'YEAR_ZERO_NOT_ALLOWED', field: 'civilYear' },
    });
    expect(convertGregorianForDisplay({
      ...anchorInput,
      hour: '12', minute: '', second: '',
    })).toMatchObject({
      ok: false,
      error: { code: 'PARTIAL_TIME', field: 'time' },
    });
  });
});
