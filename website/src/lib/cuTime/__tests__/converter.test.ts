import { describe, expect, it } from 'vitest';
import {
  convertCivilGregorianToCuTime,
  parseAndConvertCuTimeToGregorian,
  parseCuTimeInput,
} from '..';

const ANCHOR_CU_TIME = '3094134044923.50975356477292230250881';

describe('CU-Time mathematical core', () => {
  it('matches the approved forward anchor exactly', () => {
    const result = convertCivilGregorianToCuTime({
      civilYear: '2000',
      era: 'CE',
      month: 1,
      day: 1,
      time: { hour: 0, minute: 0, second: 0 },
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

  it('round-trips the approved anchor', () => {
    const result = parseAndConvertCuTimeToGregorian(ANCHOR_CU_TIME);
    expect(result).toMatchObject({
      ok: true,
      value: {
        civilYear: '2000', era: 'CE', month: 1, day: 1,
        hour: 0, minute: 0, second: 0,
      },
    });
  });

  it('round-trips BCE using the approved civil display mapping', () => {
    const forward = convertCivilGregorianToCuTime({
      civilYear: '1', era: 'BCE', month: 1, day: 1,
      time: { hour: 0, minute: 0, second: 0 },
    });
    expect(forward.ok).toBe(true);
    if (!forward.ok) return;
    const reverse = parseAndConvertCuTimeToGregorian(forward.value.cuTime.toString());
    expect(reverse).toMatchObject({
      ok: true,
      value: { civilYear: '1', era: 'BCE', month: 1, day: 1 },
    });
  });

  it('trims outer whitespace, rejects exponent syntax, and canonicalizes negative zero', () => {
    const trimmed = parseCuTimeInput(' 1.5 ');
    const exponent = parseCuTimeInput('1e3');
    const negativeZero = parseCuTimeInput('-0');

    expect(trimmed).toMatchObject({ ok: true, value: { lexical: '1.5' } });
    expect(exponent).toMatchObject({ ok: false, error: { code: 'INVALID_CU_TIME_SYNTAX' } });
    expect(negativeZero.ok).toBe(true);
    if (negativeZero.ok) {
      expect(negativeZero.value.raw).toBe('-0');
      expect(negativeZero.value.value.toString()).toBe('0');
    }
  });
});
