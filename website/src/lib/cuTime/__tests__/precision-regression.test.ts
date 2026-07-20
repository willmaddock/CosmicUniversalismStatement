import { describe, expect, it } from 'vitest';
import {
  CU_TIME_CONSTANTS,
  convertCuTimeToGregorian,
  convertGregorianToCuTime,
  toCuDecimal,
} from '..';

const ANCHOR_INPUT = {
  civilYear: '2000',
  era: 'CE' as const,
  month: 1,
  day: 1,
};

describe('precision and coordinate regression coverage', () => {
  it('preserves exact one-day coordinate relationships', () => {
    const anchor = convertGregorianToCuTime(ANCHOR_INPUT);
    const nextDay = convertGregorianToCuTime({ ...ANCHOR_INPUT, day: 2 });

    expect(anchor.ok).toBe(true);
    expect(nextDay.ok).toBe(true);
    if (!anchor.ok || !nextDay.ok) return;

    expect(nextDay.value.jdn.minus(anchor.value.jdn).toString()).toBe('1');
    expect(nextDay.value.deltaJdn.minus(anchor.value.deltaJdn).toString()).toBe('1');
    expect(
      nextDay.value.nasaCuTime.minus(nextDay.value.cuTime).equals(CU_TIME_CONSTANTS.offset),
    ).toBe(true);
    expect(
      nextDay.value.yearsSinceBigBang.equals(
        nextDay.value.nasaCuTime.minus(CU_TIME_CONSTANTS.bigBangCuNasa),
      ),
    ).toBe(true);
  });

  it('retains a one-second displacement through the 60-digit Decimal pipeline', () => {
    const anchor = convertGregorianToCuTime(ANCHOR_INPUT);
    const nextSecond = convertGregorianToCuTime({
      ...ANCHOR_INPUT,
      time: { hour: 0, minute: 0, second: 1 },
    });

    expect(anchor.ok).toBe(true);
    expect(nextSecond.ok).toBe(true);
    if (!anchor.ok || !nextSecond.ok) return;

    const jdnDifference = nextSecond.value.jdn.minus(anchor.value.jdn);
    expect(jdnDifference.toString()).toBe(
      '0.00001157407407407407407407407407407407407407407407407',
    );
    expect(jdnDifference.times(86400).toString()).toBe(
      '0.999999999999999999999999999999999999999999999999648',
    );
    expect(nextSecond.value.cuTime.minus(anchor.value.cuTime).isPositive()).toBe(true);
  });

  it('treats the executable Big Bang coordinate as zero signed model-relative years', () => {
    const converterBigBang = CU_TIME_CONSTANTS.bigBangCuNasa.minus(CU_TIME_CONSTANTS.offset);
    const result = convertCuTimeToGregorian(converterBigBang);

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.nasaCuTime.toString()).toBe('3080426000018.547');
    expect(result.value.yearsSinceBigBang.toString()).toBe('0');
  });

  it('keeps pre-Big-Bang years signed instead of rejecting or clamping them', () => {
    const converterBeforeBigBang = CU_TIME_CONSTANTS.bigBangCuNasa
      .minus(CU_TIME_CONSTANTS.offset)
      .minus('1');
    const result = convertCuTimeToGregorian(converterBeforeBigBang);

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.yearsSinceBigBang.toString()).toBe('-1');
  });

  it('returns a structured range error for a finite CU coordinate beyond calendar support', () => {
    const result = convertCuTimeToGregorian(toCuDecimal('1e20'));

    expect(result).toMatchObject({
      ok: false,
      error: { code: 'REVERSE_RESULT_OUT_OF_RANGE', field: 'cuTime' },
    });
  });
});
