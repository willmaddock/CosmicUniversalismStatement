import Decimal from 'decimal.js';

export const CU_TIME_DECIMAL_PRECISION = 60;
export const CU_TIME_DECIMAL_ROUNDING = Decimal.ROUND_HALF_UP;

/**
 * An isolated Decimal constructor prevents unrelated code from changing the
 * precision or rounding policy used by CU-Time calculations.
 */
export const CuDecimal = Decimal.clone({
  precision: CU_TIME_DECIMAL_PRECISION,
  rounding: CU_TIME_DECIMAL_ROUNDING,
});

export type CuDecimalSource = Decimal.Value;

export function toCuDecimal(value: CuDecimalSource): Decimal {
  return new CuDecimal(value);
}
