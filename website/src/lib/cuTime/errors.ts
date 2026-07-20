export type CuTimeErrorCode =
  | 'DATE_REQUIRED'
  | 'YEAR_REQUIRED'
  | 'YEAR_ZERO_NOT_ALLOWED'
  | 'YEAR_OUT_OF_RANGE'
  | 'ERA_REQUIRED'
  | 'ERA_INVALID'
  | 'MONTH_REQUIRED'
  | 'MONTH_OUT_OF_RANGE'
  | 'DAY_REQUIRED'
  | 'DAY_OUT_OF_RANGE'
  | 'INVALID_DAY_FOR_MONTH'
  | 'INVALID_LEAP_DAY'
  | 'PARTIAL_TIME'
  | 'HOUR_OUT_OF_RANGE'
  | 'MINUTE_OUT_OF_RANGE'
  | 'SECOND_OUT_OF_RANGE'
  | 'FRACTIONAL_TIME_NOT_SUPPORTED'
  | 'LEAP_SECOND_NOT_SUPPORTED'
  | 'MALFORMED_NUMERIC_TEXT'
  | 'CU_TIME_REQUIRED'
  | 'INVALID_CU_TIME_SYNTAX'
  | 'INVALID_CANONICAL_CALENDAR_INPUT'
  | 'INVALID_CANONICAL_CU_INPUT'
  | 'NONFINITE_CU_INPUT'
  | 'NONFINITE_JDN'
  | 'NONFINITE_CU_RESULT'
  | 'STRUCTURALLY_INVALID_RESULT'
  | 'REVERSE_RESULT_OUT_OF_RANGE'
  | 'INTERNAL_CONVERSION_ERROR';

export interface CuTimeError {
  code: CuTimeErrorCode;
  message: string;
  field?: string;
  details?: Readonly<Record<string, string>>;
}

export type CuTimeResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: CuTimeError };

export function failure(
  code: CuTimeErrorCode,
  message: string,
  field?: string,
  details?: Readonly<Record<string, string>>,
): CuTimeResult<never> {
  return { ok: false, error: { code, message, field, details } };
}
