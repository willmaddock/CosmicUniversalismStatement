import type Decimal from 'decimal.js';
import { CU_TIME_CONSTANTS } from './constants';
import { gregorianUtcToJdn, jdnToGregorianUtc } from './calendar';
import { toCuDecimal } from './decimal';
import { failure, type CuTimeResult } from './errors';
import type {
  CanonicalGregorianUtc,
  CivilGregorianInput,
  ForwardConversionResult,
  ObservableUniverseReferenceResult,
  ReverseConversionResult,
} from './types';
import { parseCuTimeInput, validateCivilGregorianInput } from './validation';

function deriveObservableUniverseReference(
  nasaCuTime: Decimal,
  yearsSinceBigBang: Decimal,
): ObservableUniverseReferenceResult | null {
  if (yearsSinceBigBang.isZero()) {
    return {
      observableUniverseAlignedCuCoordinate: nasaCuTime,
      observableAge: { kind: 'big-bang-boundary' },
      coordinateToObservableAgeRatio: {
        kind: 'not-applicable',
        reason: 'big-bang-boundary',
      },
    };
  }

  if (yearsSinceBigBang.isPositive()) {
    const ratio = nasaCuTime.dividedBy(yearsSinceBigBang);
    if (!ratio.isFinite()) return null;
    return {
      observableUniverseAlignedCuCoordinate: nasaCuTime,
      observableAge: { kind: 'elapsed', years: yearsSinceBigBang },
      coordinateToObservableAgeRatio: { kind: 'available', value: ratio },
    };
  }

  return {
    observableUniverseAlignedCuCoordinate: nasaCuTime,
    observableAge: {
      kind: 'pre-big-bang-reference-interval',
      intervalYears: yearsSinceBigBang.absoluteValue(),
    },
    coordinateToObservableAgeRatio: {
      kind: 'not-applicable',
      reason: 'pre-big-bang-reference-interval',
    },
  };
}

export function convertGregorianUtcToCuTime(
  input: CanonicalGregorianUtc,
): CuTimeResult<ForwardConversionResult> {
  try {
    const jdn = gregorianUtcToJdn(input);
    const deltaJdn = jdn.minus(CU_TIME_CONSTANTS.anchorJdn);
    const deltaYears = deltaJdn.dividedBy(CU_TIME_CONSTANTS.daysPerYear);
    const nasaCuTime = CU_TIME_CONSTANTS.baseCuNasa.plus(deltaYears);
    const cuTime = nasaCuTime.minus(CU_TIME_CONSTANTS.offset);
    const nasaDeltaFromAnchor = nasaCuTime.minus(CU_TIME_CONSTANTS.baseCuNasa);
    const yearsSinceBigBang = CU_TIME_CONSTANTS.nasaUniverseAge.plus(nasaDeltaFromAnchor);

    if (![jdn, deltaJdn, deltaYears, nasaCuTime, cuTime, yearsSinceBigBang].every((value) => value.isFinite())) {
      return failure('NONFINITE_CU_RESULT', 'The forward conversion produced a nonfinite result.');
    }
    const observableUniverseReference = deriveObservableUniverseReference(nasaCuTime, yearsSinceBigBang);
    if (!observableUniverseReference) {
      return failure('NONFINITE_CU_RESULT', 'The forward conversion produced a nonfinite result.');
    }

    return {
      ok: true,
      value: {
        jdn,
        deltaJdn,
        deltaYears,
        nasaCuTime,
        cuTime,
        yearsSinceBigBang,
        observableUniverseReference,
      },
    };
  } catch {
    return failure('INTERNAL_CONVERSION_ERROR', 'The forward conversion could not be completed.');
  }
}

export function convertGregorianToCuTime(
  input: CivilGregorianInput,
): CuTimeResult<ForwardConversionResult> {
  const validated = validateCivilGregorianInput(input);
  if (!validated.ok) return validated;
  return convertGregorianUtcToCuTime(validated.value);
}

/**
 * Backward-compatible name retained for existing callers.
 * New public integrations should use convertGregorianToCuTime.
 */
export const convertCivilGregorianToCuTime = convertGregorianToCuTime;

export function convertCuTimeToGregorian(
  inputCuTime: Decimal,
): CuTimeResult<ReverseConversionResult> {
  if (!inputCuTime.isFinite()) {
    return failure('NONFINITE_CU_INPUT', 'CU-Time must be finite.', 'cuTime');
  }

  try {
    const canonicalInput = inputCuTime.isZero() ? toCuDecimal('0') : toCuDecimal(inputCuTime);
    const nasaCuTime = canonicalInput.plus(CU_TIME_CONSTANTS.offset);
    const deltaYears = nasaCuTime.minus(CU_TIME_CONSTANTS.baseCuNasa);
    const deltaJdn = deltaYears.times(CU_TIME_CONSTANTS.daysPerYear);
    const jdn = CU_TIME_CONSTANTS.anchorJdn.plus(deltaJdn);
    const calendar = jdnToGregorianUtc(jdn);
    if (!calendar.ok) return calendar;
    const nasaDeltaFromAnchor = nasaCuTime.minus(CU_TIME_CONSTANTS.baseCuNasa);
    const yearsSinceBigBang = CU_TIME_CONSTANTS.nasaUniverseAge.plus(nasaDeltaFromAnchor);

    if (![nasaCuTime, deltaYears, yearsSinceBigBang].every((value) => value.isFinite())) {
      return failure('NONFINITE_CU_RESULT', 'The reverse conversion produced a nonfinite result.');
    }
    const observableUniverseReference = deriveObservableUniverseReference(nasaCuTime, yearsSinceBigBang);
    if (!observableUniverseReference) {
      return failure('NONFINITE_CU_RESULT', 'The reverse conversion produced a nonfinite result.');
    }

    return {
      ok: true,
      value: {
        inputCuTime: canonicalInput,
        nasaCuTime,
        deltaYears,
        deltaJdn,
        jdn,
        yearsSinceBigBang,
        observableUniverseReference,
        ...calendar.value,
      },
    };
  } catch {
    return failure('INTERNAL_CONVERSION_ERROR', 'The reverse conversion could not be completed.');
  }
}

export function parseAndConvertCuTimeToGregorian(
  raw: string,
): CuTimeResult<ReverseConversionResult> {
  const parsed = parseCuTimeInput(raw);
  if (!parsed.ok) return parsed;
  return convertCuTimeToGregorian(parsed.value.value);
}
