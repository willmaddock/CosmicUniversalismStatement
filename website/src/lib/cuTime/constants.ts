import { toCuDecimal } from './decimal';

/** Exact literals approved for the first-release converter engine. */
export const CU_TIME_LITERALS = Object.freeze({
  anchorJdn: '2451544.5',
  daysPerYear: '365.2421897',
  baseCuNasa: '3094213000000',
  offset: '78955076.49024643522707769749119',
  nasaUniverseAge: '13786999981.453',
});

const anchorJdn = toCuDecimal(CU_TIME_LITERALS.anchorJdn);
const daysPerYear = toCuDecimal(CU_TIME_LITERALS.daysPerYear);
const baseCuNasa = toCuDecimal(CU_TIME_LITERALS.baseCuNasa);
const offset = toCuDecimal(CU_TIME_LITERALS.offset);
const nasaUniverseAge = toCuDecimal(CU_TIME_LITERALS.nasaUniverseAge);

export const CU_TIME_CONSTANTS = Object.freeze({
  anchorJdn,
  daysPerYear,
  baseCuNasa,
  offset,
  nasaUniverseAge,
  bigBangCuNasa: baseCuNasa.minus(nasaUniverseAge),
});

/** Methodology metadata only. ANCHOR_JDN is the sole arithmetic anchor. */
export const CU_TIME_METADATA = Object.freeze({
  anchorCivilYear: '2000',
});
