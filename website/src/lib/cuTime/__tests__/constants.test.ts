import { describe, expect, it } from 'vitest';
import {
  CU_TIME_CONSTANTS,
  CU_TIME_DECIMAL_PRECISION,
  CU_TIME_DECIMAL_ROUNDING,
  CU_TIME_LITERALS,
  CuDecimal,
} from '..';

describe('CU-Time constant registry', () => {
  it('preserves the approved direct literals exactly', () => {
    expect(CU_TIME_LITERALS).toEqual({
      anchorJdn: '2451544.5',
      daysPerYear: '365.2421897',
      baseCuNasa: '3094213000000',
      offset: '78955076.49024643522707769749119',
      nasaUniverseAge: '13786999981.453',
    });
  });

  it('uses the approved Decimal policy', () => {
    expect(CU_TIME_DECIMAL_PRECISION).toBe(60);
    expect(CU_TIME_DECIMAL_ROUNDING).toBe(CuDecimal.ROUND_HALF_UP);
  });

  it('derives the executable Big Bang coordinate', () => {
    expect(CU_TIME_CONSTANTS.bigBangCuNasa.toString()).toBe('3080426000018.547');
  });
});
