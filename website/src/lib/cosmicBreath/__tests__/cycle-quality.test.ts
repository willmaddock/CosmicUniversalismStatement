import { describe, expect, it } from 'vitest';
import {
  calculateBackingStoreSize,
  getQualityProfile,
  getViewportCategory,
  normalizeVisualCapabilities,
  resolveVisualQualityMode,
  shouldAnimateContinuously,
  VISUAL_QUALITY_MODES,
  type VisualCapabilities,
} from '../cycle-quality';

const capabilities = (
  overrides: Partial<VisualCapabilities> = {},
): VisualCapabilities => ({
  viewportWidth: 1440,
  viewportHeight: 900,
  devicePixelRatio: 1,
  prefersReducedMotion: false,
  documentVisible: true,
  hardwareConcurrency: 8,
  ...overrides,
});

describe('Cosmic Breath decorative quality policy', () => {
  it('exposes the four approved quality choices', () => {
    expect(VISUAL_QUALITY_MODES).toEqual(['auto', 'high', 'balanced', 'reduced']);
    expect(Object.isFrozen(VISUAL_QUALITY_MODES)).toBe(true);
  });

  it('resolves Auto deterministically from observable capabilities', () => {
    expect(resolveVisualQualityMode('auto', capabilities())).toBe('high');
    expect(resolveVisualQualityMode('auto', capabilities({ viewportWidth: 900 }))).toBe('balanced');
    expect(resolveVisualQualityMode('auto', capabilities({
      viewportWidth: 390,
      devicePixelRatio: 3,
    }))).toBe('reduced');
  });

  it('honors reduced motion for every requested mode', () => {
    for (const mode of VISUAL_QUALITY_MODES) {
      expect(resolveVisualQualityMode(mode, capabilities({ prefersReducedMotion: true })))
        .toBe('reduced');
    }
  });

  it('uses a conservative hidden-document Auto profile', () => {
    expect(resolveVisualQualityMode('auto', capabilities({ documentVisible: false })))
      .toBe('reduced');
  });

  it('keeps explicit choices stable when motion is allowed', () => {
    expect(resolveVisualQualityMode('high', capabilities({ viewportWidth: 320 }))).toBe('high');
    expect(resolveVisualQualityMode('balanced', capabilities())).toBe('balanced');
    expect(resolveVisualQualityMode('reduced', capabilities())).toBe('reduced');
  });

  it('classifies zero, narrow, standard, and wide viewports', () => {
    expect(getViewportCategory(capabilities({ viewportWidth: 0 }))).toBe('zero');
    expect(getViewportCategory(capabilities({ viewportWidth: 390 }))).toBe('narrow');
    expect(getViewportCategory(capabilities({ viewportWidth: 768, viewportHeight: 600 })))
      .toBe('standard');
    expect(getViewportCategory(capabilities())).toBe('wide');
  });

  it('normalizes safe finite capability bounds predictably', () => {
    expect(normalizeVisualCapabilities(capabilities({
      viewportWidth: -5,
      viewportHeight: -8,
      devicePixelRatio: 0,
      hardwareConcurrency: 3.9,
    }))).toMatchObject({
      viewportWidth: 0,
      viewportHeight: 0,
      devicePixelRatio: 1,
      hardwareConcurrency: 3,
    });
    expect(() => normalizeVisualCapabilities(capabilities({ viewportWidth: Number.NaN })))
      .toThrow(RangeError);
  });

  it('returns frozen profiles with decreasing resource budgets', () => {
    const high = getQualityProfile('high', capabilities());
    const balanced = getQualityProfile('balanced', capabilities());
    const reduced = getQualityProfile('reduced', capabilities());
    expect(high.particleBudget).toBeGreaterThan(balanced.particleBudget);
    expect(balanced.particleBudget).toBeGreaterThan(reduced.particleBudget);
    expect(high.maxBackingStorePixels).toBeGreaterThan(balanced.maxBackingStorePixels);
    expect(Object.isFrozen(high)).toBe(true);
  });

  it('caps backing-store ratio, dimensions, and total pixel count', () => {
    const profile = getQualityProfile('balanced', capabilities());
    const size = calculateBackingStoreSize(8000, 5000, 4, profile);
    expect(size.width).toBeLessThanOrEqual(4096);
    expect(size.height).toBeLessThanOrEqual(4096);
    expect(size.pixelCount).toBeLessThanOrEqual(profile.maxBackingStorePixels);
    expect(size.effectiveDevicePixelRatio).toBeLessThanOrEqual(profile.maxDevicePixelRatio);
    expect(calculateBackingStoreSize(0, 500, 2, profile)).toEqual({
      width: 0,
      height: 0,
      effectiveDevicePixelRatio: 0,
      pixelCount: 0,
    });
  });

  it('applies each profile device-pixel-ratio cap without persistence metadata', () => {
    for (const mode of ['high', 'balanced', 'reduced'] as const) {
      const profile = getQualityProfile(mode, capabilities());
      const size = calculateBackingStoreSize(320, 180, 8, profile);
      expect(size.effectiveDevicePixelRatio).toBeLessThanOrEqual(profile.maxDevicePixelRatio);
      expect(Object.keys(profile)).not.toEqual(expect.arrayContaining([
        'localStorage',
        'cookie',
        'urlState',
        'analytics',
        'telemetry',
        'remoteLogging',
      ]));
    }
  });

  it('permits continuous work only when the profile and page state allow it', () => {
    const high = getQualityProfile('high', capabilities());
    const reduced = getQualityProfile('reduced', capabilities());
    expect(shouldAnimateContinuously(high, capabilities())).toBe(true);
    expect(shouldAnimateContinuously(high, capabilities({ documentVisible: false }))).toBe(false);
    expect(shouldAnimateContinuously(high, capabilities({ prefersReducedMotion: true }))).toBe(false);
    expect(shouldAnimateContinuously(reduced, capabilities())).toBe(false);
  });

  it('rejects invalid modes and non-finite rendering inputs', () => {
    expect(() => resolveVisualQualityMode('ultra' as 'auto', capabilities())).toThrow(RangeError);
    const profile = getQualityProfile('balanced', capabilities());
    expect(() => calculateBackingStoreSize(Number.POSITIVE_INFINITY, 10, 1, profile))
      .toThrow(RangeError);
  });
});
