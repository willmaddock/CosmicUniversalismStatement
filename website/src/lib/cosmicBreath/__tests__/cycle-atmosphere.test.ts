import { describe, expect, it } from 'vitest';
import {
  clampAtmosphereElapsedTime,
  createStaticAtmosphereFrame,
  getAtmosphereDescriptor,
  initializeAtmosphereParticles,
  structuralAtmosphereSeed,
  updateAtmosphereParticles,
} from '../cycle-atmosphere';
import { getQualityProfile, type VisualCapabilities } from '../cycle-quality';

const capabilities: VisualCapabilities = {
  viewportWidth: 1024,
  viewportHeight: 768,
  devicePixelRatio: 1,
  prefersReducedMotion: false,
  documentVisible: true,
  hardwareConcurrency: 8,
};

describe('Cosmic Breath deterministic decorative atmosphere', () => {
  it('maps structural phases and boundary roles to distinct decorative descriptors', () => {
    expect(getAtmosphereDescriptor('expansion', null).kind).toBe('expansion');
    expect(getAtmosphereDescriptor('compression', null).kind).toBe('compression');
    expect(getAtmosphereDescriptor('expansion', 'new-cosmic-seed').kind)
      .toBe('new-cosmic-seed');
    expect(getAtmosphereDescriptor('expansion', 'expansion-pause').kind)
      .toBe('expansion-pause');
    expect(getAtmosphereDescriptor('compression', 'reset-pause').kind)
      .toBe('reset-pause');
  });

  it('generates stable seeds from structural identity and breath count', () => {
    expect(structuralAtmosphereSeed('expansion-sub-ztom', 0))
      .toBe(structuralAtmosphereSeed('expansion-sub-ztom', 0));
    expect(structuralAtmosphereSeed('expansion-sub-ztom', 0))
      .not.toBe(structuralAtmosphereSeed('expansion-sub-ztom', 1));
    expect(() => structuralAtmosphereSeed('expansion-sub-ztom', -1)).toThrow(RangeError);
  });

  it('creates exactly the active profile particle budget', () => {
    const descriptor = getAtmosphereDescriptor('expansion', null);
    for (const mode of ['high', 'balanced', 'reduced'] as const) {
      const profile = getQualityProfile(mode, capabilities);
      expect(initializeAtmosphereParticles(42, profile, descriptor))
        .toHaveLength(profile.particleBudget);
    }
  });

  it('initializes identical particles for identical inputs', () => {
    const profile = getQualityProfile('balanced', capabilities);
    const descriptor = getAtmosphereDescriptor('expansion', null);
    expect(initializeAtmosphereParticles(31415, profile, descriptor))
      .toEqual(initializeAtmosphereParticles(31415, profile, descriptor));
  });

  it('creates different stable output for a different supplied seed', () => {
    const profile = getQualityProfile('balanced', capabilities);
    const descriptor = getAtmosphereDescriptor('expansion', null);
    expect(initializeAtmosphereParticles(31415, profile, descriptor))
      .not.toEqual(initializeAtmosphereParticles(31416, profile, descriptor));
  });

  it('initializes outward expansion and inward compression velocities', () => {
    const profile = getQualityProfile('balanced', capabilities);
    const expansion = initializeAtmosphereParticles(
      12,
      profile,
      getAtmosphereDescriptor('expansion', null),
    );
    const compression = initializeAtmosphereParticles(
      12,
      profile,
      getAtmosphereDescriptor('compression', null),
    );
    const radialVelocity = (particle: typeof expansion[number]): number =>
      ((particle.x - 0.5) * particle.velocityX) + ((particle.y - 0.5) * particle.velocityY);
    expect(expansion.every((particle) => radialVelocity(particle) > 0)).toBe(true);
    expect(compression.every((particle) => radialVelocity(particle) < 0)).toBe(true);
  });

  it('updates a particle array in place without per-frame array replacement', () => {
    const profile = getQualityProfile('balanced', capabilities);
    const descriptor = getAtmosphereDescriptor('expansion', null);
    const particles = initializeAtmosphereParticles(7, profile, descriptor);
    const originalX = particles[0].x;
    expect(updateAtmosphereParticles(particles, 16, 7, descriptor)).toBe(particles);
    expect(particles[0].x).not.toBe(originalX);
  });

  it('clamps large elapsed gaps and rejects non-finite gaps', () => {
    expect(clampAtmosphereElapsedTime(-20)).toBe(0);
    expect(clampAtmosphereElapsedTime(16)).toBe(16);
    expect(clampAtmosphereElapsedTime(5000)).toBe(50);
    expect(() => clampAtmosphereElapsedTime(Number.NaN)).toThrow(RangeError);
  });

  it('bounds a long frame to the same update as the maximum allowed frame', () => {
    const profile = getQualityProfile('reduced', capabilities);
    const descriptor = getAtmosphereDescriptor('compression', null);
    const longGap = initializeAtmosphereParticles(88, profile, descriptor);
    const cappedGap = initializeAtmosphereParticles(88, profile, descriptor);
    updateAtmosphereParticles(longGap, 10_000, 88, descriptor);
    updateAtmosphereParticles(cappedGap, 50, 88, descriptor);
    expect(longGap).toEqual(cappedGap);
  });

  it('respawns out-of-bounds particles deterministically inside safe bounds', () => {
    const profile = getQualityProfile('reduced', capabilities);
    const descriptor = getAtmosphereDescriptor('expansion', null);
    const particles = initializeAtmosphereParticles(99, profile, descriptor);
    particles[0].x = 2;
    updateAtmosphereParticles(particles, 16, 99, descriptor);
    expect(particles[0].generation).toBe(1);
    expect(particles[0].x).toBeGreaterThanOrEqual(0);
    expect(particles[0].x).toBeLessThanOrEqual(1);
  });

  it('provides a deterministic motionless frame for reduced rendering', () => {
    const profile = getQualityProfile('reduced', capabilities);
    const descriptor = getAtmosphereDescriptor('expansion', 'new-cosmic-seed');
    const frame = createStaticAtmosphereFrame(123, profile, descriptor);
    expect(frame).toEqual(createStaticAtmosphereFrame(123, profile, descriptor));
    expect(frame.every((particle) => (
      particle.velocityX === 0 && particle.velocityY === 0
    ))).toBe(true);
  });

  it('keeps particle geometry and opacity within defined initialization bounds', () => {
    const profile = getQualityProfile('high', capabilities);
    const particles = initializeAtmosphereParticles(
      2718,
      profile,
      getAtmosphereDescriptor('compression', 'reset-pause'),
    );
    expect(particles.every((particle) => (
      Number.isFinite(particle.x)
      && Number.isFinite(particle.y)
      && particle.opacity >= 0
      && particle.opacity <= 1
      && particle.radius > 0
      && particle.life > 0
      && particle.life <= 1
    ))).toBe(true);
    for (let frame = 0; frame < 1000; frame += 1) {
      updateAtmosphereParticles(
        particles,
        50,
        2718,
        getAtmosphereDescriptor('compression', 'reset-pause'),
      );
    }
    expect(particles.every((particle) => (
      particle.opacity >= 0
      && particle.opacity <= 1
      && particle.life >= 0
      && particle.life <= 1
    ))).toBe(true);
  });

  it('uses structural decorative fields only', () => {
    const profile = getQualityProfile('reduced', capabilities);
    const descriptor = getAtmosphereDescriptor('expansion', 'new-cosmic-seed');
    const particle = initializeAtmosphereParticles(1, profile, descriptor)[0];
    const fieldNames = [...Object.keys(descriptor), ...Object.keys(particle)];
    expect(fieldNames).not.toEqual(expect.arrayContaining([
      'duration',
      'formula',
      'notation',
      'magnitude',
      'chronology',
    ]));
  });
});
