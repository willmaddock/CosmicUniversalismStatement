import type { TomBoundaryRole, TomPhase } from './cycle-runtime';
import type { QualityProfile } from './cycle-quality';

export type AtmosphereKind =
  | 'new-cosmic-seed'
  | 'expansion'
  | 'expansion-pause'
  | 'compression'
  | 'reset-pause';

export interface AtmosphereDescriptor {
  readonly kind: AtmosphereKind;
  readonly phase: TomPhase;
  readonly direction: 'outward' | 'inward';
  readonly color: string;
  readonly glowMultiplier: number;
}

export interface AtmosphereParticle {
  x: number;
  y: number;
  velocityX: number;
  velocityY: number;
  opacity: number;
  radius: number;
  life: number;
  generation: number;
}

const MAX_ELAPSED_MILLISECONDS = 50;
const PARTICLE_MARGIN = 0.08;

const descriptors: Readonly<Record<AtmosphereKind, AtmosphereDescriptor>> = Object.freeze({
  'new-cosmic-seed': Object.freeze({
    kind: 'new-cosmic-seed', phase: 'expansion', direction: 'outward', color: '#f6efc8', glowMultiplier: 1.25,
  }),
  expansion: Object.freeze({
    kind: 'expansion', phase: 'expansion', direction: 'outward', color: '#5de5ff', glowMultiplier: 1,
  }),
  'expansion-pause': Object.freeze({
    kind: 'expansion-pause', phase: 'expansion', direction: 'outward', color: '#f6efc8', glowMultiplier: 1.15,
  }),
  compression: Object.freeze({
    kind: 'compression', phase: 'compression', direction: 'inward', color: '#ffbf69', glowMultiplier: 0.9,
  }),
  'reset-pause': Object.freeze({
    kind: 'reset-pause', phase: 'compression', direction: 'inward', color: '#f6efc8', glowMultiplier: 1.3,
  }),
});

const hashInteger = (value: number): number => {
  let result = value | 0;
  result = Math.imul(result ^ (result >>> 16), 0x45d9f3b);
  result = Math.imul(result ^ (result >>> 16), 0x45d9f3b);
  return (result ^ (result >>> 16)) >>> 0;
};

const seededUnit = (seed: number, index: number, channel: number): number =>
  hashInteger((seed | 0) + Math.imul(index + 1, 0x9e3779b1) + Math.imul(channel + 1, 0x85ebca6b))
  / 0x1_0000_0000;

export const getAtmosphereDescriptor = (
  phase: TomPhase,
  boundaryRole: TomBoundaryRole,
): AtmosphereDescriptor => {
  if (boundaryRole === 'new-cosmic-seed') return descriptors['new-cosmic-seed'];
  if (boundaryRole === 'expansion-pause') return descriptors['expansion-pause'];
  if (boundaryRole === 'reset-pause') return descriptors['reset-pause'];
  return descriptors[phase];
};

export const structuralAtmosphereSeed = (stateId: string, cycleCount: number): number => {
  if (!Number.isInteger(cycleCount) || cycleCount < 0) {
    throw new RangeError('Atmosphere cycle count must be a non-negative integer');
  }
  let seed = Math.imul(cycleCount + 1, 0x9e3779b1);
  for (let index = 0; index < stateId.length; index += 1) {
    seed = Math.imul(seed ^ stateId.charCodeAt(index), 16777619);
  }
  return seed >>> 0;
};

export const clampAtmosphereElapsedTime = (elapsedMilliseconds: number): number => {
  if (!Number.isFinite(elapsedMilliseconds)) {
    throw new RangeError('Atmosphere elapsed time must be finite');
  }
  return Math.min(MAX_ELAPSED_MILLISECONDS, Math.max(0, elapsedMilliseconds));
};

const writeParticle = (
  particle: AtmosphereParticle,
  seed: number,
  index: number,
  generation: number,
  descriptor: AtmosphereDescriptor,
): void => {
  const angle = seededUnit(seed, index, generation * 7) * Math.PI * 2;
  const distance = descriptor.direction === 'outward'
    ? 0.035 + (seededUnit(seed, index, (generation * 7) + 1) * 0.37)
    : 0.18 + (seededUnit(seed, index, (generation * 7) + 1) * 0.35);
  const speed = 0.008 + (seededUnit(seed, index, (generation * 7) + 2) * 0.018);
  const directionSign = descriptor.direction === 'outward' ? 1 : -1;
  particle.x = 0.5 + (Math.cos(angle) * distance);
  particle.y = 0.5 + (Math.sin(angle) * distance);
  particle.velocityX = Math.cos(angle) * speed * directionSign;
  particle.velocityY = Math.sin(angle) * speed * directionSign;
  particle.opacity = 0.18 + (seededUnit(seed, index, (generation * 7) + 3) * 0.58);
  particle.radius = 0.45 + (seededUnit(seed, index, (generation * 7) + 4) * 1.15);
  particle.life = 0.45 + (seededUnit(seed, index, (generation * 7) + 5) * 0.55);
  particle.generation = generation;
};

export const initializeAtmosphereParticles = (
  seed: number,
  profile: QualityProfile,
  descriptor: AtmosphereDescriptor,
): AtmosphereParticle[] => Array.from({ length: profile.particleBudget }, (_, index) => {
  const particle: AtmosphereParticle = {
    x: 0,
    y: 0,
    velocityX: 0,
    velocityY: 0,
    opacity: 0,
    radius: 0,
    life: 0,
    generation: 0,
  };
  writeParticle(particle, seed, index, 0, descriptor);
  return particle;
});

export const updateAtmosphereParticles = (
  particles: AtmosphereParticle[],
  elapsedMilliseconds: number,
  seed: number,
  descriptor: AtmosphereDescriptor,
): AtmosphereParticle[] => {
  const elapsed = clampAtmosphereElapsedTime(elapsedMilliseconds);
  const elapsedSeconds = elapsed / 1000;
  for (let index = 0; index < particles.length; index += 1) {
    const particle = particles[index];
    particle.x += particle.velocityX * elapsedSeconds;
    particle.y += particle.velocityY * elapsedSeconds;
    particle.life = Math.max(0, particle.life - (elapsed / 12_000));
    const outsideBounds = particle.x < -PARTICLE_MARGIN
      || particle.x > 1 + PARTICLE_MARGIN
      || particle.y < -PARTICLE_MARGIN
      || particle.y > 1 + PARTICLE_MARGIN;
    if (particle.life === 0 || outsideBounds) {
      writeParticle(particle, seed, index, particle.generation + 1, descriptor);
    }
  }
  return particles;
};

export const createStaticAtmosphereFrame = (
  seed: number,
  profile: QualityProfile,
  descriptor: AtmosphereDescriptor,
): AtmosphereParticle[] => {
  const particles = initializeAtmosphereParticles(seed, profile, descriptor);
  for (const particle of particles) {
    particle.velocityX = 0;
    particle.velocityY = 0;
  }
  return particles;
};
