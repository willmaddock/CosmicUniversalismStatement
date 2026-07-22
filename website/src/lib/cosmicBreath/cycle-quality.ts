export const VISUAL_QUALITY_MODES = Object.freeze([
  'auto',
  'high',
  'balanced',
  'reduced',
] as const);

export type VisualQualityMode = typeof VISUAL_QUALITY_MODES[number];
export type ResolvedVisualQuality = Exclude<VisualQualityMode, 'auto'>;
export type ViewportCategory = 'zero' | 'narrow' | 'standard' | 'wide';

export interface VisualCapabilities {
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly devicePixelRatio: number;
  readonly prefersReducedMotion: boolean;
  readonly documentVisible: boolean;
  readonly hardwareConcurrency?: number;
}

export interface QualityProfile {
  readonly mode: ResolvedVisualQuality;
  readonly particleBudget: number;
  readonly maxDevicePixelRatio: number;
  readonly maxBackingStorePixels: number;
  readonly targetFramesPerSecond: number;
  readonly glowStrength: number;
  readonly trailStrength: number;
  readonly continuousAnimation: boolean;
}

export interface BackingStoreSize {
  readonly width: number;
  readonly height: number;
  readonly effectiveDevicePixelRatio: number;
  readonly pixelCount: number;
}

const MAX_CANVAS_DIMENSION = 4096;

const profiles: Readonly<Record<ResolvedVisualQuality, QualityProfile>> = Object.freeze({
  high: Object.freeze({
    mode: 'high',
    particleBudget: 144,
    maxDevicePixelRatio: 2,
    maxBackingStorePixels: 2_400_000,
    targetFramesPerSecond: 60,
    glowStrength: 0.86,
    trailStrength: 0.2,
    continuousAnimation: true,
  }),
  balanced: Object.freeze({
    mode: 'balanced',
    particleBudget: 72,
    maxDevicePixelRatio: 1.5,
    maxBackingStorePixels: 1_200_000,
    targetFramesPerSecond: 40,
    glowStrength: 0.58,
    trailStrength: 0.08,
    continuousAnimation: true,
  }),
  reduced: Object.freeze({
    mode: 'reduced',
    particleBudget: 20,
    maxDevicePixelRatio: 1,
    maxBackingStorePixels: 480_000,
    targetFramesPerSecond: 1,
    glowStrength: 0.28,
    trailStrength: 0,
    continuousAnimation: false,
  }),
});

const finiteNonNegative = (value: number, name: string): number => {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return Math.max(0, value);
};

export const normalizeVisualCapabilities = (
  capabilities: VisualCapabilities,
): VisualCapabilities => Object.freeze({
  viewportWidth: finiteNonNegative(capabilities.viewportWidth, 'Viewport width'),
  viewportHeight: finiteNonNegative(capabilities.viewportHeight, 'Viewport height'),
  devicePixelRatio: Math.max(
    1,
    finiteNonNegative(capabilities.devicePixelRatio, 'Device pixel ratio'),
  ),
  prefersReducedMotion: capabilities.prefersReducedMotion === true,
  documentVisible: capabilities.documentVisible === true,
  ...(capabilities.hardwareConcurrency === undefined
    ? {}
    : {
        hardwareConcurrency: Math.max(
          1,
          Math.floor(finiteNonNegative(capabilities.hardwareConcurrency, 'Hardware concurrency')),
        ),
      }),
});

export const getViewportCategory = (
  capabilities: VisualCapabilities,
): ViewportCategory => {
  const normalized = normalizeVisualCapabilities(capabilities);
  if (normalized.viewportWidth === 0 || normalized.viewportHeight === 0) return 'zero';
  if (normalized.viewportWidth < 480 || normalized.viewportHeight < 360) return 'narrow';
  if (normalized.viewportWidth >= 1180 && normalized.viewportHeight >= 620) return 'wide';
  return 'standard';
};

export const resolveVisualQualityMode = (
  requestedMode: VisualQualityMode,
  capabilities: VisualCapabilities,
): ResolvedVisualQuality => {
  if (!VISUAL_QUALITY_MODES.includes(requestedMode)) {
    throw new RangeError(`Unknown visual quality mode: ${requestedMode}`);
  }
  const normalized = normalizeVisualCapabilities(capabilities);
  if (normalized.prefersReducedMotion) return 'reduced';
  if (requestedMode !== 'auto') return requestedMode;
  if (!normalized.documentVisible || getViewportCategory(normalized) === 'zero') return 'reduced';

  const category = getViewportCategory(normalized);
  const lowConcurrency = normalized.hardwareConcurrency !== undefined
    && normalized.hardwareConcurrency <= 4;
  if (category === 'narrow' && (normalized.devicePixelRatio > 2 || lowConcurrency)) {
    return 'reduced';
  }
  if (category !== 'wide' || normalized.devicePixelRatio > 2 || lowConcurrency) {
    return 'balanced';
  }
  return 'high';
};

export const getQualityProfile = (
  requestedMode: VisualQualityMode,
  capabilities: VisualCapabilities,
): QualityProfile => profiles[resolveVisualQualityMode(requestedMode, capabilities)];

export const shouldAnimateContinuously = (
  profile: QualityProfile,
  capabilities: VisualCapabilities,
): boolean => {
  const normalized = normalizeVisualCapabilities(capabilities);
  return profile.continuousAnimation
    && normalized.documentVisible
    && !normalized.prefersReducedMotion
    && normalized.viewportWidth > 0
    && normalized.viewportHeight > 0;
};

export const calculateBackingStoreSize = (
  cssWidth: number,
  cssHeight: number,
  devicePixelRatio: number,
  profile: QualityProfile,
): BackingStoreSize => {
  const width = finiteNonNegative(cssWidth, 'Canvas CSS width');
  const height = finiteNonNegative(cssHeight, 'Canvas CSS height');
  const requestedRatio = Math.max(1, finiteNonNegative(devicePixelRatio, 'Device pixel ratio'));
  if (width === 0 || height === 0) {
    return Object.freeze({
      width: 0,
      height: 0,
      effectiveDevicePixelRatio: 0,
      pixelCount: 0,
    });
  }

  const cappedRatio = Math.min(requestedRatio, profile.maxDevicePixelRatio);
  const rawWidth = width * cappedRatio;
  const rawHeight = height * cappedRatio;
  const scale = Math.min(
    1,
    MAX_CANVAS_DIMENSION / rawWidth,
    MAX_CANVAS_DIMENSION / rawHeight,
    Math.sqrt(profile.maxBackingStorePixels / (rawWidth * rawHeight)),
  );
  const backingWidth = Math.max(1, Math.floor(rawWidth * scale));
  const backingHeight = Math.max(1, Math.floor(rawHeight * scale));
  return Object.freeze({
    width: backingWidth,
    height: backingHeight,
    effectiveDevicePixelRatio: Math.min(backingWidth / width, backingHeight / height),
    pixelCount: backingWidth * backingHeight,
  });
};
