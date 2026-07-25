import type Decimal from 'decimal.js';

export type Era = 'CE' | 'BCE';

export interface UtcTimeInput {
  hour: number;
  minute: number;
  second: number;
}

export interface CivilGregorianInput {
  civilYear: string;
  era: Era;
  month: number;
  day: number;
  time?: Partial<UtcTimeInput> | null;
}

export interface CanonicalGregorianUtc {
  astronomicalYear: Decimal;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
}

export interface CivilGregorianUtc extends UtcTimeInput {
  civilYear: string;
  era: Era;
  month: number;
  day: number;
}

export type ObservableAgeState =
  | { kind: 'elapsed'; years: Decimal }
  | { kind: 'big-bang-boundary' }
  | { kind: 'pre-big-bang-reference-interval'; intervalYears: Decimal };

export type CoordinateToObservableAgeRatioState =
  | { kind: 'available'; value: Decimal }
  | {
      kind: 'not-applicable';
      reason: 'big-bang-boundary' | 'pre-big-bang-reference-interval';
    };

export interface ObservableUniverseReferenceResult {
  observableUniverseAlignedCuCoordinate: Decimal;
  observableAge: ObservableAgeState;
  coordinateToObservableAgeRatio: CoordinateToObservableAgeRatioState;
}

export interface ForwardConversionResult {
  jdn: Decimal;
  deltaJdn: Decimal;
  deltaYears: Decimal;
  nasaCuTime: Decimal;
  cuTime: Decimal;
  yearsSinceBigBang: Decimal;
  observableUniverseReference: ObservableUniverseReferenceResult;
}

export interface ReverseConversionResult extends CivilGregorianUtc {
  inputCuTime: Decimal;
  nasaCuTime: Decimal;
  deltaYears: Decimal;
  deltaJdn: Decimal;
  jdn: Decimal;
  yearsSinceBigBang: Decimal;
  observableUniverseReference: ObservableUniverseReferenceResult;
}

export interface ParsedCuTimeInput {
  raw: string;
  lexical: string;
  value: Decimal;
}
