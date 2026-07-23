export const empiricalSourceClasses = [
  'Primary observation',
  'Collaboration analysis',
  'Scientific standard',
  'Theoretical research',
  'Official scientific summary',
] as const;

export type EmpiricalSourceClass = (typeof empiricalSourceClasses)[number];

export interface EmpiricalSource {
  readonly id: number;
  readonly authors: string;
  readonly title: string;
  readonly organization: string;
  readonly year: number | 'n.d.';
  readonly url: string;
  readonly identifier?: string;
  readonly sourceClass: EmpiricalSourceClass;
  readonly limitation: string;
}

export const cosmicBreathEmpiricalSources: readonly EmpiricalSource[] = Object.freeze([
  {
    id: 1,
    authors: 'Adam G. Riess et al. (High-Z Supernova Search Team)',
    title:
      'Observational Evidence from Supernovae for an Accelerating Universe and a Cosmological Constant',
    organization: 'The Astronomical Journal 116, 1009',
    year: 1998,
    url: 'https://doi.org/10.1086/300499',
    identifier: 'DOI 10.1086/300499',
    sourceClass: 'Primary observation',
    limitation:
      'The inference depends on supernova calibration, systematic controls, and cosmological modeling.',
  },
  {
    id: 2,
    authors: 'S. Perlmutter et al. (Supernova Cosmology Project)',
    title: 'Measurements of Ω and Λ from 42 High-Redshift Supernovae',
    organization: 'The Astrophysical Journal 517, 565',
    year: 1999,
    url: 'https://doi.org/10.1086/307221',
    identifier: 'DOI 10.1086/307221',
    sourceClass: 'Primary observation',
    limitation:
      'The parameter constraints are conditional on the fitted cosmological models and supernova systematics.',
  },
  {
    id: 3,
    authors: 'Planck Collaboration',
    title: 'Planck 2018 Results. VI. Cosmological Parameters',
    organization: 'Astronomy & Astrophysics 641, A6',
    year: 2020,
    url: 'https://doi.org/10.1051/0004-6361/201833910',
    identifier: 'DOI 10.1051/0004-6361/201833910',
    sourceClass: 'Collaboration analysis',
    limitation:
      'The precise age and parameter estimates are conditional on base ΛCDM and the selected data combinations.',
  },
  {
    id: 4,
    authors: 'DESI Collaboration',
    title:
      'DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints',
    organization: 'Physical Review D 112, 083515',
    year: 2025,
    url: 'https://doi.org/10.1103/tr6y-kpc6',
    identifier: 'DOI 10.1103/tr6y-kpc6',
    sourceClass: 'Collaboration analysis',
    limitation:
      'The preference for evolving dark energy varies with dataset and parameterization and is not a settled discovery.',
  },
  {
    id: 5,
    authors: 'European Space Agency',
    title: 'The Light and Dark Universe',
    organization: 'ESA Euclid',
    year: 2023,
    url: 'https://www.esa.int/ESA_Multimedia/Images/2023/05/The_light_and_dark_Universe',
    sourceClass: 'Official scientific summary',
    limitation:
      'This mission overview describes competing interpretations but does not select a physical identity for dark energy.',
  },
  {
    id: 6,
    authors: 'National Aeronautics and Space Administration',
    title: 'Expand Our Knowledge of Dark Energy',
    organization: 'NASA Physics of the Cosmos',
    year: 2025,
    url: 'https://science.nasa.gov/astrophysics/programs/physics-of-the-cosmos/expand-our-knowledge-of-dark-energy/',
    sourceClass: 'Official scientific summary',
    limitation:
      'This is a scientific-program summary rather than a cosmological parameter-analysis paper.',
  },
  {
    id: 7,
    authors: 'Paul J. Steinhardt and Neil Turok',
    title: 'Cosmic Evolution in a Cyclic Universe',
    organization: 'Physical Review D 65, 126003',
    year: 2002,
    url: 'https://doi.org/10.1103/PhysRevD.65.126003',
    identifier: 'DOI 10.1103/PhysRevD.65.126003',
    sourceClass: 'Theoretical research',
    limitation:
      'The paper develops a particular cyclic construction; it does not observationally establish that our universe cycles.',
  },
  {
    id: 8,
    authors: 'Petar Pavlović and Marko Sossich',
    title: 'Dynamic Properties of Cyclic Cosmologies',
    organization: 'Physical Review D 103, 023529',
    year: 2021,
    url: 'https://doi.org/10.1103/PhysRevD.103.023529',
    identifier: 'DOI 10.1103/PhysRevD.103.023529',
    sourceClass: 'Theoretical research',
    limitation:
      'Mathematical analysis of cyclic solutions is not empirical confirmation of a cosmic cycle.',
  },
  {
    id: 9,
    authors: 'Bureau International des Poids et Mesures',
    title: 'SI Base Unit: Second (s)',
    organization: 'BIPM',
    year: 2026,
    url: 'https://www.bipm.org/en/si-base-units/second',
    sourceClass: 'Scientific standard',
    limitation:
      'The SI definition establishes a measurement unit, not a smallest physical interval.',
  },
  {
    id: 10,
    authors: 'Peter J. Mohr, David B. Newell, Barry N. Taylor, and Eite Tiesinga',
    title: 'CODATA Recommended Values of the Fundamental Physical Constants: 2022',
    organization: 'Reviews of Modern Physics 97, 025002',
    year: 2025,
    url: 'https://doi.org/10.1103/RevModPhys.97.025002',
    identifier: 'DOI 10.1103/RevModPhys.97.025002',
    sourceClass: 'Scientific standard',
    limitation:
      'Planck time is a derived natural unit, not an experimentally established minimum interval.',
  },
  {
    id: 11,
    authors: 'Donald E. Knuth',
    title: 'Mathematics and Computer Science: Coping with Finiteness',
    organization: 'Science 194, 1235–1242',
    year: 1976,
    url: 'https://doi.org/10.1126/science.194.4271.1235',
    identifier: 'DOI 10.1126/science.194.4271.1235',
    sourceClass: 'Theoretical research',
    limitation:
      'Up-arrow notation defines mathematical operations; it does not assign physical units or durations.',
  },
  {
    id: 12,
    authors: 'John A. Wheeler',
    title: 'On the Nature of Quantum Geometrodynamics',
    organization: 'Annals of Physics 2, 604–614',
    year: 1957,
    url: 'https://doi.org/10.1016/0003-4916(57)90050-7',
    identifier: 'DOI 10.1016/0003-4916(57)90050-7',
    sourceClass: 'Theoretical research',
    limitation:
      'Planck-scale geometric fluctuations remain theoretical rather than a direct observation of quantum foam.',
  },
  {
    id: 13,
    authors: 'National Aeronautics and Space Administration',
    title: 'NASA Telescopes Set Limits on Spacetime Quantum Foam',
    organization: 'NASA',
    year: 2015,
    url: 'https://www.nasa.gov/image-article/nasa-telescopes-set-limits-spacetime-quantum-foam/',
    sourceClass: 'Official scientific summary',
    limitation:
      'Observational limits on proposed effects are constraints, not a detection of quantum foam.',
  },
  {
    id: 14,
    authors: 'Fred C. Adams and Gregory Laughlin',
    title: 'A Dying Universe: The Long-Term Fate and Evolution of Astrophysical Objects',
    organization: 'Reviews of Modern Physics 69, 337–372',
    year: 1997,
    url: 'https://doi.org/10.1103/RevModPhys.69.337',
    identifier: 'DOI 10.1103/RevModPhys.69.337',
    sourceClass: 'Theoretical research',
    limitation:
      'Long-term projections, including heat-death-like outcomes, depend on specified physical and cosmological assumptions.',
  },
  {
    id: 15,
    authors: 'Raphael Bousso',
    title: 'The Holographic Principle',
    organization: 'Reviews of Modern Physics 74, 825–874',
    year: 2002,
    url: 'https://doi.org/10.1103/RevModPhys.74.825',
    identifier: 'DOI 10.1103/RevModPhys.74.825',
    sourceClass: 'Theoretical research',
    limitation:
      'The framework does not establish that the observed universe is literally a hologram or that consciousness is holographic.',
  },
  {
    id: 16,
    authors: 'Juan M. Maldacena',
    title: 'The Large N Limit of Superconformal Field Theories and Supergravity',
    organization: 'Advances in Theoretical and Mathematical Physics 2, 231–252',
    year: 1998,
    url: 'https://doi.org/10.4310/ATMP.1998.v2.n2.a1',
    identifier: 'DOI 10.4310/ATMP.1998.v2.n2.a1',
    sourceClass: 'Theoretical research',
    limitation:
      'The proposed duality applies in particular theoretical settings and is not direct empirical proof about our cosmology.',
  },
  {
    id: 17,
    authors: 'Stephen W. Hawking',
    title: 'Particle Creation by Black Holes',
    organization: 'Communications in Mathematical Physics 43, 199–220',
    year: 1975,
    url: 'https://doi.org/10.1007/BF02345020',
    identifier: 'DOI 10.1007/BF02345020',
    sourceClass: 'Theoretical research',
    limitation:
      'Astrophysical Hawking radiation has not been directly detected.',
  },
  {
    id: 18,
    authors: 'Don N. Page',
    title: 'Particle Emission Rates from a Black Hole. II. Massless Particles from a Rotating Hole',
    organization: 'Physical Review D 14, 3260',
    year: 1976,
    url: 'https://doi.org/10.1103/PhysRevD.14.3260',
    identifier: 'DOI 10.1103/PhysRevD.14.3260',
    sourceClass: 'Theoretical research',
    limitation:
      'Emission rates and evaporation times depend on black-hole properties and particle content.',
  },
  {
    id: 19,
    authors: 'National Aeronautics and Space Administration',
    title: 'How Big Is Space? We Asked a NASA Expert: Episode 61',
    organization: 'NASA',
    year: 2025,
    url: 'https://www.nasa.gov/science-research/astrophysics/how-big-is-space-we-asked-a-nasa-expert-episode-61/',
    sourceClass: 'Official scientific summary',
    limitation:
      'The approximate observable diameter is not the size of the entire universe and is not interchangeable with cosmic age.',
  },
]);
