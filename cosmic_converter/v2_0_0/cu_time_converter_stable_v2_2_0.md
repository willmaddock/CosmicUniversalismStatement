# Cosmic Universalism Time Converter v2.2.0 - Deprecated

**Status:** Archived  
**Date:** May 28, 2025

This version (v2.2.0) of the Cosmic Universalism Time Converter is no longer supported due to a faulty `BASE_CU` calibration (`3094134044923.672659`). It has been replaced by v3.0.0, which uses a recalibrated `BASE_CU` (`3094134044923.509753564772922302508810`) for accurate CU-Time calculations and includes enhanced features like responsive navigation.

Please use the updated version:

[**Go to v3.0.0**](https://willmaddock.github.io/CosmicUniversalismStatement/cosmic_converter/v3_0_0/cu_time_converter_stable_v3_0_0.html)

For more details, visit the [CosmicUniversalismStatement GitHub Repository](https://github.com/willmaddock/CosmicUniversalismStatement).

---

# This code is faulty is not calibrared correctly.

```  
*Note*: This file archives the v2.2.0 version. The original HTML content is no longer served as a web app. To access the v3.0.0 release notes and instructions, see the `cosmic_converter/v3_0_0/v3_Guides` directory.

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cosmic Universalism Time Converter v2.2.0</title>
  <script src="https://cdn.jsdelivr.net/npm/react@18.3.1/umd/react.production.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/react-dom@18.3.1/umd/react-dom.production.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.25.7/babel.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/luxon@3.5.0/build/global/luxon.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/decimal.js@10.4.3/decimal.min.js"></script>
  <link rel="icon" type="image/x-icon" href="/favicon.ico">
  <style>
    body {
      background-image: url('https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0') || url('path/to/local-image.jpg');
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
    }
    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.5);
      z-index: -1;
    }
    .error-boundary { color: red; padding: 20px; }
    .lexicon-container { max-height: 400px; overflow-y: auto; }
    .gradient-header { background: linear-gradient(to right, #1e3a8a, #3b82f6); }
    .custom-button {
      transition: all 0.3s ease;
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .custom-button:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .card {
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .delete-button {
      background-color: #dc2626;
      color: white;
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      transition: background-color 0.2s;
    }
    .delete-button:hover {
      background-color: #b91c1c;
    }
    .result-box {
      background-color: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      padding: 1rem;
      margin-bottom: 1rem;
    }
    .result-input {
      width: 100%;
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      margin-bottom: 0.5rem;
    }
    .result-textarea {
      width: 100%;
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 0.375rem;
      background-color: #f3f4f6;
      resize: none;
    }
  </style>
</head>
<body class="bg-gray-900 text-white font-sans">
  <div class="overlay"></div>
  <div id="root"></div>
  <script type="text/babel" data-presets="react,es2015">
    const { useState, useEffect } = React;
    const { DateTime } = luxon;

    // Configure Decimal.js
    Decimal.set({ precision: 60, rounding: Decimal.ROUND_HALF_UP });

    // Constants
    const CONSTANTS = {
      BASE_CU: new Decimal('3094134044923.672659'),
      SUB_ZTOM_CU: new Decimal('3094133911800.949548'),
      ANCHOR_JDN: new Decimal('2451604.5'),
      DAYS_PER_YEAR: new Decimal('365.2425'),
      SECONDS_PER_DAY: new Decimal('86400'),
      SECONDS_PER_YEAR: new Decimal('31556952'),
      COSMIC_LIFESPAN: new Decimal('13.8e9'),
      CONVERGENCE_YEAR: new Decimal('2029'),
      CTOM_START: new Decimal('3.08e12'),
      ZTOM_PROXIMITY: new Decimal('3.107e12'),
      PLANCK_TIME: new Decimal('5.39e-44'),
      CESIUM_OSCILLATIONS_PER_SECOND: new Decimal('9192631770'),
      SUB_ZTOM_THRESHOLD: new Decimal('1'),
      ZTOM_THRESHOLD: new Decimal('31556952000000'),
      DARK_ENERGY_START: new Decimal('3.08e12'),
      DARK_ENERGY_END: new Decimal('3.11e12'),
      ANTI_DARK_ENERGY_START: new Decimal('3.11e12'),
      ANTI_DARK_ENERGY_END: new Decimal('3.416e12'),
      SPECULATIVE_START: new Decimal('3.416e12')
    };

    // CU Lexicon
    const CU_LEXICON = {
      "sub-ztom": "5.39e-44 sec (Planck time, quantum core)",
      "sub-ytom": "2.704e-8 sec (Recursive core shell)",
      "sub-xtom": "2.704e-7 sec (Final ethical cap)",
      "sub-wtom": "2.704e-6 sec (Quantum firewall shell)",
      "sub-vtom": "2.704e-5 sec (Symbolic/ethical lock)",
      "sub-utom": "0.0002704 sec (Cosmic checksum layer)",
      "sub-ttom": "0.002704 sec (Holographic verification)",
      "sub-stom": "0.02704 sec (Ethical engagement)",
      "sub-rtom": "0.2704 sec (Entropy modulation)",
      "sub-qtom": "2.704 sec (Memory transfer)",
      "sub-ptom": "27.04 sec (Pre-reset bridge)",
      "sub-otom": "4.506 min (Boundary stabilization)",
      "sub-ntom": "45.06 min (Recursive feedback)",
      "sub mtom": "7.51 hr (Holographic projection)",
      "sub-ltom": "3.1296 day (Pre-Big Bang state)",
      "sub-ktom": "31.296 day (Quantum foam rebirth)",
      "sub-jtom": "0.8547 yr (Black hole age)",
      "sub-itom": "8.547 yr (Spacetime contraction begins)",
      "sub-htom": "85.47 yr (Heat death begins)",
      "sub-gtom": "427.35 yr (Quantum encoding phase)",
      "sub-ftom": "4273.5 yr (Post-biological AI expansion)",
      "sub-etom": "42735 yr (Alien/civilization stage)",
      "sub-dtom": "427350 yr (Planetary biosphere evolution)",
      "sub-ctom": "28 billion yr (Star life cycle era)",
      "sub-btom": "280 billion yr (Supercluster formation)",
      "atom": "2.8 trillion yr (Start of compression)",
      "btom": "280 billion yr (Galactic evolution and contraction)",
      "ctom": "28 billion yr (Final stellar formations)",
      "dtom": "427350 yr (Planetary collapse)",
      "etom": "42735 yr (Human/civilization memory condensation)",
      "ftom": "4273.5 yr (AI implosion stage)",
      "gtom": "427.35 yr (Consciousness holography)",
      "htom": "85.47 yr (Heat death approach)",
      "itom": "8.547 yr (Spacetime wrinkle forming)",
      "jtom": "0.8547 yr (Collapse threshold)",
      "ktom": "31.296 day (Quantum fog closing)",
      "ltom": "3.1296 day (Holographic reversal)",
      "mtom": "7.51 hr (Time lattice inversion)",
      "ntom": "45.06 min (Feedback end)",
      "otom": "4.506 min (Cosmic null stabilization)",
      "ptom": "27.04 sec (Pre-reset bridge)",
      "qtom": "2.704 sec (Final memory imprint)",
      "rtom": "0.2704 sec (Entropy zero point)",
      "stom": "0.02704 sec (Ethical firewall gate)",
      "ttom": "0.002704 sec (Collapse checksum)",
      "utom": "0.0002704 sec (Closure sequence initiated)",
      "vtom": "2.704e-5 sec (Symbolic compression)",
      "wtom": "2.704e-6 sec (Recursive limit breach)",
      "xtom": "2.704e-7 sec (Divine fall-off shell)",
      "ytom": "2.704e-8 sec (Pre-ZTOM divine echo)",
      "ztom": "1 sec (ZTOM: full universal reset, Empowered by God’s Free Will)",
      "matter": "280 billion yr (Galactic formation, matter dominance)",
      "anti-dark-matter": "427350 yr (Compression onset, anti-dark matter influence)",
      "matter-antimatter": "5.39e-44 sec (Quantum annihilation, sub-ztom)",
      "quantum-states": "Uncountable infinite states guiding cosmic evolution, empowered by God’s Free Will"
    };

    // Geological Epochs
    const BOUNDARY_YEARS = [
      -50000000000, -28000000000, -541000000, -485400000, -443800000,
      -419200000, -358900000, -298900000, -251900000, -201300000,
      -145000000, -66000000, -56000000, -33900000, -23030000,
      -5333000, -2580000, -11700, -275, -75, 2025
    ];
    const EPOCH_NAMES = [
      "Speculative Pre-Cosmic Phase", "Pre-Cambrian or Cosmic Phase", "Cambrian",
      "Ordovician", "Silurian", "Devonian", "Carboniferous", "Permian", "Triassic",
      "Jurassic", "Cretaceous", "Paleocene", "Eocene", "Oligocene", "Miocene",
      "Pliocene", "Pleistocene", "Holocene", "Industrial Revolution", "Anthropocene", "Present"
    ];
    const CU_BOUNDARIES = BOUNDARY_YEARS.map(year => {
      if (year === -28000000000) return CONSTANTS.SUB_ZTOM_CU;
      return computeCUTime(year);
    });

    // Historical Events
    const HISTORICAL_EVENTS = {
      "Moon Landing": {
        date: "1969-07-20T20:17:00+00:00",
        cu_time: new Decimal('3094134044948.672659'),
        description: "First human landing on the Moon (Apollo 11)"
      },
      "Magna Carta": {
        date: "1215-06-15T00:00:00+00:00",
        cu_time: new Decimal('3094134043789.949548'),
        description: "Signing of the Magna Carta in England"
      },
      "Supernova 1054": {
        date: "1054-07-04T00:00:00+00:00",
        cu_time: new Decimal('3094134043628.949548'),
        description: "Supernova observed, forming the Crab Nebula"
      }
    };

    // In-memory history
    let conversionHistory = [];

    // Error Boundary Component
    class ErrorBoundary extends React.Component {
      state = { hasError: false, error: null };
      static getDerivedStateFromError(error) {
        return { hasError: true, error };
      }
      render() {
        if (this.state.hasError) {
          return (
            <div className="error-boundary">
              <h2>Application Error</h2>
              <p>{this.state.error?.message || 'An unknown error occurred.'}</p>
            </div>
          );
        }
        return this.props.children;
      }
    }

    // Helper Functions
    function countLeapYears(startYear, endYear) {
      if (Math.abs(startYear) > 28000000000 || Math.abs(endYear) > 28000000000) {
        const start = Math.min(startYear, endYear);
        const end = Math.max(startYear, endYear);
        const years = end - start;
        return Math.floor(years / 4);
      }
      let start = startYear < 0 ? startYear + 1 : startYear;
      let end = endYear < 0 ? endYear + 1 : endYear;
      if (start > end) [start, end] = [end, start];
      let leapYears = Math.floor(end / 4) - Math.floor(start / 4);
      leapYears -= Math.floor(end / 100) - Math.floor(start / 100);
      leapYears += Math.floor(end / 400) - Math.floor(start / 400);
      if (start % 4 === 0 && (start % 100 !== 0 || start % 400 === 0)) {
        leapYears += 1;
      }
      return Math.max(0, leapYears);
    }

    function getJDN(year, month, day, hour = 0, minute = 0, second = 0, microsecond = 0) {
      try {
        const cacheKey = `${year},${month},${day},${hour},${minute},${second},${microsecond}`;
        if (window.jdnCache?.[cacheKey]) return window.jdnCache[cacheKey];

        let jdn;
        if (Math.abs(year) > 999999999) {
          const deltaYears = new Decimal(year).minus(2000);
          const days = deltaYears.times(CONSTANTS.DAYS_PER_YEAR);
          const fraction = new Decimal(hour).div(24)
            .plus(new Decimal(minute).div(1440))
            .plus(new Decimal(second).div(86400))
            .plus(new Decimal(microsecond).div(1000000).div(86400));
          jdn = CONSTANTS.ANCHOR_JDN.plus(days).plus(fraction);
        } else {
          const a = Math.floor((14 - month) / 12);
          const y = year + 4800 - a;
          const m = month + 12 * a - 3;
          const jdInt = Math.floor(day + Math.floor((153 * m + 2) / 5) + 365 * y + Math.floor(y / 4) - Math.floor(y / 100) + Math.floor(y / 400) - 32045);
          const fraction = new Decimal(hour).div(24)
            .plus(new Decimal(minute).div(1440))
            .plus(new Decimal(second).div(86400))
            .plus(new Decimal(microsecond).div(1000000).div(86400));
          jdn = new Decimal(jdInt).plus(fraction);
        }
        window.jdnCache = window.jdnCache || {};
        window.jdnCache[cacheKey] = jdn;
        return jdn;
      } catch (e) {
        console.error('getJDN error:', e);
        throw e;
      }
    }

    function computeCUTime(year, month = 1, day = 1, hour = 0, minute = 0, second = 0, microsecond = 0) {
      try {
        const cacheKey = `${year},${month},${day},${hour},${minute},${second},${microsecond}`;
        if (window.cuTimeCache?.[cacheKey]) return window.cuTimeCache[cacheKey];

        if (year === -28000000000) {
          window.cuTimeCache = window.cuTimeCache || {};
          window.cuTimeCache[cacheKey] = CONSTANTS.SUB_ZTOM_CU;
          return CONSTANTS.SUB_ZTOM_CU;
        }

        let cuTime;
        if (Math.abs(year) > 28000000000) {
          const deltaYears = new Decimal(year).minus(2000);
          cuTime = CONSTANTS.BASE_CU.plus(deltaYears);
        } else {
          const jdn = getJDN(year, month, day, hour, minute, second, microsecond);
          const deltaJDN = jdn.minus(CONSTANTS.ANCHOR_JDN);
          const deltaYears = deltaJDN.div(CONSTANTS.DAYS_PER_YEAR);
          cuTime = CONSTANTS.BASE_CU.plus(deltaYears);
        }

        cuTime = cuTime.toDecimalPlaces(24, Decimal.ROUND_HALF_UP);
        window.cuTimeCache = window.cuTimeCache || {};
        window.cuTimeCache[cacheKey] = cuTime;
        return cuTime;
      } catch (e) {
        console.error('computeCUTime error:', e);
        throw e;
      }
    }

    function jdnToDate(jdn) {
      try {
        const jdInt = jdn.floor();
        const fraction = jdn.minus(jdInt);
        if (jdn.minus(1721424.5).abs().div(CONSTANTS.DAYS_PER_YEAR).gt(28000000000)) {
          const year = jdn.minus(1721424.5).div(CONSTANTS.DAYS_PER_YEAR).round().toNumber();
          return { year: `January 1, ${year}`, month: 1, day: 1, hour: 0, minute: 0, second: 0, microsecond: 0 };
        }

        const z = jdInt.plus(0.5);
        const f = z.minus(z.floor());
        const alpha = z.minus(1867216.25).div(36524.25).floor();
        const a = z.plus(1).plus(alpha).minus(alpha.div(4).floor());
        const b = a.plus(1524);
        const c = b.minus(122.1).div(365.25).floor();
        const d = c.times(365.25).floor();
        const e = b.minus(d).div(30.6001).floor();
        const day = b.minus(d).minus(e.times(30.6001).floor()).toNumber();
        const month = e.lt(14) ? e.minus(1) : e.minus(13);
        const year = month.gt(2) ? c.minus(4716) : c.minus(4715);
        const hours = fraction.times(24).floor().toNumber();
        const minutes = fraction.times(24).minus(hours).times(60).floor().toNumber();
        const seconds = fraction.times(24).minus(hours).times(60).minus(minutes).times(60).floor().toNumber();
        const microseconds = fraction.times(24).minus(hours).times(60).minus(minutes).times(60).minus(seconds).times(1000000).floor().toNumber();
        return { year: year.toNumber(), month: month.toNumber(), day, hour: hours, minute: minutes, second: seconds, microsecond: microseconds };
      } catch (e) {
        console.error('jdnToDate error:', e);
        throw e;
      }
    }

    function parseJulianDate(dateStr) {
      try {
        const parts = dateStr.split('/');
        if (parts.length !== 3) throw new Error('Invalid Julian date format');
        const day = parseInt(parts[0]);
        const month = parseInt(parts[1]);
        const year = parseInt(parts[2]);
        const a = Math.floor((14 - month) / 12);
        const y = year + 4800 - a;
        const m = month + 12 * a - 3;
        const jd = new Decimal(day).plus(
          new Decimal((153 * m + 2) / 5).floor()
        ).plus(
          new Decimal(365).times(y)
        ).plus(
          new Decimal(y / 4).floor()
        ).minus(32083);
        return jdnToDate(jd);
      } catch (e) {
        console.error('parseJulianDate error:', e);
        throw e;
      }
    }

    function getCosmicPhase(cuTime) {
      try {
        if (cuTime.lt(CONSTANTS.DARK_ENERGY_START)) {
          return "Speculative Phase (Pre-expansion, theoretical)";
        } else if (cuTime.gte(CONSTANTS.DARK_ENERGY_START) && cuTime.lte(CONSTANTS.DARK_ENERGY_END)) {
          return "Dark Energy Phase (Expansion, sub-ztom to atom)";
        } else if (cuTime.gte(CONSTANTS.ANTI_DARK_ENERGY_START) && cuTime.lte(CONSTANTS.ANTI_DARK_ENERGY_END)) {
          return "Anti-Dark Energy Phase (Compression, btom to ztom, Empowered by God’s Free Will)";
        } else {
          return "Speculative Phase (Beyond known cosmic phases)";
        }
      } catch (e) {
        console.error('getCosmicPhase error:', e);
        return "Unknown Phase";
      }
    }

    function getDominantForces(cuTime) {
      try {
        const phase = getCosmicPhase(cuTime);
        if (phase.includes("Anti-Dark Energy Phase")) {
          return ["matter-antimatter"];
        } else if (phase.includes("Dark Energy Phase")) {
          return ["matter"];
        } else {
          return ["anti-dark-matter (theoretical)"];
        }
      } catch (e) {
        console.error('getDominantForces error:', e);
        return ["unknown"];
      }
    }

    function getGeologicalEpoch(cuTime) {
      try {
        for (let i = 0; i < CU_BOUNDARIES.length - 1; i++) {
          if (cuTime.gte(CU_BOUNDARIES[i]) && cuTime.lt(CU_BOUNDARIES[i + 1])) {
            return EPOCH_NAMES[i];
          }
        }
        return EPOCH_NAMES[EPOCH_NAMES.length - 1];
      } catch (e) {
        console.error('getGeologicalEpoch error:', e);
        return "Unknown Epoch";
      }
    }

    function getHistoricalEvents(year, month, day, hour, minute, second) {
      try {
        const inputDt = DateTime.fromObject({ year, month, day, hour, minute, second }, { zone: 'UTC' });
        const events = [];
        for (const [name, event] of Object.entries(HISTORICAL_EVENTS)) {
          const eventDt = DateTime.fromISO(event.date);
          const delta = Math.abs(inputDt.diff(eventDt).as('seconds'));
          if (delta <= 86400) {
            events.push(`${name}: ${event.description} (CU-Time: ${event.cu_time.toString()})`);
          }
        }
        return events;
      } catch (e) {
        console.error('getHistoricalEvents error:', e);
        return [];
      }
    }

    function parseTomDuration(tom) {
      try {
        if (!(tom in CU_LEXICON)) return new Decimal('0');
        const description = CU_LEXICON[tom];
        const match = description.match(/([\d\.e-]+)\s*(sec|min|hr|day|yr)/);
        if (!match) return new Decimal('0');
        const value = new Decimal(match[1]);
        const unit = match[2];
        if (unit === "sec") return value;
        if (unit === "min") return value.times(60);
        if (unit === "hr") return value.times(3600);
        if (unit === "day") return value.times(CONSTANTS.SECONDS_PER_DAY);
        if (unit === "yr") return value.times(CONSTANTS.SECONDS_PER_YEAR);
        return new Decimal('0');
      } catch (e) {
        console.error('parseTomDuration error:', e);
        return new Decimal('0');
      }
    }

    function ethicalClassifier(input) {
      try {
        if (input.toLowerCase().includes('invalid') || input.toLowerCase().includes('error')) {
          return { valid: false, message: "Unethical: Invalid input detected", note: null };
        }
        if (input.length > 1000) {
          return { valid: false, message: "Unethical: Input too long for processing", note: null };
        }
        for (const tom in CU_LEXICON) {
          if (input.toLowerCase().includes(tom) && !new RegExp(`\\b${tom}\\b`, 'i').test(input)) {
            return { valid: false, message: `Unethical: Misuse of CU Lexicon term '${tom}'`, note: null };
          }
        }

        const cuRegex = /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/;
        if (cuRegex.test(input)) {
          try {
            const cuTime = new Decimal(input);
            if (cuTime.lt(0)) {
              return { valid: false, message: "Unethical: CU-Time cannot be negative", note: null };
            }
            if (cuTime.minus('3.094e12').abs().gt('1e15')) {
              return { valid: false, message: "Unethical: CU-Time too far from current era (3.094T)", note: null };
            }
            const note = cuTime.lt('3e12') || cuTime.gt('4e12') ? "Speculative CU-Time; conversion is theoretical due to extreme value" : null;
            if (cuTime.minus(CONSTANTS.ANTI_DARK_ENERGY_END).abs().lt('1e6')) {
              return { valid: true, message: "Ethical: Input aligns with CU principles and is empowered by God’s Free Will", note };
            }
            return { valid: true, message: "Ethical: Input aligns with CU principles", note };
          } catch (e) {
            return { valid: false, message: "Unethical: Invalid input format", note: null };
          }
        }

        try {
          const parsed = parseDateInput(input);
          const year = parsed.year;
          if (year < -28000000000 || year > 28000000000) {
            return { valid: false, message: "Unethical: Date must be between 28 billion BCE and 28 billion CE", note: null };
          }
          const note = year < -10000 || year > 10000 ? "Speculative date; conversion is theoretical due to extreme time range" : null;
          const cuTime = computeCUTime(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second, parsed.microsecond);
          const keyToms = ["ztom", "btom", "ctom"];
          if (keyToms.some(tom => input.toLowerCase().includes(tom)) || cuTime.minus(CONSTANTS.ANTI_DARK_ENERGY_END).abs().lt('1e6')) {
            return { valid: true, message: "Ethical: Input aligns with CU principles and is empowered by God’s Free Will", note };
          }
          return { valid: true, message: "Ethical: Input aligns with CU principles", note };
        } catch (e) {
          return { valid: false, message: `Unethical: Invalid input format: ${e.message}`, note: null };
        }
      } catch (e) {
        console.error('ethicalClassifier error:', e);
        return { valid: false, message: "Unethical: Internal error", note: null };
      }
    }

    function parseDateInput(input) {
      try {
        input = input.trim();

        // Julian calendar input
        if (input.toLowerCase().startsWith('julian:')) {
          const dateStr = input.split(':')[1].trim();
          return parseJulianDate(dateStr);
        }

        // BCE regex with AM/PM support
        const bcePattern = /^(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{1,11})\s*BCE\s*(?:(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?))?\s*(AM|PM)?(?:\s*([A-Za-z][A-Za-z\/]+|[+-]\d{2}:\d{2}|Z|UTC))?$/i;
        const bceMatch = input.match(bcePattern);
        if (bceMatch) {
          const year = -parseInt(bceMatch[3]);
          const month = parseInt(bceMatch[1]);
          const day = parseInt(bceMatch[2]);
          let hour = parseInt(bceMatch[4] || 0);
          const minute = parseInt(bceMatch[5] || 0);
          const secondStr = bceMatch[6] || "0";
          const ampm = bceMatch[7]?.toUpperCase();
          const microsecond = secondStr.includes('.') ? parseInt(parseFloat(`0.${secondStr.split('.')[1]}`) * 1000000) : 0;
          const second = parseInt(secondStr.split('.')[0]);
          if (ampm) {
            if (ampm === 'PM' && hour < 12) hour += 12;
            if (ampm === 'AM' && hour === 12) hour = 0;
          }
          return { year, month, day, hour, minute, second, microsecond };
        }

        // Gregorian regex with AM/PM support
        const datePattern = /^(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4,11})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?\s*(AM|PM)?(?:\s*([A-Za-z][A-Za-z\/]+|[+-]\d{2}:\d{2}|Z|UTC))?$/i;
        const dateMatch = input.match(datePattern);
        if (dateMatch) {
          const month = parseInt(dateMatch[1]);
          const day = parseInt(dateMatch[2]);
          const year = parseInt(dateMatch[3]);
          let hour = parseInt(dateMatch[4] || 0);
          const minute = parseInt(dateMatch[5] || 0);
          const secondStr = dateMatch[6] || "0";
          const ampm = dateMatch[7]?.toUpperCase();
          const microsecond = secondStr.includes('.') ? parseInt(parseFloat(`0.${secondStr.split('.')[1]}`) * 1000000) : 0;
          const second = parseInt(secondStr.split('.')[0]);
          if (ampm) {
            if (ampm === 'PM' && hour < 12) hour += 12;
            if (ampm === 'AM' && hour === 12) hour = 0;
          }
          return { year, month, day, hour, minute, second, microsecond };
        }

        // ISO format
        try {
          const dt = DateTime.fromISO(input.replace("Z", "+00:00")).toUTC();
          if (!dt.isValid) throw new Error(`Invalid ISO date: ${dt.invalidReason}`);
          return {
            year: dt.year,
            month: dt.month,
            day: dt.day,
            hour: dt.hour,
            minute: dt.minute,
            second: dt.second,
            microsecond: dt.millisecond * 1000
          };
        } catch (e) {
          throw new Error("Invalid date format. Use MM/DD/YYYY [HH:MM:SS[.ffffff]] [AM/PM], ISO, BCE, or Julian:DD/MM/YYYY format.");
        }
      } catch (e) {
        console.error('parseDateInput error:', e);
        throw e;
      }
    }

    function formatCUValue(cuValue) {
      try {
        cuValue = cuValue.toDecimalPlaces(24, Decimal.ROUND_HALF_UP);
        const fullNumeric = cuValue.toFixed(24);
        const exponential = cuValue.toExponential(6);
        const [integerPart, fractionPart] = fullNumeric.split('.');
        const intVal = parseInt(integerPart);
        const trillion = Math.floor(intVal / 1000000000000);
        const remainder = intVal % 1000000000000;
        const billion = Math.floor(remainder / 1000000000);
        const million = Math.floor((remainder % 1000000000) / 1000000);
        const thousand = Math.floor((remainder % 1000000) / 1000);
        const units = remainder % 1000;
        const parts = [];
        if (trillion) parts.push(`${trillion} Trillion`);
        if (billion) parts.push(`${billion} Billion`);
        if (million) parts.push(`${million} Million`);
        if (thousand) parts.push(`${thousand} Thousand`);
        if (units || !parts.length) parts.push(`${units}`);
        const humanFriendly = parts.join(' ') + `.${fractionPart} CU-Time`;
        return {
          humanFriendly,
          fullNumeric,
          exponential
        };
      } catch (e) {
        console.error('formatCUValue error:', e);
        return { humanFriendly: 'Error', fullNumeric: 'Error', exponential: 'Error' };
      }
    }

    function parseCSVRow(row) {
      const fields = [];
      let field = '';
      let inQuotes = false;
      for (let i = 0; i < row.length; i++) {
        const char = row[i];
        if (char === '"' && (i === 0 || row[i - 1] !== '\\')) {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          fields.push(field);
          field = '';
        } else {
          field += char;
        }
      }
      fields.push(field);
      return fields.map(s => s.trim().replace(/^"|"$/g, '').replace(/""/g, '"'));
    }

    function downloadCSV(results) {
      try {
        const csvContent = results.map(r => {
          const dateMatch = r.desc.match(/Date: ([^U]+) UTC/);
          const date = dateMatch ? dateMatch[1].trim() : r.desc;
          return `"${r.desc.replace(/"/g, '""')}","${date.replace(/"/g, '""')}","${r.output.replace(/"/g, '""')}"`;
        }).join('\n');
        const timestamp = DateTime.utc().toFormat('yyyy-MM-dd_HH-mm-ss');
        const blob = new Blob([`"Description","DateTime","Output"\n${csvContent}`], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cu_time_results_${timestamp}.csv`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error('downloadCSV error:', e);
      }
    }

    function ConversionHistory() {
      const getHistory = () => conversionHistory;
      const saveConversion = (input, output, type) => {
        if (conversionHistory.some(entry => entry.input === input && entry.type === type)) return;
        conversionHistory.push({
          input,
          output,
          type,
          timestamp: DateTime.utc().toISO(),
          favorite: false
        });
      };
      const markFavorite = (input) => {
        conversionHistory = conversionHistory.map(entry =>
          entry.input === input ? { ...entry, favorite: !entry.favorite } : entry
        );
      };
      const getFavorites = () => conversionHistory.filter(entry => entry.favorite);
      return { getHistory, saveConversion, markFavorite, getFavorites };
    }

    const HISTORY = ConversionHistory();

    function Modal({ message, onConfirm, onCancel }) {
      return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg max-w-sm w-full">
            <p className="mb-4 text-gray-800">{message}</p>
            <div className="flex justify-between">
              <button
                onClick={onConfirm}
                className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-700 custom-button"
              >
                Confirm
              </button>
              {onCancel && (
                <button
                  onClick={onCancel}
                  className="bg-gray-600 text-white p-2 rounded-md hover:bg-gray-700 custom-button"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }

    function saveEditedCSV(results) {
      try {
        const csvContent = results.map(r => {
          const dateMatch = r.desc.match(/Date: ([^U]+) UTC/);
          const date = dateMatch ? dateMatch[1].trim() : r.desc;
          return `"${r.desc.replace(/"/g, '""')}","${date.replace(/"/g, '""')}","${r.output.replace(/"/g, '""')}"`;
        }).join('\n');
        const timestamp = DateTime.utc().toFormat('yyyy-MM-dd_HH-mm-ss');
        const blob = new Blob([`"Description","DateTime","Output"\n${csvContent}`], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cu_time_results_edited_${timestamp}.csv`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error('saveEditedCSV error:', e);
      }
    }

    function App() {
      const [tab, setTab] = useState('gregorian');
      const [input, setInput] = useState({
        date: '',
        time: '',
        ampm: 'AM',
        era: 'CE',
        cuTime: '',
        tom: '',
        date2: '',
        time2: '',
        ampm2: 'AM',
        era2: 'CE'
      });
      const [results, setResults] = useState([]);
      const [history, setHistory] = useState(HISTORY.getHistory());
      const [showFavorites, setShowFavorites] = useState(false);
      const [modal, setModal] = useState(null);
      const [showLexicon, setShowLexicon] = useState(false);
      const [showResults, setShowResults] = useState(true);
      const [editedResults, setEditedResults] = useState([]);

      useEffect(() => {
        window.addEventListener('error', (e) => console.error('Global error:', e));
        return () => window.removeEventListener('error', (e) => console.error('Global error:', e));
      }, []);

      const formatDateInput = (value) => {
        const clean = value.replace(/[^0-9]/g, '');
        let formatted = '';
        if (clean.length > 0) formatted += clean.slice(0, 2);
        if (clean.length > 2) formatted += '/' + clean.slice(2, 4);
        if (clean.length > 4) formatted += '/' + clean.slice(4, 12);
        return formatted;
      };

      const formatTimeInput = (value) => {
        const clean = value.replace(/[^0-9]/g, '');
        let formatted = '';
        if (clean.length > 0) formatted += clean.slice(0, 2);
        if (clean.length > 2) formatted += ':' + clean.slice(2, 4);
        if (clean.length > 4) formatted += ':' + clean.slice(4, 6);
        return formatted;
      };

      const handleDateChange = (e, field = 'date') => {
        const value = e.target.value;
        const formatted = formatDateInput(value);
        setInput({ ...input, [field]: formatted });
      };

      const handleTimeChange = (e, field = 'time') => {
        const value = e.target.value;
        const formatted = formatTimeInput(value);
        setInput({ ...input, [field]: formatted });
      };

      const handleAmPmChange = (value, field = 'ampm') => {
        setInput({ ...input, [field]: value });
      };

      const validateTime = (time) => {
        const timePattern = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/;
        const match = time.match(timePattern);
        if (!match) return false;
        const hours = parseInt(match[1]);
        const minutes = parseInt(match[2]);
        const seconds = match[3] ? parseInt(match[3]) : 0;
        return hours >= 1 && hours <= 12 && minutes >= 0 && minutes <= 59 && seconds >= 0 && seconds <= 59;
      };

      const handleDateSubmit = () => {
        try {
          let fullInput = `${input.date}${input.era === 'BCE' ? ' BCE' : ''} ${input.time || '00:00:01'} ${input.ampm}`;
          if (!input.date.includes('/')) {
            setModal({
              message: 'Invalid date format in MM/DD/YYYY format.',
              onConfirm: () => setModal(null),
              onCancel: () => setModal(null)
            });
            return;
          }
          if (!input.era) {
            setModal({
              message: 'Specify if the date is BCE or CE.',
              onConfirm: () => setInput({ ...input, era: 'CE' }),
              onCancel: () => setInput({ ...input, era: 'BCE' })
            });
            return;
          }
          if (!input.time) {
            setModal({
              message: 'No time provided. Use 00:00:01 AM?',
              onConfirm: () => {
                setInput({ ...input, time: '00:00:01', ampm: 'AM' });
                setModal(null);
                submitDateWithInput(`${input.date}${input.era === 'BCE' ? ' BCE' : ''} 00:00:01 AM`);
              },
              onCancel: () => setModal(null)
            });
            return;
          }
          if (!validateTime(input.time)) {
            setModal({
              message: 'Invalid time format in HH:MM:SS format (hours 1-12, minutes/seconds 0-59).',
              onConfirm: () => setModal(null),
              onCancel: () => setModal(null)
            });
            return;
          }
          submitDateWithInput(fullInput);
        } catch (e) {
          console.error('submit error:', e);
          setResults(prev => [...prev, { desc: 'Error', output: e.message }]);
        }
      };

      const submitDateWithInput = (fullInput) => {
        try {
          const { valid, message, note } = ethicalClassifier(fullInput);
          if (!valid) {
            setResults(prev => [...prev, { desc: 'Error', output: message }]);
            return;
          }

          const parsed = parseDateInput(fullInput);
          const cuTime = computeCUTime(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second, parsed.microsecond);
          const formatted = formatCUValue(cuTime);
          const cosmicPhase = getCosmicPhase(cuTime);
          const dominantForces = getDominantForces(cuTime);
          const geologicalEpoch = getGeologicalEpoch(cuTime);
          const historicalEvents = getHistoricalEvents(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second);
          const refCuTime = computeCUTime(2025, 5, 26, 9, 48, 0);
          const timeDiff = cuTime.minus(refCuTime).abs().toDecimalPlaces(24);
          const yearsDiff = Math.abs(parsed.year - 2025);
          let leapYearsStr = '';
          if (typeof parsed.year === 'number' && Math.abs(parsed.year) <= 28000000000) {
            const leapYears = countLeapYears(-28000000000, parsed.year);
            leapYearsStr = `Leap Years from Sub-ZTOM (28000000000 BCE): ${leapYears}`;
          } else {
            leapYearsStr = 'Leap Years: N/A (Year out of range)';
          }
          const output = [
            `Input: ${fullInput} UTC`,
            `CU-Time: ${formatted.humanFriendly}`,
            `Full Numeric: ${formatted.fullNumeric} CU-Time`,
            `Exponential: ${formatted.exponential}`,
            `Epoch/Period: ${cuTime.lt(CONSTANTS.BASE_CU) ? 'Past' : 'Future'}`,
            `Geological Epoch: ${geologicalEpoch}`,
            `Cosmic Phase: ${cosmicPhase}`,
            `Dominant Forces: ${dominantForces.join(', ')}`,
            leapYearsStr,
            historicalEvents.map(event => `Historical Event: ${event}`),
            `Time Difference from 2025-05-26T09:48:00+00:00: ${timeDiff} CU-Time (~${yearsDiff} years)`,
            `Ethical Status: ${message}`,
            note ? `Note: ${note}` : ''
          ].filter(line => line);

          setResults(prev => [...prev, { desc: `Date: ${fullInput} UTC`, output: output.join('\n') }]);
          setEditedResults(prev => [...prev, { desc: `Date: ${fullInput} UTC`, output: output.join('\n') }]);
          HISTORY.saveConversion(fullInput, formatted.humanFriendly, 'Gregorian to CU');
          setHistory(HISTORY.getHistory());
          setShowResults(true);
        } catch (e) {
          console.error('submitDateWithInput error:', e);
          setResults(prev => [...prev, { desc: 'Error', output: e.message }]);
        }
      };

      const handleCUTimeSubmit = () => {
        try {
          if (!input.cuTime) {
            setModal({
              message: 'Please enter a CU-Time value.',
              onConfirm: () => setModal(null),
              onCancel: () => setModal(null)
            });
            return;
          }

          const cuTimeStr = input.cuTime.replace("CU-Time: ", "").trim();

          const { valid, message, note } = ethicalClassifier(cuTimeStr);
          if (!valid) {
            setResults(prev => [...prev, { desc: `CU-Time: ${cuTimeStr}`, output: message }]);
            return;
          }

          const cuTime = new Decimal(cuTimeStr);
          if (cuTime.lt(0)) throw new Error('CU-Time cannot be negative');
          const deltaYears = cuTime.minus(CONSTANTS.BASE_CU);
          const deltaJDN = deltaYears.times(CONSTANTS.DAYS_PER_YEAR);
          const jdn = CONSTANTS.ANCHOR_JDN.plus(deltaJDN);
          const date = jdnToDate(jdn);
          const formattedDate = typeof date.year === 'string' ? date.year :
            `${date.month.toString().padStart(2, '0')}/${date.day.toString().padStart(2, '0')}/${Math.abs(date.year)}${date.year < 0 ? ' BCE' : ''} ${date.hour.toString().padStart(2, '0')}:${date.minute.toString().padStart(2, '0')}:${date.second.toString().padStart(2, '0')} UTC`;
          const cosmicPhase = getCosmicPhase(cuTime);
          const dominantForces = getDominantForces(cuTime);
          const geologicalEpoch = getGeologicalEpoch(cuTime);
          const refCuTime = computeCUTime(2025, 5, 26, 9, 48, 0);
          const timeDiff = cuTime.minus(refCuTime).abs().toDecimalPlaces(24);
          const yearsDiff = typeof date.year === 'string' ? timeDiff.toNumber() : Math.abs(date.year - 2025);
          const output = [
            `Input: ${cuTimeStr} CU-Time`,
            `Gregorian Date: ${formattedDate}`,
            `Epoch/Period: ${cuTime.lt(CONSTANTS.BASE_CU) ? 'Past' : 'Future'}`,
            `Geological Epoch: ${geologicalEpoch}`,
            `Cosmic Phase: ${cosmicPhase}`,
            `Dominant Forces: ${dominantForces.join(', ')}`,
            `Time Difference from 2025-05-26T09:48:00+00:00: ${timeDiff} CU-Time (~${yearsDiff} years)`,
            `Ethical Status: ${message}`,
            note ? `Note: ${note}` : ''
          ].filter(line => line);

          setResults(prev => [...prev, { desc: `CU-Time: ${cuTimeStr}`, output: output.join('\n') }]);
          setEditedResults(prev => [...prev, { desc: `CU-Time: ${cuTimeStr}`, output: output.join('\n') }]);
          HISTORY.saveConversion(cuTimeStr, formattedDate, 'CU to Gregorian');
          setHistory(HISTORY.getHistory());
          setShowResults(true);
        } catch (e) {
          console.error('handleCUTimeSubmit error:', e);
          setResults(prev => [...prev, { desc: 'Error', output: e.message }]);
        }
      };

      const handleTomSubmit = () => {
        try {
          if (!input.tom) {
            setModal({
              message: 'Invalid tom from the CU Lexicon.',
              onConfirm: () => setModal(null),
              onCancel: () => setModal(null)
            });
            return;
          }

          const durationSeconds = parseTomDuration(input.tom);
          const cuDuration = durationSeconds.div(CONSTANTS.SECONDS_PER_YEAR).toDecimalPlaces(24);
          const output = [
            `Input: ${input.tom}`,
            `Duration: ${cuDuration} CU-Time (${durationSeconds.toExponential(6)} seconds)`,
            `Description: ${CU_LEXICON[input.tom]}`,
            `Ethical Status: Ethical: Input aligns with CU principles`
          ];
          setResults(prev => [...prev, { desc: `Tom: ${input.tom}`, output: output.join('\n') }]);
          setEditedResults(prev => [...prev, { desc: `Tom: ${input.tom}`, output: output.join('\n') }]);
          HISTORY.saveConversion(input.tom, `${cuDuration} CU-Time`, 'Tom Duration');
          setHistory(HISTORY.getHistory());
          setShowResults(true);
        } catch (e) {
          console.error('handleTomSubmit error:', e);
          setResults(prev => [...prev, { desc: 'Error', output: e.message }]);
        }
      };

      const handleDurationSubmit = () => {
        try {
          if (!input.date || !input.date2) {
            setModal({
              message: 'Invalid dates for duration calculation.',
              onConfirm: () => setModal(null),
              onCancel: () => setModal(null)
            });
            return;
          }

          if (!validateTime(input.time || '12:00:00') || !validateTime(input.time2 || '12:00:00')) {
            setModal({
              message: 'Invalid times in HH:MM:SS format (hours 1-12, minutes/seconds 0-59).',
              onConfirm: () => setModal(null),
              onCancel: () => setModal(null)
            });
            return;
          }

          const fullInput1 = `${input.date}${input.era === 'BCE' ? ' BCE' : ''} ${input.time || '00:00:01'} ${input.ampm} UTC`;
          const fullInput2 = `${input.date2}${input.era2 === 'BCE' ? ' BCE' : ''} ${input.time2 || '00:00:01'} ${input.ampm2} UTC`;

          const { valid: valid1, message: message1 } = ethicalClassifier(fullInput1);
          const { valid: valid2, message: message2 } = ethicalClassifier(fullInput2);
          if (!valid1 || !valid2) {
            setResults(prev => [...prev, { desc: 'Error', output: `${message1}\n${message2}` }]);
            return;
          }

          const parsed1 = parseDateInput(fullInput1);
          const parsed2 = parseDateInput(fullInput2);
          const cuTime1 = computeCUTime(parsed1.year, parsed1.month, parsed1.day, parsed1.hour, parsed1.minute, parsed1.second, parsed1.microsecond);
          const cuTime2 = computeCUTime(parsed2.year, parsed2.month, parsed2.day, parsed2.hour, parsed2.minute, parsed2.second, parsed2.microsecond);
          const cuDiff = cuTime1.minus(cuTime2).abs().toDecimalPlaces(24);
          const yearsDiff = cuDiff.toNumber();

          const formattedCuTime1 = formatCUValue(cuTime1);
          const formattedCuTime2 = formatCUValue(cuTime2);

          const output = [
            `Duration between ${fullInput1} and ${fullInput2}:`,
            '',
            `Start Date:`,
            `- Gregorian: ${fullInput1}`,
            `- CU-Time:`,
            `  - Human-Friendly: ${formattedCuTime1.humanFriendly}`,
            `  - Full Numeric: ${formattedCuTime1.fullNumeric} CU-Time`,
            `  - Exponential: ${formattedCuTime1.exponential}`,
            '',
            `End Date:`,
            `- Gregorian: ${fullInput2}`,
            `- CU-Time:`,
            `  - Human-Friendly: ${formattedCuTime2.humanFriendly}`,
            `  - Full Numeric: ${formattedCuTime2.fullNumeric} CU-Time`,
            `  - Exponential: ${formattedCuTime2.exponential}`,
            '',
            `Duration:`,
            `- ${cuDiff} CU-Time (~${yearsDiff.toFixed(2)} years)`,
            '',
            `Ethical Status: Ethical: Input aligns with CU principles`
          ].join('\n');

          const date1Str = input.era === 'BCE' ? `${input.date} BCE` : input.date;
          const date2Str = input.era2 === 'BCE' ? `${input.date2} BCE` : input.date2;
          const desc = `Duration: ${date1Str} to ${date2Str}`;

          setResults(prev => [...prev, { desc, output }]);
          setEditedResults(prev => [...prev, { desc, output }]);
          HISTORY.saveConversion(`${fullInput1} to ${fullInput2}`, output, 'Duration');
          setHistory(HISTORY.getHistory());
          setShowResults(true);
        } catch (e) {
          console.error('handleDurationSubmit error:', e);
          setResults(prev => [...prev, { desc: 'Error', output: e.message }]);
        }
      };

      const handleCSVUpload = async (e) => {
        try {
          const file = e.target.files[0];
          if (!file) return;
          const text = await file.text();
          const rows = text.split('\n').filter(row => row.trim());
          if (!rows[0].toLowerCase().startsWith('"description","datetime","output"')) {
            setModal({
              message: 'CSV must have headers "Description,DateTime,Output"',
              onConfirm: () => setModal(null)
            });
            return;
          }
          const newResults = [];
          for (let i = 1; i < rows.length; i++) {
            const fields = parseCSVRow(rows[i]);
            if (fields.length !== 3) {
              console.warn(`Skipping row ${i + 1}: Expected 3 fields, got ${fields.length}`);
              continue;
            }
            let [desc, dateTime, output] = fields;
            desc = desc.trim();
            dateTime = dateTime.trim();
            output = output.trim();
            if (!desc || !dateTime) {
              console.warn(`Skipping row ${i + 1}: Description or DateTime is empty`);
              continue;
            }

            if (desc.startsWith("CU-Time:")) {
              const cuTimeStr = dateTime.replace("CU-Time: ", "").trim();
              try {
                const { valid, message, note } = ethicalClassifier(cuTimeStr);
                if (!valid) {
                  newResults.push({ desc, date: dateTime, output: message });
                  continue;
                }
                const cuTime = new Decimal(cuTimeStr);
                const deltaYears = cuTime.minus(CONSTANTS.BASE_CU);
                const deltaJDN = deltaYears.times(CONSTANTS.DAYS_PER_YEAR);
                const jdn = CONSTANTS.ANCHOR_JDN.plus(deltaJDN);
                const date = jdnToDate(jdn);
                const formattedDate = typeof date.year === 'string' ? date.year :
                  `${date.month.toString().padStart(2, '0')}/${date.day.toString().padStart(2, '0')}/${Math.abs(date.year)}${date.year < 0 ? ' BCE' : ''} ${date.hour.toString().padStart(2, '0')}:${date.minute.toString().padStart(2, '0')}:${date.second.toString().padStart(2, '0')} UTC`;
                const cosmicPhase = getCosmicPhase(cuTime);
                const dominantForces = getDominantForces(cuTime);
                const geologicalEpoch = getGeologicalEpoch(cuTime);
                const refCuTime = computeCUTime(2025, 5, 26, 9, 48, 0);
                const timeDiff = cuTime.minus(refCuTime).abs().toDecimalPlaces(24);
                const yearsDiff = typeof date.year === 'string' ? timeDiff.toNumber() : Math.abs(date.year - 2025);
                const outputText = [
                  `Input: ${cuTimeStr} CU-Time`,
                  `Gregorian Date: ${formattedDate}`,
                  `Epoch/Period: ${cuTime.lt(CONSTANTS.BASE_CU) ? 'Past' : 'Future'}`,
                  `Geological Epoch: ${geologicalEpoch}`,
                  `Cosmic Phase: ${cosmicPhase}`,
                  `Dominant Forces: ${dominantForces.join(', ')}`,
                  `Time Difference from 2025-05-26T09:48:00+00:00: ${timeDiff} CU-Time (~${yearsDiff} years)`,
                  `Ethical Status: ${message}`,
                  note ? `Note: ${note}` : ''
                ].filter(line => line).join('\n');
                newResults.push({ desc, date: dateTime, output: outputText });
              } catch (error) {
                newResults.push({ desc, date: dateTime, output: error.message });
              }
            } else if (desc.startsWith("Tom:")) {
              const tom = dateTime.replace("Tom: ", "").trim();
              try {
                const durationSeconds = parseTomDuration(tom);
                const cuDuration = durationSeconds.div(CONSTANTS.SECONDS_PER_YEAR).toDecimalPlaces(24);
                const outputText = [
                  `Input: ${tom}`,
                  `Duration: ${cuDuration} CU-Time (${durationSeconds.toExponential(6)} seconds)`,
                  `Description: ${CU_LEXICON[tom]}`,
                  `Ethical Status: Ethical: Input aligns with CU principles`
                ].join('\n');
                newResults.push({ desc, date: dateTime, output: outputText });
              } catch (error) {
                newResults.push({ desc, date: dateTime, output: error.message });
              }
            } else if (desc.startsWith("Duration:")) {
              const dateRangeMatch = desc.match(/Duration: ([^\s].*?)\s+to\s+([^\s].*?)$/);
              if (!dateRangeMatch) {
                newResults.push({ desc, date: dateTime, output: "Invalid duration format" });
                continue;
              }
              const [ , startDateStr, endDateStr ] = dateRangeMatch;
              const dateRange = `${startDateStr} to ${endDateStr}`;
              try {
                const parsed1 = parseDateInput(startDateStr + " 12:00:00 AM");
                const parsed2 = parseDateInput(endDateStr + " 12:00:00 AM");
                const cuTime1 = computeCUTime(parsed1.year, parsed1.month, parsed1.day, parsed1.hour, parsed1.minute, parsed1.second, parsed1.microsecond);
                const cuTime2 = computeCUTime(parsed2.year, parsed2.month, parsed2.day, parsed2.hour, parsed2.minute, parsed2.second, parsed2.microsecond);
                const cuDiff = cuTime1.minus(cuTime2).abs().toDecimalPlaces(24);
                const yearsDiff = cuDiff.toNumber();
                const formattedCuTime1 = formatCUValue(cuTime1);
                const formattedCuTime2 = formatCUValue(cuTime2);
                const outputText = [
                  `Duration between ${startDateStr} 12:00:00 AM UTC and ${endDateStr} 12:00:00 AM UTC:`,
                  '',
                  `Start Date:`,
                  `- Gregorian: ${startDateStr} 12:00:00 AM UTC`,
                  `- CU-Time:`,
                  `  - Human-Friendly: ${formattedCuTime1.humanFriendly}`,
                  `  - Full Numeric: ${formattedCuTime1.fullNumeric} CU-Time`,
                  `  - Exponential: ${formattedCuTime1.exponential}`,
                  '',
                  `End Date:`,
                  `- Gregorian: ${endDateStr} 12:00:00 AM UTC`,
                  `- CU-Time:`,
                  `  - Human-Friendly: ${formattedCuTime2.humanFriendly}`,
                  `  - Full Numeric: ${formattedCuTime2.fullNumeric} CU-Time`,
                  `  - Exponential: ${formattedCuTime2.exponential}`,
                  '',
                  `Duration:`,
                  `- ${cuDiff} CU-Time (~${yearsDiff.toFixed(2)} years)`,
                  '',
                  `Ethical Status: Ethical: Input aligns with CU principles`
                ].join('\n');
                newResults.push({ desc, date: dateRange, output: outputText });
              } catch (error) {
                newResults.push({ desc, date: dateRange, output: error.message });
              }
            } else {
              try {
                const { valid, message, note } = ethicalClassifier(dateTime);
                if (!valid) {
                  newResults.push({ desc, date: dateTime, output: message });
                  continue;
                }
                const parsed = parseDateInput(dateTime);
                const cuTime = computeCUTime(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second, parsed.microsecond);
                const formatted = formatCUValue(cuTime);
                const cosmicPhase = getCosmicPhase(cuTime);
                const dominantForces = getDominantForces(cuTime);
                const geologicalEpoch = getGeologicalEpoch(cuTime);
                const historicalEvents = getHistoricalEvents(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second);
                const refCuTime = computeCUTime(2025, 5, 26, 9, 48, 0);
                const timeDiff = cuTime.minus(refCuTime).abs().toDecimalPlaces(24);
                const yearsDiff = Math.abs(parsed.year - 2025);
                let leapYearsStr = '';
                if (typeof parsed.year === 'number' && Math.abs(parsed.year) <= 28000000000) {
                  const leapYears = countLeapYears(-28000000000, parsed.year);
                  leapYearsStr = `Leap Years from Sub-ZTOM (28000000000 BCE): ${leapYears}`;
                } else {
                  leapYearsStr = 'Leap Years: N/A (Year out of range)';
                }
                const outputText = [
                  `Input: ${dateTime} UTC`,
                  `CU-Time: ${formatted.humanFriendly}`,
                  `Full Numeric: ${formatted.fullNumeric} CU-Time`,
                  `Exponential: ${formatted.exponential}`,
                  `Epoch/Period: ${cuTime.lt(CONSTANTS.BASE_CU) ? 'Past' : 'Future'}`,
                  `Geological Epoch: ${geologicalEpoch}`,
                  `Cosmic Phase: ${cosmicPhase}`,
                  `Dominant Forces: ${dominantForces.join(', ')}`,
                  leapYearsStr,
                  historicalEvents.map(event => `Historical Event: ${event}`).join('\n'),
                  `Time Difference from 2025-05-26T09:48:00+00:00: ${timeDiff} CU-Time (~${yearsDiff} years)`,
                  `Ethical Status: ${message}`,
                  note ? `Note: ${note}` : ''
                ].filter(line => line).join('\n');
                newResults.push({ desc, date: dateTime, output: outputText });
              } catch (error) {
                newResults.push({ desc, date: dateTime, output: error.message });
              }
            }
          }
          setResults(newResults);
          setEditedResults(newResults);
          setShowResults(true);
        } catch (e) {
          console.error('handleCSVUpload error:', e);
          setResults([{ desc: 'Error', output: e.message }]);
        }
      };

      const handleResultEdit = (index, field, value) => {
        const updatedResults = [...editedResults];
        updatedResults[index] = { ...updatedResults[index], [field]: value };
        setEditedResults(updatedResults);
      };

      const handleDeleteResult = (index) => {
        const updatedResults = editedResults.filter((_, i) => i !== index);
        setEditedResults(updatedResults);
        setResults(updatedResults);
      };

      const toggleFavorite = (input) => {
        try {
          HISTORY.markFavorite(input);
          setHistory(HISTORY.getHistory());
        } catch (e) {
          console.error('toggleFavorite error:', e);
        }
      };

      return (
              <ErrorBoundary>
                <div className="container mx-auto p-6">
                  <header className="mb-6 p-6 bg-gradient-to-r from-indigo-950 via-blue-950 to-purple-950 border border-indigo-800/30 rounded-lg shadow-lg container mx-auto">
                    <div className="flex flex-col items-center md:flex-row md:justify-between">
                      <a href="https://github.com/willmaddock/CosmicUniversalismStatement.git" target="_blank"
                         rel="noopener noreferrer"
                         className="transition-transform duration-300 ease-in-out hover:scale-105 hover:brightness-125 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <h1 className="text-4xl font-extrabold text-white tracking-tight mb-2 md:mb-0 text-center md:text-left">
                          Cosmic Universalism Time Converter
                        </h1>
                      </a>
                      <a href="https://github.com/willmaddock/CosmicUniversalismStatement.git" target="_blank"
                         rel="noopener noreferrer"
                         className="w-full max-w-xl transition-transform duration-300 ease-in-out hover:scale-105 hover:brightness-125 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <div className="p-4 bg-indigo-950/20 border border-indigo-800/30 rounded-md">
                          <h2 className="text-lg font-semibold text-gray-100 mb-2 text-center md:text-left">Cosmic
                            Universalism Statement v2.2.0</h2>
                          <p className="text-sm text-gray-200 leading-relaxed text-center md:text-left font-light"
                             role="description">
                            We are sub z-tomically inclined, countably infinite, composed of foundational elements (the
                            essence of conscious existence), grounded on b-tom (as vast as our shared worlds and their
                            atmospheres), and looking up to c-tom (encompassing the entirety of the cosmos), guided by the
                            uncountable infinite quantum states of intelligence and empowered by God’s Free Will.
                          </p>
                        </div>
                      </a>
                    </div>
                  </header>
                  <nav className="bg-gray-800 p-4 rounded-lg mb-6">
                    <ul className="flex flex-col md:flex-row justify-center space-y-2 md:space-y-0 md:space-x-8"
                        role="tablist">
                      <li>
                        <button
                                onClick={() => setTab('gregorian')}
                                className={`px-4 py-2 text-white rounded ${tab === 'gregorian' ? 'bg-blue-600' : 'hover:bg-blue-700'} transition duration-300`}
                        >
                          Gregorian to CU-Time
                        </button>
                      </li>
                      <li>
                        <button
                                onClick={() => setTab('cuTime')}
                                className={`px-4 py-2 text-white rounded ${tab === 'cuTime' ? 'bg-blue-600' : 'hover:bg-blue-700'} transition duration-300`}
                        >
                          CU-Time to Gregorian
                        </button>
                      </li>
                      <li>
                        <button
                                onClick={() => setTab('tom')}
                                className={`px-4 py-2 text-white rounded ${tab === 'tom' ? 'bg-blue-600' : 'hover:bg-blue-700'} transition duration-300`}
                        >
                          Tom Duration
                        </button>
                      </li>
                      <li>
                        <button
                                onClick={() => setTab('duration')}
                                className={`px-4 py-2 text-white rounded ${tab === 'duration' ? 'bg-blue-600' : 'hover:bg-blue-700'} transition duration-300`}
                        >
                          Calculate Duration
                        </button>
                      </li>
                    </ul>
                  </nav>
                  <main>
                    {tab === 'gregorian' && (
                            <section id="gregorian-to-cu" className="mb-8 bg-gray-800 rounded-lg shadow-md p-6">
                              <h2 className="text-2xl font-semibold text-white mb-4">Gregorian to CU-Time
                                Conversion</h2>
                              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="date">Date (MM/DD/YYYY)</label>
                                  <input
                                          type="text"
                                          id="date"
                                          value={input.date}
                                          onChange={(e) => handleDateChange(e)}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                          placeholder="e.g., 05/26/2025"
                                  />
                                </div>
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="time">Time (HH:MM:SS)</label>
                                  <div className="flex space-x-2">
                                    <input
                                            type="text"
                                            id="time"
                                            value={input.time}
                                            onChange={(e) => handleTimeChange(e)}
                                            className="flex-1 px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                            placeholder="e.g., 03:48:00"
                                    />
                                    <select
                                            value={input.ampm}
                                            onChange={(e) => handleAmPmChange(e.target.value)}
                                            className="px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                    >
                                      <option value="AM">AM</option>
                                      <option value="PM">PM</option>
                                    </select>
                                  </div>
                                </div>
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="era">Era</label>
                                  <select
                                          id="era"
                                          value={input.era}
                                          onChange={(e) => setInput({...input, era: e.target.value})}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                  >
                                    <option value="CE">CE</option>
                                    <option value="BCE">BCE</option>
                                  </select>
                                </div>
                                <div className="md:col-span-3">
                                  <button
                                          onClick={handleDateSubmit}
                                          className="w-full md:w-auto px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition duration-300"
                                  >
                                    Convert
                                  </button>
                                </div>
                              </div>
                            </section>
                    )}

                    {tab === 'cuTime' && (
                            <section id="cu-to-gregorian" className="mb-8 bg-gray-800 rounded-lg shadow-md p-6">
                              <h2 className="text-2xl font-semibold text-white mb-4">CU-Time to Gregorian
                                Conversion</h2>
                              <div className="grid grid-cols-1 gap-4">
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="cuTime">CU-Time</label>
                                  <input
                                          type="text"
                                          id="cuTime"
                                          value={input.cuTime}
                                          onChange={(e) => setInput({...input, cuTime: e.target.value})}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                          placeholder="e.g., 3094134044966.672659"
                                  />
                                </div>
                                <div>
                                  <button
                                          onClick={handleCUTimeSubmit}
                                          className="w-full md:w-auto px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition duration-300"
                                  >
                                    Convert
                                  </button>
                                </div>
                              </div>
                            </section>
                    )}

                    {tab === 'tom' && (
                            <section id="tom-duration" className="mb-8 bg-gray-800 rounded-lg shadow-md p-6">
                              <h2 className="text-2xl font-semibold text-white mb-4">Tom Duration</h2>
                              <div className="grid grid-cols-1 gap-4">
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="tom">Select Tom</label>
                                  <select
                                          id="tom"
                                          value={input.tom}
                                          onChange={(e) => setInput({...input, tom: e.target.value})}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                  >
                                    <option value="">Select a tom</option>
                                    {Object.keys(CU_LEXICON).map(tom => (
                                            <option key={tom} value={tom}>{tom}</option>
                                    ))}
                                  </select>
                                </div>
                                <div>
                                  <button
                                          onClick={handleTomSubmit}
                                          className="w-full md:w-auto px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition duration-300"
                                  >
                                    Calculate
                                  </button>
                                </div>
                              </div>
                            </section>
                    )}

                    {tab === 'duration' && (
                            <section id="calculate-duration" className="mb-8 bg-gray-800 rounded-lg shadow-md p-6">
                              <h2 className="text-2xl font-semibold text-white mb-4">Calculate Duration</h2>
                              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="date">First Date
                                    (MM/DD/YYYY)</label>
                                  <input
                                          type="text"
                                          id="date"
                                          value={input.date}
                                          onChange={(e) => handleDateChange(e, 'date')}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                          placeholder="e.g., 05/26/2025"
                                  />
                                </div>
                                <div>
                                  <label className="block LABEL text-gray-300 mb-1" htmlFor="time">Time
                                    (HH:MM:SS)</label>
                                  <div className="flex space-x-2">
                                    <input
                                            type="text"
                                            id="time"
                                            value={input.time}
                                            onChange={(e) => handleTimeChange(e, 'time')}
                                            className="flex-1 px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                            placeholder="e.g., 03:48:00"
                                    />
                                    <select
                                            value={input.ampm}
                                            onChange={(e) => handleAmPmChange(e.target.value, 'ampm')}
                                            className="px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                    >
                                      <option value="AM">AM</option>
                                      <option value="PM">PM</option>
                                    </select>
                                  </div>
                                </div>
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="era">Era</label>
                                  <select
                                          id="era"
                                          value={input.era}
                                          onChange={(e) => setInput({...input, era: e.target.value})}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                  >
                                    <option value="CE">CE</option>
                                    <option value="BCE">BCE</option>
                                  </select>
                                </div>
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="date2">Second Date
                                    (MM/DD/YYYY)</label>
                                  <input
                                          type="text"
                                          id="date2"
                                          value={input.date2}
                                          onChange={(e) => handleDateChange(e, 'date2')}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                          placeholder="e.g., 05/26/2025"
                                  />
                                </div>
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="time2">Time (HH:MM:SS)</label>
                                  <div className="flex space-x-2">
                                    <input
                                            type="text"
                                            id="time2"
                                            value={input.time2}
                                            onChange={(e) => handleTimeChange(e, 'time2')}
                                            className="flex-1 px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                            placeholder="e.g., 03:48:00"
                                    />
                                    <select
                                            value={input.ampm2}
                                            onChange={(e) => handleAmPmChange(e.target.value, 'ampm2')}
                                            className="px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                    >
                                      <option value="AM">AM</option>
                                      <option value="PM">PM</option>
                                    </select>
                                  </div>
                                </div>
                                <div>
                                  <label className="block text-gray-300 mb-1" htmlFor="era2">Era</label>
                                  <select
                                          id="era2"
                                          value={input.era2}
                                          onChange={(e) => setInput({...input, era2: e.target.value})}
                                          className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                                  >
                                    <option value="CE">CE</option>
                                    <option value="BCE">BCE</option>
                                  </select>
                                </div>
                                <div className="md:col-span-3">
                                  <button
                                          onClick={handleDurationSubmit}
                                          className="w-full md:w-auto px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition duration-300"
                                  >
                                    Calculate
                                  </button>
                                </div>
                              </div>
                            </section>
                    )}

                    <section className="mb-8 bg-gray-800 rounded-lg shadow-md p-6">
                      <h2 className="text-2xl font-semibold text-white mb-4">Upload CSV</h2>
                      <input
                              type="file"
                              accept=".csv"
                              onChange={handleCSVUpload}
                              className="w-full px-3 py-2 border rounded bg-gray-700 text-white focus:ring-2 focus:ring-blue-500"
                      />
                      <p className="text-sm text-gray-300 mt-2">
                        CSV must have headers "Description,DateTime,Output". Example: "Today","05/26/2025 03:48:00
                        AM","3094134044966.672659 CU-Time"
                      </p>
                    </section>

                    <div className="flex justify-between mb-6">
                      <button
                              onClick={() => setShowLexicon(!showLexicon)}
                              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition duration-300"
                      >
                        {showLexicon ? 'Hide' : 'Show'} CU Lexicon
                      </button>
                      <div className="space-x-2">
                        <button
                                onClick={() => downloadCSV(results)}
                                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition duration-300"
                        >
                          Download Results
                        </button>
                        <button
                                onClick={() => saveEditedCSV(editedResults)}
                                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition duration-300"
                                disabled={editedResults.length === 0}
                        >
                          Save Edited CSV
                        </button>
                        <button
                                onClick={() => {
                                  setResults([]);
                                  setEditedResults([]);
                                }}
                                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition duration-300"
                        >
                          Clear Results
                        </button>
                      </div>
                    </div>

                    {showLexicon && (
                            <section className="mb-8 bg-gray-800 rounded-lg shadow-md p-6 lexicon-container">
                              <h2 className="text-2xl font-semibold text-white mb-4">CU Lexicon</h2>
                              {Object.entries(CU_LEXICON).map(([tom, desc]) => (
                                      <p key={tom} className="text-sm text-gray-300">
                                        <strong>{tom}:</strong> {desc}
                                      </p>
                              ))}
                            </section>
                    )}

                    {showResults && editedResults.length > 0 && (
                            <section className="mb-8">
                              <h2 className="text-2xl font-semibold text-white mb-4">Results</h2>
                              {editedResults.map((result, idx) => (
                                      <div key={idx} className="bg-gray-900 rounded-lg shadow-md p-6 mb-4 relative">
                                        <button
                                                onClick={() => handleDeleteResult(idx)}
                                                className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded hover:bg-red-600 transition duration-300"
                                        >
                                          Delete
                                        </button>
                                        <input
                                                type="text"
                                                value={result.desc}
                                                onChange={(e) => handleResultEdit(idx, 'desc', e.target.value)}
                                                className="w-full px-3 py-2 border rounded bg-gray-700 text-white mb-2"
                                        />
                                        <textarea
                                                value={result.output}
                                                readOnly
                                                className="w-full px-3 py-2 border rounded bg-gray-800 text-white"
                                                rows="10"
                                        />
                                      </div>
                              ))}
                            </section>
                    )}

                    <section className="mb-8">
                      <div className="flex justify-between mb-4">
                        <h2 className="text-2xl font-semibold text-white">History</h2>
                        <button
                                onClick={() => setShowFavorites(!showFavorites)}
                                className="text-blue-400 hover:text-blue-500 transition duration-300"
                        >
                          {showFavorites ? 'Show All' : 'Show Favorites'}
                        </button>
                      </div>
                      {(showFavorites ? HISTORY.getFavorites() : history).map((entry, idx) => (
                              <div key={idx}
                                   className="bg-gray-800 rounded-lg shadow-md p-4 mb-2 flex justify-between items-center">
                                <div>
                                  <p className="text-sm text-gray-300">{entry.input}</p>
                                  <p className="text-sm text-gray-300">{entry.output}</p>
                                  <p className="text-xs text-gray-500">{entry.type} | {DateTime.fromISO(entry.timestamp).toLocaleString(DateTime.DATETIME_FULL)}</p>
                                </div>
                                <button
                                        onClick={() => toggleFavorite(entry.input)}
                                        className="text-blue-400 hover:text-blue-500 transition duration-300"
                                >
                                  {entry.favorite ? '★' : '☆'}
                                </button>
                              </div>
                      ))}
                    </section>
                  </main>
                  {modal && <Modal {...modal} />}
                </div>
                <footer className="mt-8 p-6 bg-gradient-to-r from-indigo-950 via-blue-950 to-purple-950 border border-indigo-800/30 rounded-lg shadow-lg container mx-auto text-center text-gray-200">
                  <p className="text-sm font-light">
                    © 2025 Cosmic Universalism Computational Intelligence Initiative
                  </p>
                  <p className="text-sm font-light mt-2">
                    Conceptual content licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"
                                                         target="_blank" rel="noopener noreferrer"
                                                         className="underline hover:text-white transition-colors">CC
                    BY-NC-SA 4.0</a>.
                    Source code licensed under <a
                          href="https://github.com/willmaddock/CosmicUniversalismStatement/blob/main/LICENSE"
                          target="_blank" rel="noopener noreferrer"
                          className="underline hover:text-white transition-colors">MIT License</a>.
                  </p>
                </footer>
              </ErrorBoundary>
      );
    }

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<App/>);
  </script>
</body>
</html>
```