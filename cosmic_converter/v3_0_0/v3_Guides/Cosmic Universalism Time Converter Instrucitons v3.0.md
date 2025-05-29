# 🌌 Cosmic Universalism Time Converter v3.0.0 Instructions

📅 **Version Release Date:** May 28, 2025  
📦 **Version:** 3.0.0  
🔗 **Live Demo:** [CU Time Converter v3.0.0](https://willmaddock.github.io/CosmicUniversalismStatement/cosmic_converter/v3_0_0/cu_time_converter_stable_v3_0_0.html)  
🔗 **Repository:** [Cosmic Universalism GitHub](https://github.com/willmaddock/CosmicUniversalismStatement)  
📜 **License:** Conceptual content under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), Source code under [MIT License](https://github.com/willmaddock/CosmicUniversalismStatement/blob/main/LICENSE)

The Cosmic Universalism Time Converter v3.0.0 is a web-based tool that converts between Gregorian dates and Cosmic Universalism (CU) Time, a system aligned with cosmic phases and geological epochs. This guide explains how to use the application’s features, including its responsive navigation, conversion tools, history management, and CSV import/export capabilities.

## 🚀 Getting Started

1. **Access the Application**:
   - Open the [live demo](https://willmaddock.github.io/CosmicUniversalismStatement/cosmic_converter/v3_0_0/cu_time_converter_stable_v3_0_0.html) in a modern web browser (e.g., Chrome, Firefox, Safari).
   - The app is optimized for both desktop and mobile devices.

2. **Understand the Interface**:
   - **Header**: Displays the app title and the Cosmic Universalism Statement, with a link to the GitHub repository.
   - **Navigation**:
     - **Large Screens (≥768px)**: A fixed sidebar on the left contains tabs for “Gregorian to CU-Time,” “CU-Time to Gregorian,” and “CU-Time Calculator.”
     - **Small Screens (<768px)**: A top navigation bar in the header provides the same tabs, ensuring easy access on mobile devices.
   - **Main Content**: Includes input forms for conversions, a results section, conversion history, and a lexicon of CU terms.
   - **Footer**: Contains licensing information and links.

3. **Prerequisites**:
   - A stable internet connection for loading external libraries (e.g., React, Tailwind CSS).
   - No software installation is required, as the app runs entirely in the browser.

## 📋 Features and Usage

### 1. Gregorian to CU-Time Conversion
Convert a Gregorian date to CU-Time, which represents time relative to cosmic phases.

- **Steps**:
  1. Navigate to the “Gregorian to CU-Time” tab.
  2. **Geological Period (Optional)**: Select a period (e.g., “Holocene,” “Black Hole Era”) to auto-fill a representative date, or leave as “Auto-Detect.”
  3. **Date**: Enter a date in `MM/DD/YYYY` format (e.g., `07/20/1969` for the Moon Landing).
     - For BCE dates, select “Past” in the Era dropdown and enter a positive year.
     - For extreme dates (e.g., >9999 CE or BCE), the app automatically handles exotic formats.
  4. **Time (Optional)**: Enter a time in either:
     - **UTC**: `HH:MM:SS` (24-hour, e.g., `15:30:00`).
     - **AM/PM**: `HH:MM:SS AM/PM` (12-hour, e.g., `03:30:00 PM`).
     - If omitted, defaults to `00:00:00` (UTC) or `12:00:00 AM` (AM/PM).
  5. **Time Format**: Choose “UTC” or “AM/PM.”
  6. **Era**: Select “Present” (CE) or “Past” (BCE).
  7. Click **Convert** to generate the CU-Time.

- **Output**:
  - CU-Time in human-friendly, numeric, and exponential formats.
  - Geological epoch, cosmic phase, dominant forces, leap years (if applicable), and historical events (if near a known event, e.g., Moon Landing).
  - Time difference from May 26, 2025, and ethical status (aligned with CU principles).

- **Example**:
  - Input: `07/20/1969 20:17:00 UTC`, Era: Present
  - Output: `CU-Time: 3094134044948.672659`, Epoch: Anthropocene, Phase: Dark Energy, etc.

### 2. CU-Time to Gregorian Conversion
Convert a CU-Time value back to a Gregorian date.

- **Steps**:
  1. Navigate to the “CU-Time to Gregorian” tab.
  2. Enter a CU-Time value (e.g., `3094134044948.672659`).
  3. Click **Convert**.

- **Output**:
  - Corresponding Gregorian date (e.g., `07/20/1969 20:17:00 UTC`).
  - Epoch, cosmic phase, dominant forces, and time difference from May 26, 2025.

- **Note**:
  - CU-Time must be positive and within valid ranges (e.g., not excessively far from 3.094T).

### 3. CU-Time Calculator
Perform calculations by adding or subtracting years from a starting CU-Time.

- **Steps**:
  1. Navigate to the “CU-Time Calculator” tab.
  2. **Starting CU-Time**: Enter a CU-Time value or click “Use Current CU-Time” to populate the current time.
  3. **Operation**: Choose “Add” or “Subtract.”
  4. **Years**: Enter a positive number of years (e.g., `10000000000`).
  5. Click **Calculate**.

- **Output**:
  - Resulting CU-Time and equivalent Gregorian date.
  - Epoch, phase, forces, and time difference.

- **Example**:
  - Input: Starting CU-Time `3094134044923.509753564772922302508810`, Add `100` years
  - Output: New CU-Time, date ~2125 CE, etc.

### 4. CSV Import/Export
Import multiple conversions or export results to CSV.

- **Import**:
  1. Go to the “Upload CSV” section.
  2. Select a CSV file with headers: `Description,DateTime,Output`.
     - Example: `"Moon Landing","07/20/1969 20:17:00 UTC","3094134044948.672659 CU-Time"`
  3. Upload to process conversions and display results.

- **Export**:
  1. After performing conversions, click **Download Results** to save a CSV file.
  2. To save edited results, click **Save Edited CSV** (available if results are modified).

- **Note**:
  - Ensure CSV files are properly formatted to avoid errors.
  - Malformed files may trigger error messages.

### 5. Conversion History and Favorites
Track and manage past conversions.

- **View History**:
  - Scroll to the “History” section to see all conversions, including input, output, type, and timestamp.
- **Favorites**:
  - Click the star (☆) next to a history entry to mark it as a favorite (★).
  - Toggle “Show Favorites” to view only favorited entries.
- **Note**:
  - History is stored in the browser session and may clear on refresh.

### 6. Cosmic Breath Lexicon
Explore CU terminology.

- **Steps**:
  1. Click **Show Cosmic Breath** to display the lexicon.
  2. Browse terms like `ztom`, `btom`, `anti-dark-energy`, with their definitions.
  3. Click **Hide Cosmic Breath** to collapse the section.

## 🛠️ Tips and Best Practices

- **Input Validation**:
  - Ensure dates are in `MM/DD/YYYY` format and times match the selected format (UTC or AM/PM).
  - For BCE dates, use the “Past” era and positive years (e.g., `01/01/3000` with Past for 3000 BCE).
- **Extreme Dates**:
  - The app supports dates up to ~322 billion years, but extreme values may be speculative.
- **Mobile Usage**:
  - On smaller screens, use the top navigation bar to switch between tabs.
  - Ensure inputs are clear, as modals and forms adjust for mobile displays.
- **Error Handling**:
  - If an error occurs (e.g., invalid input), a modal will display the issue. Correct the input and retry.
- **Performance**:
  - For older browsers, expect slight delays due to in-browser compilation.
  - Clear results if the interface slows down after many conversions.

## ⚠️ Known Limitations

- **Browser Compatibility**: May have rendering delays on older browsers due to Babel compilation.
- **CSV Imports**: Malformed or large CSV files may cause errors; ensure proper formatting.
- **Responsive Design**: Sidebar spacing on narrow desktop screens (768–900px) may need adjustment.
- **History Persistence**: Conversion history resets on browser refresh.

## 🌌 About Cosmic Universalism Time
CU-Time is a unique system that aligns time with cosmic phases (e.g., Dark Energy, Anti-Dark Energy) and geological epochs, using a recalibrated `BASE_CU` (`3094134044923.509753564772922302508810`) for precision. It’s designed to contextualize human and cosmic events within the universe’s vast timeline, guided by the principles of Cosmic Universalism and empowered by God’s Free Will.

Enjoy exploring the cosmos with the Cosmic Universalism Time Converter v3.0.0! 🚀