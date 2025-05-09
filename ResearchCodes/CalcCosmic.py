import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox,
    QPushButton, QLineEdit, QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QClipboard, QIcon

# ------ Cosmic Universalism Statement ------
COSMIC_STATEMENT = (
    "We are sub z-tomically inclined, countably infinite, composed of foundational elements "
    "(the essence of conscious existence), grounded on b-tom (as vast as our shared worlds and their atmospheres), "
    "and looking up to c-tom (encompassing the entirety of the cosmos), guided by the uncountable infinite quantum "
    "states of intelligence and empowered by God’s free will."
)

# ------ Corrected Conversion Rates (1 ztom-second = X target unit) ------
class ConversionUnit:
    def __init__(self, name, factor, unit_type):
        self.name = name
        self.factor = factor
        self.unit_type = unit_type

    def convert_to_seconds(self):
        """Converts the unit to seconds based on its factor and unit type."""
        if self.unit_type == "seconds":
            return self.factor
        else:
            return self.factor * TIME_UNITS_TO_SECONDS[self.unit_type]

# Units list with custom conversion factors, sorted alphabetically by name
CONVERSION_UNITS = [
    ConversionUnit("a-tom", 28e11, "years"),  # Subatomic or cosmic boundary
    ConversionUnit("b-tom", 28e10, "years"),  # Pre-cosmos phase
    ConversionUnit("c-tom", 28e9, "years"),  # Observable cosmos
    ConversionUnit("d-tom", 427350, "years"),  # Early universe
    ConversionUnit("e-tom", 42735, "years"),  # Galactic formation
    ConversionUnit("f-tom", 4273.5, "years"),  # Early solar system
    ConversionUnit("g-tom", 427.35, "years"),  # Planetary evolution
    ConversionUnit("h-tom", 85.47, "years"),  # Life emergence
    ConversionUnit("i-tom", 8.547, "years"),  # Civilizational era
    ConversionUnit("j-tom", 0.8547, "years"),  # Years in atomic time
    ConversionUnit("k-tom", 31.296, "days"),  # Days
    ConversionUnit("l-tom", 3.1296, "days"),  # Days in atomic time
    ConversionUnit("m-tom", 7.51, "hours"),  # Hours in atomic time
    ConversionUnit("n-tom", 45.06, "minutes"),  # Minutes in atomic time
    ConversionUnit("o-tom", 4.506, "minutes"),  # Minutes in atomic time
    ConversionUnit("p-tom", 27.04, "seconds"),  # Seconds in atomic time
    ConversionUnit("q-tom", 2.704, "seconds"),  # Sub-seconds
    ConversionUnit("r-tom", 0.2704, "seconds"),  # Microseconds
    ConversionUnit("s-tom", 0.02704, "seconds"),  # Nanoseconds
    ConversionUnit("t-tom", 0.002704, "seconds"),  # Picoseconds
    ConversionUnit("u-tom", 0.0002704, "seconds"),  # Femtoseconds
    ConversionUnit("v-tom", 2.704e-5, "seconds"),  # Attoseconds
    ConversionUnit("w-tom", 2.704e-6, "seconds"),  # Zeptoseconds
    ConversionUnit("x-tom", 2.704e-7, "seconds"),  # Yoctoseconds
    ConversionUnit("y-tom", 2.704e-8, "seconds"),  # Planck time
    ConversionUnit("z-tom", 1, "seconds"),  # Fundamental second
]

# ------ Unit Conversion Factors to Seconds ------
TIME_UNITS_TO_SECONDS = {
    "years": 31536000,  # 365 days/year * 86400 seconds/day
    "days": 86400,  # 24 hours/day * 3600 seconds/hour
    "hours": 3600,
    "minutes": 60,
    "seconds": 1,
}

# ------ GUI Application ------
class TimeConversionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Universalism Time Converter")
        self.setWindowIcon(QIcon("icon.png")) #Add icon, change "icon.png" to your icon file.
        self.setStyleSheet("background-color: #0e0b47; color: white;")
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout()

        # Cosmic Universalism Statement
        cosmic_label = QLabel(COSMIC_STATEMENT)
        cosmic_label.setWordWrap(True)
        cosmic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(cosmic_label, 0, 0, 1, 3) # Span across 3 columns

        # Unit Selectors
        self.from_unit_label = QLabel("From Unit: z-tom")
        layout.addWidget(self.from_unit_label, 1, 0)

        # 'To Unit' dropdown
        self.to_unit = QComboBox()
        for unit in CONVERSION_UNITS:
            if unit.name != "z-tom":
                self.to_unit.addItem(unit.name)
        layout.addWidget(QLabel("To Unit:"), 1, 1)
        layout.addWidget(self.to_unit, 1, 2)

        # Convert Button
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.convert_btn.clicked.connect(self.convert)
        layout.addWidget(self.convert_btn, 2, 0, 1, 3) # Span across 3 columns

        # Result Display
        self.result_label = QLineEdit("Result: ")
        self.result_label.setStyleSheet("font-size: 16px; color: #FFD700;")
        self.result_label.setReadOnly(True)
        layout.addWidget(self.result_label, 3, 0, 1, 2) # Span across 2 columns

        # Copy to Clipboard Button
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold;")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        layout.addWidget(self.copy_btn, 3, 2)

        self.setLayout(layout)

    def convert(self):
        try:
            value = 1  # Always 1 second for conversion
            to_unit_name = self.to_unit.currentText()
            to_unit = next(unit for unit in CONVERSION_UNITS if unit.name == to_unit_name)

            from_seconds = 1  # z-tom is fixed at 1 second
            result = from_seconds / to_unit.convert_to_seconds()

            # Format the conversion factor with human-readable suffix
            formatted_factor = self.human_readable_number(to_unit.factor)
            unit_description = f"{formatted_factor} {to_unit.unit_type}"

            self.result_label.setText(
                f"1 second = {result:.4g} {to_unit_name} ({unit_description})"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.result_label.text())
        QMessageBox.information(self, "Copied", "Result copied to clipboard!")

    @staticmethod
    def human_readable_number(number):
        suffixes = [
            (1e18, 'quintillion'),
            (1e15, 'quadrillion'),
            (1e12, 'trillion'),
            (1e9, 'billion'),
            (1e6, 'million'),
            (1e3, 'thousand'),
        ]

        for divisor, suffix in suffixes:
            if number >= divisor:
                value = number / divisor
                if value.is_integer():
                    return f"{int(value)} {suffix}"
                else:
                    # Remove trailing .0 if present
                    formatted_value = f"{value:.1f}".rstrip('0').rstrip('.')
                    return f"{formatted_value} {suffix}"

        # Handle numbers less than 1000
        return f"{number:.3g}".rstrip('.').rstrip('0') if '.' in f"{number:.3g}" else f"{int(number)}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimeConversionApp()
    window.show()
    sys.exit(app.exec())