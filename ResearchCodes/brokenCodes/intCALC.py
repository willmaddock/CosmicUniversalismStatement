import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import Qt

# ------ Cosmic Universalism Statement ------
COSMIC_STATEMENT = (
    "We are sub z-tomically inclined, countably infinite, composed of foundational elements "
    "(the essence of conscious existence), grounded on b-tom (as vast as our shared worlds and their atmospheres), "
    "and looking up to c-tom (encompassing the entirety of the cosmos), guided by the uncountable infinite quantum "
    "states of intelligence and empowered by God’s free will."
)

# ------ Corrected Conversion Rates (1 ztom-second = X target unit) ------
# Units mapped to their time equivalents (years, days, hours, etc.)
CONVERSION_DATA = {
    "z-tom": {"factor": 1, "unit": "second"},
    "c-tom": {"factor": 28e9, "unit": "years"},          # 28 billion years
    "d-tom": {"factor": 427350, "unit": "years"},        # 427,350 years
    "e-tom": {"factor": 42735, "unit": "years"},         # 42,735 years
    "f-tom": {"factor": 4273.5, "unit": "years"},        # 4,273.5 years
    "g-tom": {"factor": 427.35, "unit": "years"},        # 427.35 years
    "h-tom": {"factor": 85.47, "unit": "years"},         # 85.47 years
    "i-tom": {"factor": 8.547, "unit": "years"},         # 8.547 years
    "j-tom": {"factor": 0.8547, "unit": "years"},        # ~312.96 days
    "k-tom": {"factor": 31.296, "unit": "days"},         # 31.296 days
    "l-tom": {"factor": 3.1296, "unit": "days"},         # 3.1296 days
    "m-tom": {"factor": 7.51, "unit": "hours"},          # 7.51 hours
    "n-tom": {"factor": 45.06, "unit": "minutes"},       # 45.06 minutes
    "o-tom": {"factor": 4.506, "unit": "minutes"},       # 4.506 minutes
    "p-tom": {"factor": 27.04, "unit": "seconds"},       # 27.04 seconds
    "q-tom": {"factor": 2.704, "unit": "seconds"},       # 2.704 seconds
    "r-tom": {"factor": 0.2704, "unit": "seconds"},      # 270.4 milliseconds
    "s-tom": {"factor": 0.02704, "unit": "seconds"},     # 27.04 milliseconds
    "t-tom": {"factor": 0.002704, "unit": "seconds"},    # 2.704 milliseconds
    "u-tom": {"factor": 0.0002704, "unit": "seconds"},   # 0.2704 milliseconds
    "v-tom": {"factor": 2.704e-5, "unit": "seconds"},    # 27.04 microseconds
    "w-tom": {"factor": 2.704e-6, "unit": "seconds"},    # 2.704 microseconds
    "x-tom": {"factor": 2.704e-7, "unit": "seconds"},    # 0.2704 microseconds
    "y-tom": {"factor": 2.704e-8, "unit": "seconds"},    # 27.04 nanoseconds
}

# ------ Unit Conversion Factors to Seconds ------
TIME_UNITS_TO_SECONDS = {
    "years": 31536000,    # 365 days/year * 86400 seconds/day
    "days": 86400,        # 24 hours/day * 3600 seconds/hour
    "hours": 3600,
    "minutes": 60,
    "seconds": 1,
}

# ------ GUI Application ------
class TimeConversionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Universalism Time Converter")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Cosmic Universalism Statement
        cosmic_label = QLabel(COSMIC_STATEMENT)
        cosmic_label.setWordWrap(True)
        cosmic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(cosmic_label)

        # Dropdown for selecting time (1 to 2 seconds)
        self.time_selector = QComboBox()
        self.time_selector.addItems(["1", "2"])  # Choices from 1 to 2 seconds
        layout.addWidget(QLabel("Select time (seconds):"))
        layout.addWidget(self.time_selector)

        # Unit Selectors
        self.from_unit = QComboBox()
        self.from_unit.addItems(CONVERSION_DATA.keys())
        self.to_unit = QComboBox()
        self.to_unit.addItems(CONVERSION_DATA.keys())
        layout.addWidget(QLabel("From Unit:"))
        layout.addWidget(self.from_unit)
        layout.addWidget(QLabel("To Unit:"))
        layout.addWidget(self.to_unit)

        # Convert Button
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self.convert)
        layout.addWidget(self.convert_btn)

        # Result Display
        self.result_label = QLabel("Result: ")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def convert(self):
        try:
            # Get selected time value
            value = int(self.time_selector.currentText())
            from_unit = self.from_unit.currentText()
            to_unit = self.to_unit.currentText()

            # Convert from the selected unit to z-tom seconds
            from_factor = CONVERSION_DATA[from_unit]["factor"]
            from_unit_type = CONVERSION_DATA[from_unit]["unit"]

            # Handle conversion from the unit to seconds
            if from_unit == "z-tom":
                from_seconds = value * from_factor
            else:
                from_seconds = value * from_factor * TIME_UNITS_TO_SECONDS[from_unit_type]

            # Convert from z-tom seconds to the target unit
            to_factor = CONVERSION_DATA[to_unit]["factor"]
            to_unit_type = CONVERSION_DATA[to_unit]["unit"]

            if to_unit == "z-tom":
                result = from_seconds / to_factor
            else:
                result = from_seconds / (to_factor * TIME_UNITS_TO_SECONDS[to_unit_type])

            # Format result
            self.result_label.setText(
                f"{value} {from_unit} = {result:.4g} {to_unit} "
                f"({CONVERSION_DATA[to_unit]['factor']} {CONVERSION_DATA[to_unit]['unit']})"
            )

        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimeConversionApp()
    window.show()
    sys.exit(app.exec())