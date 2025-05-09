import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout, QSlider
from PyQt6.QtCore import Qt

# ------ Human-Friendly Time Units ------
TIME_UNITS = {
    "y-tom": "Planck Time",
    "x-tom": "Yoctoseconds",
    "w-tom": "Zeptoseconds",
    "v-tom": "Attoseconds",
    "u-tom": "Femtoseconds",
    "t-tom": "Picoseconds",
    "s-tom": "Nanoseconds",
    "r-tom": "Microseconds",
    "q-tom": "Sub-seconds",
    "p-tom": "Seconds in Atomic Time",
    "o-tom": "Minutes in Atomic Time",
    "n-tom": "Minutes",
    "m-tom": "Hours",
    "l-tom": "Days",
    "k-tom": "Days",
    "j-tom": "Years in Atomic Time",
    "i-tom": "Civilizational Era",
    "h-tom": "Life Emergence",
    "g-tom": "Planetary Evolution",
    "f-tom": "Early Solar System",
    "e-tom": "Galactic Formation",
    "d-tom": "Early Universe",
    "c-tom": "Observable Cosmos",
    "b-tom": "Pre-Cosmos Phase",
    "a-tom": "Subatomic or Cosmic Boundary",
    "z-tom": "Fundamental Second",
}

# Conversion Factors and Time Units to Seconds (for calculation purposes)
CONVERSION_UNITS = [
    {"name": "y-tom", "factor": 2.704e-8, "unit_type": "seconds"},
    {"name": "x-tom", "factor": 2.704e-7, "unit_type": "seconds"},
    {"name": "w-tom", "factor": 2.704e-6, "unit_type": "seconds"},
    {"name": "v-tom", "factor": 2.704e-5, "unit_type": "seconds"},
    {"name": "u-tom", "factor": 0.0002704, "unit_type": "seconds"},
    {"name": "t-tom", "factor": 0.002704, "unit_type": "seconds"},
    {"name": "s-tom", "factor": 0.02704, "unit_type": "seconds"},
    {"name": "r-tom", "factor": 0.2704, "unit_type": "seconds"},
    {"name": "q-tom", "factor": 2.704, "unit_type": "seconds"},
    {"name": "p-tom", "factor": 27.04, "unit_type": "seconds"},
    {"name": "o-tom", "factor": 4.506, "unit_type": "minutes"},
    {"name": "n-tom", "factor": 45.06, "unit_type": "minutes"},
    {"name": "m-tom", "factor": 7.51, "unit_type": "hours"},
    {"name": "l-tom", "factor": 3.1296, "unit_type": "days"},
    {"name": "k-tom", "factor": 31.296, "unit_type": "days"},
    {"name": "j-tom", "factor": 0.8547, "unit_type": "years"},
    {"name": "i-tom", "factor": 8.547, "unit_type": "years"},
    {"name": "h-tom", "factor": 85.47, "unit_type": "years"},
    {"name": "g-tom", "factor": 427.35, "unit_type": "years"},
    {"name": "f-tom", "factor": 4273.5, "unit_type": "years"},
    {"name": "e-tom", "factor": 42735, "unit_type": "years"},
    {"name": "d-tom", "factor": 427350, "unit_type": "years"},
    {"name": "c-tom", "factor": 28e9, "unit_type": "years"},
    {"name": "b-tom", "factor": 28e10, "unit_type": "years"},
    {"name": "a-tom", "factor": 28e11, "unit_type": "years"},
    {"name": "z-tom", "factor": 1, "unit_type": "seconds"},
]

TIME_UNITS_TO_SECONDS = {
    "years": 31536000,
    "days": 86400,
    "hours": 3600,
    "minutes": 60,
    "seconds": 1,
}


# ------ Helper Function for Converting Time ------
class TimeConversion:
    def __init__(self, value, from_unit, to_unit):
        self.value = value
        self.from_unit = from_unit
        self.to_unit = to_unit

    def convert_to_seconds(self, unit):
        """Convert unit to seconds."""
        for unit_info in CONVERSION_UNITS:
            if unit_info["name"] == unit:
                return self.value * unit_info["factor"]
        return 0

    def convert(self):
        """Perform the conversion and return the result."""
        from_seconds = self.convert_to_seconds(self.from_unit)
        to_seconds = self.convert_to_seconds(self.to_unit)
        result = from_seconds / to_seconds
        return result


# ------ GUI Application ------
class TimeConversionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Universalism Time Converter")
        self.setStyleSheet("background-color: #0e0b47; color: white;")  # Dark mode with cosmic theme
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Cosmic Universalism Statement
        cosmic_label = QLabel("Welcome to the Cosmic Universalism Time Converter\n\n")
        cosmic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(cosmic_label)

        # Time Selector
        time_selector_layout = QHBoxLayout()
        self.time_selector = QSlider(Qt.Orientation.Horizontal)
        self.time_selector.setRange(1, 100)  # Allowing time to range from 1 to 100 seconds
        self.time_selector.setTickInterval(1)
        self.time_selector.setTickPosition(QSlider.TickPosition.TicksBelow)
        time_selector_layout.addWidget(QLabel("Select Time (seconds):"))
        time_selector_layout.addWidget(self.time_selector)
        layout.addLayout(time_selector_layout)

        # TOM Unit Selectors
        self.from_tom_unit = QComboBox()
        self.to_tom_unit = QComboBox()

        for unit in TIME_UNITS:
            self.from_tom_unit.addItem(TIME_UNITS[unit])
            self.to_tom_unit.addItem(TIME_UNITS[unit])

        layout.addWidget(QLabel("From TOM Unit (TOM Scale):"))
        layout.addWidget(self.from_tom_unit)
        layout.addWidget(QLabel("To TOM Unit (TOM Scale):"))
        layout.addWidget(self.to_tom_unit)

        # Friendly Time Unit Selectors
        self.from_friendly_unit = QComboBox()
        self.to_friendly_unit = QComboBox()

        for unit in TIME_UNITS:
            self.from_friendly_unit.addItem(unit)
            self.to_friendly_unit.addItem(unit)

        layout.addWidget(QLabel("From Friendly Unit (Human-Friendly):"))
        layout.addWidget(self.from_friendly_unit)
        layout.addWidget(QLabel("To Friendly Unit (Human-Friendly):"))
        layout.addWidget(self.to_friendly_unit)

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
            value = self.time_selector.value()
            from_tom_unit_name = self.from_tom_unit.currentText()
            to_tom_unit_name = self.to_tom_unit.currentText()
            from_friendly_unit_name = self.from_friendly_unit.currentText()
            to_friendly_unit_name = self.to_friendly_unit.currentText()

            # Map back to unit names
            from_tom_unit = next(unit for unit in TIME_UNITS if TIME_UNITS[unit] == from_tom_unit_name)
            to_tom_unit = next(unit for unit in TIME_UNITS if TIME_UNITS[unit] == to_tom_unit_name)

            # Friendly time conversion
            from_friendly_unit = from_friendly_unit_name
            to_friendly_unit = to_friendly_unit_name

            # Perform the conversion for TOM to seconds
            conversion_tom = TimeConversion(value, from_tom_unit, to_tom_unit)
            result_tom = conversion_tom.convert()

            # Perform the conversion for Friendly units to seconds
            conversion_friendly = TimeConversion(value, from_friendly_unit, to_friendly_unit)
            result_friendly = conversion_friendly.convert()

            # Display the results
            self.result_label.setText(
                f"{value} seconds in TOM = {result_tom:.4g} {to_tom_unit_name} (TOM)\n"
                f"or {value} seconds in Friendly Units = {result_friendly:.4g} {to_friendly_unit_name} (Human-Friendly)"
            )

        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimeConversionApp()
    window.show()
    sys.exit(app.exec())