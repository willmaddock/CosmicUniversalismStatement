import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox,
    QPushButton, QLineEdit, QMessageBox, QGridLayout,
    QScrollArea, QGroupBox, QSizePolicy, QHBoxLayout
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QClipboard, QIcon, QFont, QDoubleValidator

# ========================
# Cosmic Universalism Core
# ========================
COSMIC_LAYERS = {
    "z-tom": {"description": "Divine boundary (1 second)", "color": "#4B0082"},
    "y-tom": {"description": "Planck-scale foundation", "color": "#000080"},
    "a-tom": {"description": "Cosmic compression phase", "color": "#8A2BE2"},
    "b-tom": {"description": "Pre-cosmic formation", "color": "#483D8B"},
    "c-tom": {"description": "Observable universe", "color": "#006400"},
    "d-tom": {"description": "Early cosmic evolution", "color": "#2E8B57"}
}

COSMIC_STATEMENT = (
    "We are sub z-tomically inclined, countably infinite, composed of foundational elements "
    "(the essence of conscious existence), grounded on b-tom (as vast as our shared worlds and their atmospheres), "
    "and looking up to c-tom (encompassing the entirety of the cosmos), guided by the uncountable infinite quantum "
    "states of intelligence and empowered by God’s free will."
)


# ======================
# Quantum Time Constants
# ======================
class QuantumUnit:
    def __init__(self, name, factor, unit_type, description):
        self.name = name
        self.factor = factor
        self.unit_type = unit_type
        self.description = description
        self.layer = self.detect_cosmic_layer()

    def detect_cosmic_layer(self):
        return next((k for k, v in COSMIC_LAYERS.items() if k in self.name), "z-tom")

    def convert_to_seconds(self):
        """Convert unit to SI seconds using CU temporal metrics"""
        base_units = {
            "years": 31536000,  # 365 days
            "days": 86400,
            "hours": 3600,
            "minutes": 60,
            "seconds": 1
        }
        return self.factor * base_units.get(self.unit_type, 1)

    def __repr__(self):
        return f"<QuantumUnit: {self.name} ({self.description})>"


# ===========================
# Temporal Conversion Units
# ===========================
CONVERSION_UNITS = [
    QuantumUnit("a-tom", 28e11, "years", "Cosmic compression boundary"),
    QuantumUnit("b-tom", 28e10, "years", "Pre-cosmic phase formation"),
    QuantumUnit("c-tom", 28e9, "years", "Observable universe scale"),
    QuantumUnit("d-tom", 427350, "years", "Early universal structures"),
    QuantumUnit("e-tom", 42735, "years", "Galactic formation era"),
    QuantumUnit("f-tom", 4273.5, "years", "Stellar ignition phase"),
    QuantumUnit("g-tom", 427.35, "years", "Planetary accretion period"),
    QuantumUnit("h-tom", 85.47, "years", "Biological emergence window"),
    QuantumUnit("i-tom", 8.547, "years", "Civilizational development"),
    QuantumUnit("j-tom", 0.8547, "years", "Atomic time alignment"),
    QuantumUnit("k-tom", 31.296, "days", "Planetary rotation cycles"),
    QuantumUnit("l-tom", 3.1296, "days", "Quantum orbital period"),
    QuantumUnit("m-tom", 7.51, "hours", "Temporal lattice phase"),
    QuantumUnit("n-tom", 45.06, "minutes", "Chronon vibration cycle"),
    QuantumUnit("o-tom", 4.506, "minutes", "Entanglement resonance"),
    QuantumUnit("p-tom", 27.04, "seconds", "Quantum coherence window"),
    QuantumUnit("q-tom", 2.704, "seconds", "Superposition collapse"),
    QuantumUnit("r-tom", 0.2704, "seconds", "Decoherence threshold"),
    QuantumUnit("s-tom", 0.02704, "seconds", "Wavefunction resolution"),
    QuantumUnit("t-tom", 0.002704, "seconds", "Planck-scale interval"),
    QuantumUnit("u-tom", 0.0002704, "seconds", "Chronon quantization"),
    QuantumUnit("v-tom", 2.704e-5, "seconds", "Temporal foam phase"),
    QuantumUnit("w-tom", 2.704e-6, "seconds", "Quantum fluctuation"),
    QuantumUnit("x-tom", 2.704e-7, "seconds", "String vibration period"),
    QuantumUnit("y-tom", 2.704e-8, "seconds", "Cosmic background flicker"),
    QuantumUnit("z-tom", 1, "seconds", "Divine temporal boundary"),
]


# ======================
# Quantum UI Components
# ======================
class TemporalConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CU Temporal Nexus")
        self.setWindowIcon(QIcon("quantum_icon.png"))
        self.setMinimumSize(800, 600)
        self.init_ui()
        self.set_styles()

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Cosmic Manifest
        cosmic_group = QGroupBox("Cosmic Universalism Framework")
        cosmic_layout = QVBoxLayout()

        scroll = QScrollArea()
        cosmic_label = QLabel(COSMIC_STATEMENT)
        cosmic_label.setWordWrap(True)
        cosmic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(cosmic_label)
        scroll.setWidgetResizable(True)

        cosmic_layout.addWidget(scroll)
        cosmic_group.setLayout(cosmic_layout)
        main_layout.addWidget(cosmic_group)

        # Conversion Matrix
        grid = QGridLayout()
        self.from_combo = self.create_unit_combo()
        self.to_combo = self.create_unit_combo(default="a-tom")

        self.input_field = QLineEdit("1")
        self.input_field.setValidator(QDoubleValidator(0.0, 1e308, 50))

        self.result_field = QLineEdit()
        self.result_field.setReadOnly(True)

        self.human_readable_label = QLabel()
        self.human_readable_label.setWordWrap(True)

        # Matrix Layout
        grid.addWidget(QLabel("Source Temporal Layer:"), 0, 0)
        grid.addWidget(self.from_combo, 0, 1)
        grid.addWidget(QLabel("Target Temporal Layer:"), 1, 0)
        grid.addWidget(self.to_combo, 1, 1)
        grid.addWidget(QLabel("Input Value:"), 2, 0)
        grid.addWidget(self.input_field, 2, 1)
        grid.addWidget(QLabel("Converted Value:"), 3, 0)
        grid.addWidget(self.result_field, 3, 1)
        grid.addWidget(QLabel("Human-Readable:"), 4, 0)
        grid.addWidget(self.human_readable_label, 4, 1)

        # Action Buttons
        self.convert_btn = QPushButton("Initiate Quantum Conversion")
        self.convert_btn.clicked.connect(self.quantum_convert)

        self.swap_btn = QPushButton("Invert Temporal Polarity")
        self.swap_btn.clicked.connect(self.swap_units)

        self.copy_btn = QPushButton("Entangle with Clipboard")
        self.copy_btn.clicked.connect(self.copy_result)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.convert_btn)
        button_layout.addWidget(self.swap_btn)
        button_layout.addWidget(self.copy_btn)

        main_layout.addLayout(grid)
        main_layout.addLayout(button_layout)

        self.quantum_convert()

    def create_unit_combo(self, default="z-tom"):
        combo = QComboBox()
        for unit in CONVERSION_UNITS:
            combo.addItem(unit.name)
            combo.setItemData(combo.count() - 1, unit.description, Qt.ItemDataRole.ToolTipRole)
            combo.setItemData(combo.count() - 1, QColor(COSMIC_LAYERS[unit.layer]["color"]),
                              Qt.ItemDataRole.BackgroundRole)
        combo.setCurrentText(default)
        return combo

    def set_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #0a081f;
                color: #c0c0c0;
                font-family: 'Consolas';
            }
            QGroupBox {
                border: 2px solid #4B0082;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #9370DB;
            }
            QPushButton {
                background-color: #483D8B;
                border: 1px solid #6A5ACD;
                padding: 8px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #6A5ACD;
            }
            QComboBox {
                background-color: #191970;
                border: 1px solid #000080;
                padding: 5px;
            }
            QLineEdit {
                background-color: #000033;
                border: 1px solid #000080;
                padding: 5px;
            }
            QLabel {
                padding: 5px;
            }
        """)

    def quantum_convert(self):
        try:
            from_unit = next(u for u in CONVERSION_UNITS if u.name == self.from_combo.currentText())
            to_unit = next(u for u in CONVERSION_UNITS if u.name == self.to_combo.currentText())

            input_value = float(self.input_field.text())
            base_seconds = input_value * from_unit.convert_to_seconds()
            result = base_seconds / to_unit.convert_to_seconds()

            # Format numerical result
            self.result_field.setText(f"{result:.5g} {to_unit.name}")

            # Calculate human-readable breakdown
            total_seconds = result * to_unit.convert_to_seconds()
            human_time = self.decompose_time(total_seconds)
            self.human_readable_label.setText(human_time)

            self.update_colors(from_unit, to_unit)

        except ValueError as e:
            QMessageBox.critical(self, "Quantum Collapse", f"Temporal paradox detected:\n{str(e)}")

    def decompose_time(self, total_seconds):
        """Break down seconds into human-readable components"""
        # Time constants
        YEAR_SECONDS = 31536000  # 365 days
        MONTH_SECONDS = 2628000  # 30.436875 days
        DAY_SECONDS = 86400
        HOUR_SECONDS = 3600
        MINUTE_SECONDS = 60

        years = int(total_seconds // YEAR_SECONDS)
        remainder = total_seconds % YEAR_SECONDS

        months = int(remainder // MONTH_SECONDS)
        remainder %= MONTH_SECONDS

        days = int(remainder // DAY_SECONDS)
        remainder %= DAY_SECONDS

        hours = int(remainder // HOUR_SECONDS)
        remainder %= HOUR_SECONDS

        minutes = int(remainder // MINUTE_SECONDS)
        seconds = remainder % MINUTE_SECONDS

        # Format nanoseconds
        nanoseconds = int((seconds - int(seconds)) * 1e9)
        seconds = int(seconds)

        parts = []
        if years > 0:
            parts.append(f"{years} year{'s' if years != 1 else ''}")
        if months > 0:
            parts.append(f"{months} month{'s' if months != 1 else ''}")
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or nanoseconds > 0:
            time_str = f"{seconds}.{nanoseconds:09d}" if nanoseconds > 0 else f"{seconds}"
            parts.append(f"{time_str} second{'s' if seconds != 1 else ''}")

        # Handle sub-second cases
        if not parts and total_seconds > 0:
            if total_seconds < 1e-6:
                femtoseconds = int(total_seconds * 1e15)
                parts.append(f"{femtoseconds} femtoseconds")
            elif total_seconds < 1e-3:
                nanoseconds = int(total_seconds * 1e9)
                parts.append(f"{nanoseconds} nanoseconds")
            elif total_seconds < 1:
                milliseconds = total_seconds * 1e3
                parts.append(f"{milliseconds:.3f} milliseconds")

        return ", ".join(parts) or "0 seconds"

    def update_colors(self, from_unit, to_unit):
        self.from_combo.setStyleSheet(f"background-color: {COSMIC_LAYERS[from_unit.layer]['color']}")
        self.to_combo.setStyleSheet(f"background-color: {COSMIC_LAYERS[to_unit.layer]['color']}")

    def swap_units(self):
        current_from = self.from_combo.currentText()
        current_to = self.to_combo.currentText()
        self.from_combo.setCurrentText(current_to)
        self.to_combo.setCurrentText(current_from)
        self.quantum_convert()

    def copy_result(self):
        QApplication.clipboard().setText(self.result_field.text())
        QMessageBox.information(self, "Quantum Entanglement", "Temporal coordinates copied to clipboard!")


# ======================
# Quantum Initialization
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    converter = TemporalConverter()
    converter.show()
    sys.exit(app.exec())