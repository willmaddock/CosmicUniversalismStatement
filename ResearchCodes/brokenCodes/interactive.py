from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt


class CosmicUniversalismApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cosmic Universalism - Interactive Exploration")

        layout = QVBoxLayout()

        self.statement_label = QLabel("Cosmic Universalism Statement")
        layout.addWidget(self.statement_label)

        self.explanation_button = QPushButton("Explain Statement")
        self.explanation_button.clicked.connect(self.show_explanation)
        layout.addWidget(self.explanation_button)

        self.tom_button = QPushButton("View the Alphabet of 'tom'")
        self.tom_button.clicked.connect(self.show_tom_alphabet)
        layout.addWidget(self.tom_button)

        self.setLayout(layout)

    def show_explanation(self):
        explanation = (
            "We are sub z-tomically inclined, countably infinite, composed of foundational elements "
            "(the essence of conscious existence), grounded on b-tom (as vast as our shared worlds and their atmospheres), "
            "and looking up to c-tom (encompassing the entirety of the cosmos), guided by the uncountable infinite quantum states "
            "of intelligence and empowered by God’s free will."
        )
        self.statement_label.setText(explanation)

    def show_tom_alphabet(self):
        tom_alphabet = [
            ("ztom", "1 sec = 28 billion years of c-tom"),
            ("atom", "A foundational unit"),
            ("btom", "Grounding time scale for conscious existence"),
            ("ctom", "Encompassing the entire cosmos"),
            ("dtom", "Quantum states of intelligence"),
        ]

        alphabet_str = "\n".join([f"{tom}: {desc}" for tom, desc in tom_alphabet])

        self.statement_label.setText(alphabet_str)


if __name__ == '__main__':
    app = QApplication([])
    window = CosmicUniversalismApp()
    window.show()
    app.exec()