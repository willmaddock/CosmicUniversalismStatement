import colorsys
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSlider, QVBoxLayout,
                             QWidget, QPushButton, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class OpenGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.angle_x = 30
        self.angle_y = 30
        self.level = 1
        self.zoom = -6
        self.last_pos = None
        self.sphere_quad = gluNewQuadric()

        # Initialize timer for smooth animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.2, 0.2, 0.2, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialf(GL_FRONT, GL_SHININESS, 50)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 0.1, 100.0)
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.angle_x, 1, 0, 0)
        glRotatef(self.angle_y, 0, 1, 0)

        self.draw_quantum_hierarchy(0, 0, 0, 0)

    def draw_quantum_hierarchy(self, x, y, z, depth):
        if depth > self.level:
            return

        glPushMatrix()
        glTranslatef(x, y, z)

        # Color based on depth using HSV to RGB conversion
        hue = depth / 5.0
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        glColor3f(*rgb)

        # Draw sphere with lighting
        gluSphere(self.sphere_quad, 0.2, 16, 16)

        # Draw connections to child nodes
        if depth < self.level:
            glBegin(GL_LINES)
            for dx in [-0.5, 0.5]:
                for dy in [-0.5, 0.5]:
                    for dz in [-0.5, 0.5]:
                        glVertex3f(0, 0, 0)
                        glVertex3f(dx, dy, dz)
            glEnd()

        # Recursively draw child nodes
        for dx in [-0.5, 0.5]:
            for dy in [-0.5, 0.5]:
                for dz in [-0.5, 0.5]:
                    self.draw_quantum_hierarchy(dx, dy, dz, depth + 1)

        glPopMatrix()

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if event.buttons() & Qt.LeftButton:
            self.angle_x += dy * 0.5
            self.angle_y += dx * 0.5
            self.update()

        self.last_pos = event.pos()

    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() * 0.01
        self.update()

    def set_level(self, value):
        self.level = min(value, 5)  # Limit recursion depth for performance
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Hierarchy Visualizer")
        self.setGeometry(100, 100, 800, 600)

        self.glWidget = OpenGLWidget()

        # Create UI controls
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(1, 5)
        self.level_slider.setValue(1)
        self.level_slider.valueChanged.connect(self.glWidget.set_level)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(-20, -3)
        self.zoom_slider.setValue(-6)
        self.zoom_slider.valueChanged.connect(lambda v: setattr(self.glWidget, 'zoom', v))

        # Info labels
        self.level_label = QLabel("Quantum Level: 1")
        self.zoom_label = QLabel("Zoom Level: -6")

        # Control layout
        control_layout = QVBoxLayout()
        control_layout.addWidget(QLabel("Quantum Level:"))
        control_layout.addWidget(self.level_slider)
        control_layout.addWidget(self.level_label)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Zoom:"))
        control_layout.addWidget(self.zoom_slider)
        control_layout.addWidget(self.zoom_label)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.glWidget, 1)
        main_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Update labels when values change
        self.level_slider.valueChanged.connect(lambda v: self.level_label.setText(f"Quantum Level: {v}"))
        self.zoom_slider.valueChanged.connect(lambda v: self.zoom_label.setText(f"Zoom Level: {v}"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())