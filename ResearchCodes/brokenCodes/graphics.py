import colorsys
import sys
import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSlider, QVBoxLayout,
                             QWidget, QHBoxLayout, QLabel)
from PyQt6.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PyQt6.QtGui import QSurfaceFormat, QMatrix4x4
from PyQt6.QtCore import *

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.angle_x = 30
        self.angle_y = 30
        self.level = 1
        self.zoom = -6
        self.last_pos = None

        # Sphere properties
        self.sphere_vbo = None
        self.sphere_vao = None
        self.sphere_vertex_count = 0
        self.sphere_resolution = 32  # Higher resolution for smoother spheres

        # Shaders
        self.program = None
        self.vertex_shader = None
        self.fragment_shader = None

        # Initialize timer for smooth animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)

        # Initialize shaders
        self.init_shaders()

        # Create sphere geometry
        self.create_sphere_vbo()

    def init_shaders(self):
        vertex_shader_source = """
        #version 330 core
        layout(location = 0) in vec3 vertexPosition;
        layout(location = 1) in vec3 vertexNormal;

        out vec3 fragNormal;
        out vec3 fragPosition;

        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;

        void main() {
            fragPosition = vec3(modelViewMatrix * vec4(vertexPosition, 1.0));
            fragNormal = mat3(modelViewMatrix) * vertexNormal;
            gl_Position = projectionMatrix * vec4(fragPosition, 1.0);
        }
        """

        fragment_shader_source = """
        #version 330 core
        in vec3 fragNormal;
        in vec3 fragPosition;

        out vec4 FragColor;

        uniform vec3 lightPosition;
        uniform vec3 lightColor;
        uniform vec3 objectColor;

        void main() {
            float ambientStrength = 0.1;
            vec3 ambient = ambientStrength * lightColor;

            vec3 norm = normalize(fragNormal);
            vec3 lightDir = normalize(lightPosition - fragPosition);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            vec3 result = (ambient + diffuse) * objectColor;
            FragColor = vec4(result, 1.0);
        }
        """

        # Compile vertex shader
        self.vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vertex_shader, vertex_shader_source)
        glCompileShader(self.vertex_shader)
        if not glGetShaderiv(self.vertex_shader, GL_COMPILE_STATUS):
            raise RuntimeError("Vertex shader compilation failed: " + str(glGetShaderInfoLog(self.vertex_shader)))

        # Compile fragment shader
        self.fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fragment_shader, fragment_shader_source)
        glCompileShader(self.fragment_shader)
        if not glGetShaderiv(self.fragment_shader, GL_COMPILE_STATUS):
            raise RuntimeError("Fragment shader compilation failed: " + str(glGetShaderInfoLog(self.fragment_shader)))

        # Create shader program
        self.program = glCreateProgram()
        glAttachShader(self.program, self.vertex_shader)
        glAttachShader(self.program, self.fragment_shader)
        glLinkProgram(self.program)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            raise RuntimeError("Program linking failed: " + str(glGetProgramInfoLog(self.program)))
        glUseProgram(self.program)

    def create_sphere_vbo(self):
        """Generate a sphere using vertices and normals for lighting."""
        vertices = []
        normals = []

        for i in range(self.sphere_resolution + 1):
            lat = np.pi * (-0.5 + float(i) / self.sphere_resolution)
            for j in range(self.sphere_resolution + 1):
                lon = 2 * np.pi * float(j) / self.sphere_resolution
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                vertices.extend([x, y, z])
                normals.extend([x, y, z])

        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)

        # Create VAO and VBOs
        self.sphere_vao = glGenVertexArrays(1)
        self.sphere_vbo = glGenBuffers(2)

        glBindVertexArray(self.sphere_vao)

        # Vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo[0])
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo[1])
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

        self.sphere_vertex_count = len(vertices) // 3

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Activate the shader program first
        glUseProgram(self.program)

        # Set up matrices
        model_view_matrix = QMatrix4x4()
        model_view_matrix.translate(0, 0, self.zoom)
        model_view_matrix.rotate(self.angle_x, 1, 0, 0)
        model_view_matrix.rotate(self.angle_y, 0, 1, 0)

        projection_matrix = QMatrix4x4()
        projection_matrix.perspective(45.0, self.width() / self.height(), 0.1, 100.0)

        # Get uniform locations after binding program
        mv_loc = glGetUniformLocation(self.program, "modelViewMatrix")
        proj_loc = glGetUniformLocation(self.program, "projectionMatrix")

        # Upload matrices with transpose=False (QMatrix4x4 is column-major)
        glUniformMatrix4fv(mv_loc, 1, GL_FALSE, np.array(model_view_matrix.data(), dtype=np.float32))
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, np.array(projection_matrix.data(), dtype=np.float32))

        # Set light properties
        glUniform3f(glGetUniformLocation(self.program, "lightPosition"), 5.0, 5.0, 5.0)
        glUniform3f(glGetUniformLocation(self.program, "lightColor"), 1.0, 1.0, 1.0)

        # Draw hierarchy
        self.draw_quantum_hierarchy(0, 0, 0, 0)

    def draw_quantum_hierarchy(self, x, y, z, depth):
        if depth > self.level:
            return

        matrix = QMatrix4x4()
        matrix.translate(x, y, z)

        # Set object color based on depth
        hue = depth / 5.0
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(self.program, "objectColor"), *rgb)

        # Update model matrix
        glUniformMatrix4fv(
            glGetUniformLocation(self.program, "modelViewMatrix"),
            1, GL_FALSE,
            matrix.data().tobytes()  # Use the local transformation matrix
        )

        # Draw sphere
        glBindVertexArray(self.sphere_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, self.sphere_vertex_count)
        glBindVertexArray(0)

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
            for dy in [-0.5, 0.0, 0.5]:  # Allow 0 for additional depth control
                for dz in [-0.5, 0.5]:
                    self.draw_quantum_hierarchy(dx, dy, dz, depth + 1)

    def mousePressEvent(self, event):
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if event.buttons() & Qt.LeftButton:
            self.angle_x += dy * 0.5
            self.angle_y += dx * 0.5
            self.update()

        self.last_pos = event.position()

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
        self.level_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_slider.setRange(1, 5)
        self.level_slider.setValue(1)
        self.level_slider.valueChanged.connect(self.glWidget.set_level)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
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


def set_opengl_version():
    # Set OpenGL version 3.3
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)  # Request OpenGL 3.3 or higher
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)


if __name__ == "__main__":
    set_opengl_version()  # Set the OpenGL version before creating the app
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())