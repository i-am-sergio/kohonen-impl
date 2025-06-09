#include <vector>
#include <iostream>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "Reader.hpp"

using namespace std;

// Definición e implementación de la clase MNISTViewer
class MNISTViewer {
public:
    // Puntero estático a la instancia actual de la clase.
    // Necesario porque FreeGLUT requiere funciones de callback estáticas.
    static MNISTViewer * s_instance;

    // Constructor de la clase: ahora acepta un vector de imágenes
    MNISTViewer(const vector<vector<float>>& allImageData, int width, int height)
        : m_allImageData(allImageData),
          m_imageWidth(width),
          m_imageHeight(height),
          m_currentImageIndex(0), // Empieza con la primera imagen
          m_cameraDistance(3.0f),
          m_rotationX(0.0f),
          m_rotationY(0.0f),
          m_lastMouseX(0),
          m_lastMouseY(0),
          m_isDragging(false) {
        // Establecer la instancia estática a este objeto
        s_instance = this;
    }

    // Método para inicializar FreeGLUT y ejecutar el bucle principal
    void run(int argc, char** argv) {
        // Inicialización de FreeGLUT
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // Añadir GLUT_DEPTH
        glutInitWindowSize(600, 600); // Un tamaño de ventana más grande
        glutCreateWindow("MNIST 3D Viewer (Multiple Images)");

        // Inicializa OpenGL y carga las texturas
        initGL();
        loadAllTextures(); // Cargar todas las imágenes como texturas

        // Registrar funciones de callback estáticas
        glutDisplayFunc(displayCallback);
        glutReshapeFunc(reshapeCallback);
        glutMotionFunc(mouseMotionCallback);
        glutMouseFunc(mouseButtonCallback);
        // Ahora usamos glutSpecialFunc para las flechas
        glutSpecialFunc(specialKeyboardCallback);
        // Si aún quieres manejar otras teclas como ESC, mantén glutKeyboardFunc
        glutKeyboardFunc(keyboardCallback); 
        glutIdleFunc(idleCallback); // Hace que la esfera gire lentamente
        // Inicia el bucle principal de FreeGLUT
        glutMainLoop();

        // Limpieza de las texturas al salir del bucle principal
        if (!m_textureIDs.empty()) {
            glDeleteTextures(m_textureIDs.size(), m_textureIDs.data());
        }
    }

private:
    // Datos de TODAS las imágenes MNIST
    vector<vector<float>> m_allImageData;
    int m_imageWidth;
    int m_imageHeight;
    int m_currentImageIndex; // Índice de la imagen que se está mostrando actualmente

    // IDs de las texturas de OpenGL (un ID por cada imagen)
    vector<GLuint> m_textureIDs;

    // Variables para el control de la cámara y la rotación
    float m_cameraDistance;
    float m_rotationX;
    float m_rotationY;
    int m_lastMouseX, m_lastMouseY;
    bool m_isDragging;

    float m_autoRotationAngle = 0.0f;


    void actualIdle() {
        m_autoRotationAngle += 0.2f; // velocidad del giro
        if (m_autoRotationAngle >= 360.0f) m_autoRotationAngle -= 360.0f;
        glutPostRedisplay();
    }

    static void idleCallback() {
        if (s_instance) s_instance->actualIdle();
    }



    // Métodos internos para la funcionalidad de OpenGL
    // Carga TODAS las imágenes en texturas
    void loadAllTextures() {
        if (m_allImageData.empty()) {
            cerr << "Error: No hay datos de imagen MNIST cargados para crear texturas." << endl;
            return;
        }

        m_textureIDs.resize(m_allImageData.size());
        glGenTextures(m_allImageData.size(), m_textureIDs.data());

        for (size_t i = 0; i < m_allImageData.size(); ++i) {
            glBindTexture(GL_TEXTURE_2D, m_textureIDs[i]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // Preparar datos para la textura.
            // Convertir float [0,1] a byte [0,255].
            vector<GLubyte> textureData(m_imageWidth * m_imageHeight);
            for (int j = 0; j < m_imageWidth * m_imageHeight; ++j) {
                textureData[j] = static_cast<GLubyte>(m_allImageData[i][j] * 255.0f);
            }

            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, m_imageWidth, m_imageHeight, 0,
                         GL_LUMINANCE, GL_UNSIGNED_BYTE, textureData.data());
        }
        glBindTexture(GL_TEXTURE_2D, 0); // Desvincular al final
    }

    void initGL() {
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glShadeModel(GL_SMOOTH);

        // Configuración básica de iluminación
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        GLfloat light_position[] = { 1.0f, 1.0f, 1.0f, 0.0f };
        GLfloat ambient_light[] = { 0.3f, 0.3f, 0.3f, 1.0f };
        GLfloat diffuse_light[] = { 0.7f, 0.7f, 0.7f, 1.0f };
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light);
        glEnable(GL_COLOR_MATERIAL);
    }

    // Métodos de instancia que contienen la lógica OpenGL
    void actualDisplay() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        gluLookAt(0.0, 0.0, m_cameraDistance,
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0);

        glRotatef(m_rotationX, 1.0f, 0.0f, 0.0f);
        glRotatef(m_rotationY, 0.0f, 1.0f, 0.0f);

        // Dibuja la esfera semitransparente girando lentamente
        glPushMatrix();
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDisable(GL_LIGHTING); // para que el alambre se vea más claro

            glColor4f(0.3f, 0.5f, 1.0f, 0.2f); // Color azul translúcido

            glRotatef(m_autoRotationAngle, 0.0f, 1.0f, 0.0f); // Giro automático

            glutWireSphere(1.2f, 40, 40); // Radio y resolución de la esfera

            glEnable(GL_LIGHTING);
            glDisable(GL_BLEND);
        glPopMatrix();

        // Dibuja el plano con la textura actual
        glEnable(GL_TEXTURE_2D);
        if (m_currentImageIndex >= 0 && m_currentImageIndex < m_textureIDs.size()) {
            glBindTexture(GL_TEXTURE_2D, m_textureIDs[m_currentImageIndex]);
        } else {
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        glBegin(GL_QUADS);
            glNormal3f(0.0f, 0.0f, 1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex3f(-0.5f, -0.5f, 0.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex3f( 0.5f, -0.5f, 0.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex3f( 0.5f,  0.5f, 0.0f);
            glTexCoord2f(0.0f, 0.0f); glVertex3f(-0.5f,  0.5f, 0.0f);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        glutSwapBuffers();
    }


    void actualReshape(int width, int height) {
        if (height == 0) height = 1;
        float aspect = (float)width / (float)height;

        glViewport(0, 0, width, height);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        gluPerspective(45.0f, aspect, 0.1f, 100.0f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    void actualMouseMotion(int x, int y) {
        if (m_isDragging) {
            m_rotationY += (x - m_lastMouseX) * 0.5f;
            m_rotationX += (y - m_lastMouseY) * 0.5f;
            m_lastMouseX = x;
            m_lastMouseY = y;
            glutPostRedisplay();
        }
    }

    void actualMouseButton(int button, int state, int x, int y) {
        if (button == GLUT_LEFT_BUTTON) {
            if (state == GLUT_DOWN) {
                m_lastMouseX = x;
                m_lastMouseY = y;
                m_isDragging = true;
            } else { // GLUT_UP
                m_isDragging = false;
            }
        }
        else if (button == 3) { // Rueda hacia arriba (zoom in)
            m_cameraDistance -= 0.1f;
            if (m_cameraDistance < 0.5f) m_cameraDistance = 0.5f;
            glutPostRedisplay();
        }
        else if (button == 4) { // Rueda hacia abajo (zoom out)
            m_cameraDistance += 0.1f;
            if (m_cameraDistance > 10.0f) m_cameraDistance = 10.0f;
            glutPostRedisplay();
        }
    }

    // Manejo de teclas normales (para ESC, por ejemplo)
    void actualKeyboard(unsigned char key, int x, int y) {
        switch (key) {
            case 27: // Tecla ESC para salir
                glutLeaveMainLoop();
                break;
        }
    }

    // Nuevo: manejo de teclas especiales (flechas) para cambiar de imagen
    void actualSpecialKeyboard(int key, int x, int y) {
        switch (key) {
            case GLUT_KEY_RIGHT: // Flecha derecha para siguiente imagen
                m_currentImageIndex = (m_currentImageIndex + 1) % m_allImageData.size();
                cout << "Mostrando imagen: " << m_currentImageIndex << endl;
                glutPostRedisplay();
                break;
            case GLUT_KEY_LEFT: // Flecha izquierda para imagen previa
                m_currentImageIndex = (m_currentImageIndex - 1 + m_allImageData.size()) % m_allImageData.size();
                cout << "Mostrando imagen: " << m_currentImageIndex << endl;
                glutPostRedisplay();
                break;
        }
    }

    // Métodos estáticos de callback para FreeGLUT
    static void displayCallback() {
        if (s_instance) s_instance->actualDisplay();
    }
    static void reshapeCallback(int width, int height) {
        if (s_instance) s_instance->actualReshape(width, height);
    }
    static void mouseMotionCallback(int x, int y) {
        if (s_instance) s_instance->actualMouseMotion(x, y);
    }
    static void mouseButtonCallback(int button, int state, int x, int y) {
        if (s_instance) s_instance->actualMouseButton(button, state, x, y);
    }
    static void keyboardCallback(unsigned char key, int x, int y) {
        if (s_instance) s_instance->actualKeyboard(key, x, y);
    }
    static void specialKeyboardCallback(int key, int x, int y) { // Nuevo callback estático
        if (s_instance) s_instance->actualSpecialKeyboard(key, x, y);
    }
};

// Inicialización del miembro estático
// Debe estar fuera de la definición de la clase en un archivo .cpp,
// pero para una solución de un solo .hpp, se coloca aquí.
MNISTViewer * MNISTViewer::s_instance = nullptr;


int main(int argc, char** argv) {
    // Cargar datos de entrenamiento
    vector<vector<float>> raw_X_train;
    vector<vector<float>> raw_Y_train;

    // Carga TODAS las imágenes de entrenamiento (60000)
    Reader::load_csv("../topicos-inteligencia-artificial/datasets/MNISTdataset/mnist_test_flat.csv", raw_X_train, raw_Y_train, 60000);

    cout << "Nro de X de entrenamiento: " << raw_X_train.size() << endl;
    cout << "Nro de Y de entrenamiento: " << raw_Y_train.size() << endl;

    if (raw_X_train.empty()) {
        cerr << "No se cargaron datos de entrenamiento. Saliendo." << endl;
        return 1;
    }

    // Crear una instancia de la clase MNISTViewer, pasando TODAS las imágenes
    MNISTViewer viewer(raw_X_train, 28, 28);

    // Ejecutar el visualizador
    viewer.run(argc, argv);

    return 0;
}