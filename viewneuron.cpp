#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <cmath>
#include "include/Reader.hpp"

// ViewNeuron class to display a single MNIST image
class ViewNeuron {
public:
    static ViewNeuron* s_instance;
    ViewNeuron(const std::vector<std::vector<float>>& allImageData, int width, int height)
        : m_allImageData(allImageData),
          m_imageWidth(width),
          m_imageHeight(height),
          m_cameraDistance(3.0f),
          m_rotationX(0.0f),
          m_rotationY(0.0f),
          m_autoRotationAngle(0.0f),
          m_lastMouseX(0),
          m_lastMouseY(0),
          m_isDragging(false),
          m_textureID(0) {
        s_instance = this;
    }

    void run(int argc, char** argv) {
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(600, 600);
        glutCreateWindow("MNIST Neuron Viewer");
        initGL();
        loadTexture();
        glutDisplayFunc(displayCallback);
        glutReshapeFunc(reshapeCallback);
        glutMotionFunc(mouseMotionCallback);
        glutMouseFunc(mouseButtonCallback);
        glutIdleFunc(idleCallback);
        glutMainLoop();
        if (m_textureID != 0) {
            glDeleteTextures(1, &m_textureID);
        }
    }

private:
    const std::vector<std::vector<float>>& m_allImageData;
    int m_imageWidth;
    int m_imageHeight;
    GLuint m_textureID;
    float m_cameraDistance;
    float m_rotationX;
    float m_rotationY;
    float m_autoRotationAngle;
    int m_lastMouseX, m_lastMouseY;
    bool m_isDragging;

    void initGL() {
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glShadeModel(GL_SMOOTH);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        GLfloat light_position[] = {1.0f, 1.0f, 1.0f, 0.0f};
        GLfloat ambient_light[] = {0.3f, 0.3f, 0.3f, 1.0f};
        GLfloat diffuse_light[] = {0.7f, 0.7f, 0.7f, 1.0f};
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light);
        glEnable(GL_COLOR_MATERIAL);
    }

    void loadTexture() {
        if (m_allImageData.empty()) {
            std::cerr << "Error: No MNIST image data loaded." << std::endl;
            return;
        }
        glGenTextures(1, &m_textureID);
        glBindTexture(GL_TEXTURE_2D, m_textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        std::vector<GLubyte> textureData(m_imageWidth * m_imageHeight);
        for (int i = 0; i < m_imageWidth * m_imageHeight; ++i) {
            textureData[i] = static_cast<GLubyte>(m_allImageData[0][i] * 255.0f);
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, m_imageWidth, m_imageHeight, 0,
                     GL_LUMINANCE, GL_UNSIGNED_BYTE, textureData.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void actualDisplay() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        gluLookAt(0.0, 0.0, m_cameraDistance,
                  0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0);
        glRotatef(m_rotationX, 1.0f, 0.0f, 0.0f);
        glRotatef(m_rotationY, 0.0f, 1.0f, 0.0f);

        // Draw semi-transparent sphere
        glPushMatrix();
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_LIGHTING);
        glColor4f(0.3f, 0.5f, 1.0f, 0.2f);
        glRotatef(m_autoRotationAngle, 0.0f, 1.0f, 0.0f);
        glutWireSphere(1.2f, 40, 40);
        glEnable(GL_LIGHTING);
        glDisable(GL_BLEND);
        glPopMatrix();

        // Draw textured plane
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, m_textureID);
        glBegin(GL_QUADS);
        glNormal3f(0.0f, 0.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex3f(-0.5f, -0.5f, 0.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f(0.5f, -0.5f, 0.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex3f(0.5f, 0.5f, 0.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex3f(-0.5f, 0.5f, 0.0f);
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
            } else {
                m_isDragging = false;
            }
        } else if (button == 3) { // Scroll up (zoom in)
            m_cameraDistance -= 0.1f;
            if (m_cameraDistance < 0.5f) m_cameraDistance = 0.5f;
            glutPostRedisplay();
        } else if (button == 4) { // Scroll down (zoom out)
            m_cameraDistance += 0.1f;
            if (m_cameraDistance > 10.0f) m_cameraDistance = 10.0f;
            glutPostRedisplay();
        }
    }

    void actualIdle() {
        m_autoRotationAngle += 0.2f;
        if (m_autoRotationAngle >= 360.0f) m_autoRotationAngle -= 360.0f;
        glutPostRedisplay();
    }

    static void displayCallback() { if (s_instance) s_instance->actualDisplay(); }
    static void reshapeCallback(int width, int height) { if (s_instance) s_instance->actualReshape(width, height); }
    static void mouseMotionCallback(int x, int y) { if (s_instance) s_instance->actualMouseMotion(x, y); }
    static void mouseButtonCallback(int button, int state, int x, int y) { if (s_instance) s_instance->actualMouseButton(button, state, x, y); }
    static void idleCallback() { if (s_instance) s_instance->actualIdle(); }
};

// Static member initialization
ViewNeuron* ViewNeuron::s_instance = nullptr;

// Main function
int main(int argc, char** argv) {
    std::vector<std::vector<float>> raw_X_train;
    std::vector<std::vector<float>> raw_Y_train;
    Reader::load_csv("database/mnist_train_flat_3.csv", raw_X_train, raw_Y_train, 60000);
    std::cout << "Nro de X de entrenamiento: " << raw_X_train.size() << std::endl;
    std::cout << "Nro de Y de entrenamiento: " << raw_Y_train.size() << std::endl;

    ViewNeuron viewer(raw_X_train, 28, 28);
    viewer.run(argc, argv);
    return 0;
}