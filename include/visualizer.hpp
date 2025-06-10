#pragma once
#include <vector>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <cmath>
#include "neuron.hpp"

class Visualizer
{
private:
    static inline std::vector<Neuron> *neurons = nullptr;
    static inline int dim_x = 10, dim_y = 10, dim_z = 10;
    static inline float angleX = 0.0f, angleY = 0.0f;
    static inline int lastX = -1, lastY = -1;
    static inline bool mousePressed = false;

public:
    static void show(const std::vector<Neuron> &n, int argc, char **argv)
    {
        neurons = new std::vector<Neuron>(n); // copiar
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
        glutInitWindowSize(800, 600);
        glutCreateWindow("SOM Visualizer");
        glEnable(GL_DEPTH_TEST);
        glutDisplayFunc(display);
        glutIdleFunc(glutPostRedisplay);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutMainLoop();
        delete neurons;
    }

private:
    static void display()
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        glTranslatef(0, 0, -60.0f);
        glRotatef(angleY, 1, 0, 0);
        glRotatef(angleX, 0, 1, 0);

        int idx = 0;
        for (int z = 0; z < dim_z; ++z)
        {
            for (int y = 0; y < dim_y; ++y)
            {
                for (int x = 0; x < dim_x; ++x)
                {
                    if (idx >= neurons->size())
                        break;
                    glPushMatrix();
                    glTranslatef((x - dim_x / 2.0f) * 3.0f,
                                 (y - dim_y / 2.0f) * 3.0f,
                                 (z - dim_z / 2.0f) * 3.0f);
                    drawImage((*neurons)[idx].get_weights());
                    glPopMatrix();
                    ++idx;
                }
            }
        }

        glutSwapBuffers();
    }

    static void reshape(int w, int h)
    {
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, (float)w / h, 1.0, 1000.0);
        glMatrixMode(GL_MODELVIEW);
    }

    static void mouse(int button, int state, int x, int y)
    {
        if (button == GLUT_LEFT_BUTTON)
        {
            mousePressed = (state == GLUT_DOWN);
            lastX = x;
            lastY = y;
        }
    }

    static void motion(int x, int y)
    {
        if (mousePressed)
        {
            angleX += (x - lastX);
            angleY += (y - lastY);
            lastX = x;
            lastY = y;
            glutPostRedisplay();
        }
    }

    static void drawImage(const std::vector<double> &weights)
    {
        int size = (int)std::sqrt(weights.size()); // 28
        float pixelSize = 0.05f;
        glBegin(GL_QUADS);
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                int idx = y * size + x;
                float gray = weights[idx]; // en [0, 1]
                glColor3f(gray, gray, gray);

                float xf = (x - size / 2) * pixelSize;
                float yf = (y - size / 2) * pixelSize;

                glVertex3f(xf, yf, 0);
                glVertex3f(xf + pixelSize, yf, 0);
                glVertex3f(xf + pixelSize, yf + pixelSize, 0);
                glVertex3f(xf, yf + pixelSize, 0);
            }
        }
        glEnd();
    }
};
