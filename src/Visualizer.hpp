#pragma once

#include "Neuron.hpp"
#include <GL/freeglut.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

class Visualizer {
private:
  // Punteros y variables estáticas para que las funciones de callback de GLUT puedan acceder a ellas
  static inline const std::vector<Neuron> *neurons_ptr = nullptr;
  static inline int dim_x = 0, dim_y = 0, dim_z = 0;
  static inline float angle_x = 20.0f, angle_y = -30.0f, zoom = -60.0f;
  static inline int last_mouse_x = -1, last_mouse_y = -1;
  static inline bool mouse_left_down = false;
  static inline bool mouse_right_down = false;

public:
  static void show(const std::vector<Neuron> &neurons, int dX, int dY, int dZ, int argc, char **argv) {
    neurons_ptr = &neurons;
    dim_x = dX;
    dim_y = dY;
    dim_z = dZ;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("Visualizador de Red Kohonen 3D");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    glutDisplayFunc(display_callback);
    glutReshapeFunc(reshape_callback);
    glutMouseFunc(mouse_callback);
    glutMotionFunc(motion_callback);

    glutMainLoop();
  }

private:
  static void draw_image(const std::vector<double> &weights) {
    if (weights.empty())
      return;

    // 1. Encontrar el valor mínimo y máximo en los pesos de ESTA neurona
    const auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
    const double min_w = *min_it;
    const double max_w = *max_it;
    const double range = max_w - min_w;

    int size = static_cast<int>(std::sqrt(weights.size()));
    float pixel_size = 1.0f / size;
    glPushMatrix();
    glTranslatef(-0.5f, -0.5f, 0.0f);

    glBegin(GL_QUADS);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        int idx = y * size + x;

        // 2. Normalizar el valor del píxel para maximizar el contraste
        float gray = 0.5f; // Gris por defecto si el rango es cero
        if (range > std::numeric_limits<double>::epsilon()) {
          gray = static_cast<float>((weights[idx] - min_w) / range);
        }

        glColor3f(gray, gray, gray);

        float x_pos = x * pixel_size;
        float y_pos = y * pixel_size;

        glVertex3f(x_pos, y_pos, 0.0f);
        glVertex3f(x_pos + pixel_size, y_pos, 0.0f);
        glVertex3f(x_pos + pixel_size, y_pos + pixel_size, 0.0f);
        glVertex3f(x_pos, y_pos + pixel_size, 0.0f);
      }
    }
    glEnd();
    glPopMatrix();
  }

  static void display_callback() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, zoom);
    glRotatef(angle_y, 1.0f, 0.0f, 0.0f);
    glRotatef(angle_x, 0.0f, 1.0f, 0.0f);

    // float spacing = 2.5f;
    float spacing = 1.5f;
    float offset_x = -(dim_x - 1) * spacing / 2.0f;
    float offset_y = -(dim_y - 1) * spacing / 2.0f;
    float offset_z = -(dim_z - 1) * spacing / 2.0f;

    for (int z = 0; z < dim_z; ++z) {
      for (int y = 0; y < dim_y; ++y) {
        for (int x = 0; x < dim_x; ++x) {
          int idx = z * (dim_x * dim_y) + y * dim_x + x;
          if (idx < neurons_ptr->size()) {
            glPushMatrix();
            glTranslatef(offset_x + x * spacing, offset_y + y * spacing, offset_z + z * spacing);
            draw_image((*neurons_ptr)[idx].get_weights());
            glPopMatrix();
          }
        }
      }
    }
    glutSwapBuffers();
  }

  static void reshape_callback(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / h, 1.0, 1000.0);
    glMatrixMode(GL_MODELVIEW);
  }

  static void mouse_callback(int button, int state, int x, int y) {
    last_mouse_x = x;
    last_mouse_y = y;
    if (button == GLUT_LEFT_BUTTON)
      mouse_left_down = (state == GLUT_DOWN);
    if (button == GLUT_RIGHT_BUTTON)
      mouse_right_down = (state == GLUT_DOWN);
  }

  static void motion_callback(int x, int y) {
    if (mouse_left_down) {
      angle_x += (x - last_mouse_x);
      angle_y += (y - last_mouse_y);
    }
    if (mouse_right_down) {
      zoom += (y - last_mouse_y) * 0.1f;
    }
    last_mouse_x = x;
    last_mouse_y = y;
    glutPostRedisplay();
  }
};
