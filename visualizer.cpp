#include <GL/glut.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include "Loader.hpp"
#include "RedKohonen.hpp"
#include "Reader.hpp"      // si quieres cargar un sample desde CSV


// --- CONSTANTES ---
const int GRID_SIZE = 10; // 8;
const int NUM_NEURONS = GRID_SIZE*GRID_SIZE*GRID_SIZE;
const int IMAGE_SIZE = 28;
const float SPACING = 2.0f;


// --- VARIABLES GLOBALES ---
float angle = 0.0f;
float cam_radius = 45.0f;
float cam_yaw = 45.0f, cam_pitch = 30.0f;
int last_x = 0, last_y = 0;
bool left_button_down = false;

// --- VIEWNEURON CLASS ---
class ViewNeuron {
    float radius;
    float r, g, b;
    float mesh_r, mesh_g, mesh_b;
    std::vector<double> image;

public:
    ViewNeuron(float radius, const std::vector<double>& image, float r=1.0f, float g=1.0f, float b=1.0f,
               float mesh_r=0.0f, float mesh_g=1.0f, float mesh_b=0.0f)
        : radius(radius), image(image), r(r), g(g), b(b),
          mesh_r(mesh_r), mesh_g(mesh_g), mesh_b(mesh_b) {}

    void draw(float x, float y, float z) {
        if (image.empty()) return;

        // 1. Encontrar el valor mínimo y máximo en los pesos de ESTA neurona
        const auto [min_it, max_it] = std::minmax_element(image.begin(), image.end());
        const double min_w = *min_it;
        const double max_w = *max_it;
        const double range = max_w - min_w;

        glPushMatrix();
        glTranslatef(x, y, z);
        glRotatef(angle, 0.0f, 1.0f, 0.0f);  // rotación de la imagen
        angle += 0.001f;

        // Dibujar la imagen (en el centro)
        glBegin(GL_QUADS);
        float scale = radius * 1.5f / IMAGE_SIZE;
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                int idx = i * IMAGE_SIZE + j;
                
                // 2. Normalizar el valor del píxel para maximizar el contraste
                float gray = 0.5f; // Gris por defecto si el rango es cero
                if (range > std::numeric_limits<double>::epsilon()) {
                    gray = static_cast<float>((image[idx] - min_w) / range);
                }

                glColor3f(gray, gray, gray);  // Blanco a negro
                float x0 = (j - IMAGE_SIZE / 2.0f) * scale;
                float y0 = (i - IMAGE_SIZE / 2.0f) * scale;
                float x1 = x0 + scale;
                float y1 = y0 + scale;

                glVertex3f(x0, y0, 0.0f);
                glVertex3f(x1, y0, 0.0f);
                glVertex3f(x1, y1, 0.0f);
                glVertex3f(x0, y1, 0.0f);
            }
        }
        glEnd();

        // Esfera translúcida
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(mesh_r, mesh_g, mesh_b, 0.25f);  // 25% opacidad
        glutSolidSphere(radius, 32, 32);
        glDisable(GL_BLEND);

        glPopMatrix();
    }

    void drawCube(float x, float y, float z) {
        if (image.empty()) return;

        // 1. Encontrar el valor mínimo y máximo en los pesos de ESTA neurona
        const auto [min_it, max_it] = std::minmax_element(image.begin(), image.end());
        const double min_w = *min_it;
        const double max_w = *max_it;
        const double range = max_w - min_w;

        glPushMatrix();
        glTranslatef(x, y, z);
        glRotatef(angle, 0.0f, 1.0f, 0.0f);  // rotación del cubo

        // Dibujar los 6 planos (cubo)
        float cube_size = radius * 1.2f; // Tamaño del cubo
        float scale = cube_size / IMAGE_SIZE;

        // Función para dibujar un plano en una posición dada
        auto draw_plane = [&](float nx, float ny, float nz, float angle, float ax, float ay, float az) {
            glPushMatrix();
            glRotatef(angle, ax, ay, az);
            glTranslatef(0, 0, cube_size/2);

            glBegin(GL_QUADS);
            for (int i = 0; i < IMAGE_SIZE; ++i) {
                for (int j = 0; j < IMAGE_SIZE; ++j) {
                    int idx = i * IMAGE_SIZE + j;
                    float gray = 0.5f;
                    if (range > std::numeric_limits<double>::epsilon()) {
                        gray = static_cast<float>((image[idx] - min_w) / range);
                    }

                    glColor3f(gray, gray, gray);
                    float x0 = (j - IMAGE_SIZE/2.0f) * scale;
                    float y0 = (i - IMAGE_SIZE/2.0f) * scale;
                    float x1 = x0 + scale;
                    float y1 = y0 + scale;

                    glVertex3f(x0, y0, 0.0f);
                    glVertex3f(x1, y0, 0.0f);
                    glVertex3f(x1, y1, 0.0f);
                    glVertex3f(x0, y1, 0.0f);
                }
            }
            glEnd();
            glPopMatrix();
        };

        // Dibujar los 6 planos del cubo
        draw_plane(0, 0, 1, 0, 0, 1, 0);   // Frente
        draw_plane(0, 0, -1, 180, 0, 1, 0); // Atrás
        draw_plane(1, 0, 0, 90, 0, 1, 0);   // Derecha
        draw_plane(-1, 0, 0, -90, 0, 1, 0); // Izquierda
        draw_plane(0, 1, 0, 90, 1, 0, 0);   // Arriba
        draw_plane(0, -1, 0, -90, 1, 0, 0); // Abajo

        // Esfera translúcida (ajustada al tamaño del cubo)
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(mesh_r, mesh_g, mesh_b, 0.15f);  // 15% opacidad (más transparente)
        glutSolidSphere(cube_size * 0.85f, 32, 32); // Esfera ligeramente más pequeña que el cubo
        glDisable(GL_BLEND);

        glPopMatrix();
    }
};


// Índice de la neurona a resaltar (BMU). -1 = ninguno
int highlight_idx = -1;

// Llamar desde main() para fijar qué neurona pintarás en rojo
void setHighlightIdx(int idx) {
    highlight_idx = idx;
}

std::vector<ViewNeuron> neurons;

// --- FUNCIONES OPENGL ---
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float cam_x = cam_radius * cos(cam_pitch * M_PI / 180.0) * sin(cam_yaw * M_PI / 180.0);
    float cam_y = cam_radius * sin(cam_pitch * M_PI / 180.0);
    float cam_z = cam_radius * cos(cam_pitch * M_PI / 180.0) * cos(cam_yaw * M_PI / 180.0);
    gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0);

    int index = 0;
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int z = 0; z < GRID_SIZE; ++z) {
                if (index >= neurons.size()) break;
                float xpos = (x - GRID_SIZE/2) * SPACING;
                float ypos = (y - GRID_SIZE/2) * SPACING;
                float zpos = (z - GRID_SIZE/2) * SPACING;
                neurons[index++].draw(xpos, ypos, zpos);
            }
        }
    }

    glutSwapBuffers();
}

void idle() {
    angle += 0.2f;
    if (angle > 360.0f) angle -= 360.0f;
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        left_button_down = (state == GLUT_DOWN);
        last_x = x;
        last_y = y;
    }

    if (button == 3) cam_radius -= 1.0f; // scroll up
    if (button == 4) cam_radius += 1.0f; // scroll down
    if (cam_radius < 5.0f) cam_radius = 5.0f;
    if (cam_radius > 150.0f) cam_radius = 150.0f;
}

void motion(int x, int y) {
    if (left_button_down) {
        cam_yaw += (x - last_x) * 0.5f;
        cam_pitch -= (y - last_y) * 0.5f;  // nota el signo menos aquí
        if (cam_pitch > 89.0f) cam_pitch = 89.0f;
        if (cam_pitch < -89.0f) cam_pitch = -89.0f;
        last_x = x;
        last_y = y;
    }
}

void initOpenGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND); // Importante para transparencia
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0, 1.0, 1.0, 1000.0);
    glMatrixMode(GL_MODELVIEW);

    glClearColor(0, 0, 0, 1);
}


// --- MAIN ---
int main(int argc, char** argv) {
    // auto weights = load_som_weights("model.bin");
    auto weights = load_som_weights_txt("model_10ep.txt");
    std::cout << "Loaded " << weights.size() << " neurons\n";
    std::cout << "Each neuron has " << weights[0].size() << " weights\n";

    for(int i=0; i<weights[0].size(); i++){
        std::cout << weights[0][i] << " ";
    }

    float mesh_r = 182.0f / 255.0f; // ≈ 0.7137
    float mesh_g = 174.0f / 255.0f; // ≈ 0.6824
    float mesh_b = 235.0f / 255.0f; // ≈ 0.9215

    for (const auto& image : weights) {
        neurons.emplace_back(0.85f, image, 1.0f, 1.0f, 1.0f, mesh_r, mesh_g, mesh_b);
    }

    
    glutInit(&argc, argv) ;
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 1000);
    glutCreateWindow("Neural Grid Viewer");

    initOpenGL();
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMainLoop();

    return 0;
}
