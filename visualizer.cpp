#include <GL/glut.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include "Loader.hpp"    // para load_som_weights_txt
#include "Reader.hpp"    // para Reader::load_csv

// --- CONSTANTES ---
const int GRID_SIZE   = 10;           // 10×10×10 grilla
const int IMAGE_SIZE  = 28;
const float SPACING   = 2.0f;

// --- VARIABLES GLOBALES OPENGL ---
static float angle       = 0.0f;
static float cam_radius  = 45.0f;
static float cam_yaw     = 45.0f, cam_pitch = 30.0f;
static int   last_x      = 0, last_y = 0;
static bool  left_down   = false;

// --- DATOS DE RED Y PREDICCIÓN ---
std::vector<std::vector<double>> weights;  // pesos cargados
int pred_idx = -1;                          // índice de la BMU

// Encuentra la BMU de un vector (distancia euclidiana)
int find_bmu(const std::vector<double>& sample) {
    double best = std::numeric_limits<double>::max();
    int    bi   = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        double d = 0;
        for (size_t j = 0; j < sample.size(); ++j) {
            double diff = sample[j] - weights[i][j];
            d += diff * diff;
        }
        if (d < best) {
            best = d;
            bi   = static_cast<int>(i);
        }
    }
    return bi;
}

// --- VIEWNEURON CLASS ---
class ViewNeuron {
    const std::vector<double>& image;
    float radius;
    float mesh_r, mesh_g, mesh_b;
    bool highlight;

public:
    ViewNeuron(float r_in, const std::vector<double>& img,
               float br = 1.0f, float bg = 1.0f, float bb = 1.0f,
               float mr = 0.0f, float mg = 1.0f, float mb = 0.0f)
        : image(img), radius(r_in), mesh_r(mr), mesh_g(mg), mesh_b(mb), highlight(false) {}

    void set_highlight(bool h) { highlight = h; }

    void draw(float x, float y, float z) const {
        if (image.empty()) return;
        auto[min_it, max_it] = std::minmax_element(image.begin(), image.end());
        double mn = *min_it;
        double mx = *max_it;
        double range = mx - mn;

        glPushMatrix();
        glTranslatef(x, y, z);
        glRotatef(angle, 0.0f, 1.0f, 0.0f);

        // Dibujar la imagen en un QUAD
        glBegin(GL_QUADS);
        float scale = radius * 1.5f / IMAGE_SIZE;
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                int idx = i * IMAGE_SIZE + j;
                float gray = (range > std::numeric_limits<double>::epsilon())
                             ? static_cast<float>((image[idx] - mn) / range)
                             : 0.5f;
                glColor3f(gray, gray, gray);
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

        // Dibujar esfera, resaltada si es la BMU
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        if (highlight) {
            glColor4f(1.0f, 0.0f, 0.0f, 0.5f);
        } else {
            glColor4f(mesh_r, mesh_g, mesh_b, 0.25f);
        }
        glutSolidSphere(radius, 20, 20);
        glDisable(GL_BLEND);
        glPopMatrix();
    }
};

std::vector<ViewNeuron> neurons;

// --- OPENGL CALLBACKS ---
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float cx = cam_radius * cosf(cam_pitch * M_PI / 180.0f) * sinf(cam_yaw * M_PI / 180.0f);
    float cy = cam_radius * sinf(cam_pitch * M_PI / 180.0f);
    float cz = cam_radius * cosf(cam_pitch * M_PI / 180.0f) * cosf(cam_yaw * M_PI / 180.0f);
    gluLookAt(cx, cy, cz, 0, 0, 0, 0, 1, 0);

    int index = 0;
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int z = 0; z < GRID_SIZE; ++z) {
                float px = (x - GRID_SIZE / 2) * SPACING;
                float py = (y - GRID_SIZE / 2) * SPACING;
                float pz = (z - GRID_SIZE / 2) * SPACING;
                neurons[index].draw(px, py, pz);
                ++index;
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
        left_down = (state == GLUT_DOWN);
        last_x = x;
        last_y = y;
    }
    if (button == 3) cam_radius = std::max(5.0f, cam_radius - 1.0f);
    if (button == 4) cam_radius = std::min(150.0f, cam_radius + 1.0f);
}

void motion(int x, int y) {
    if (!left_down) return;
    cam_yaw   += (x - last_x) * 0.5f;
    cam_pitch -= (y - last_y) * 0.5f;
    cam_pitch = std::clamp(cam_pitch, -89.0f, 89.0f);
    last_x = x;
    last_y = y;
}

void initGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0, 1.0, 1.0, 1000.0);
    glMatrixMode(GL_MODELVIEW);
    glClearColor(0, 0, 0, 1);
}

// --- MAIN ---
int main(int argc, char** argv) {
    // 1) Cargo pesos
    weights = load_som_weights_txt("model_10ep.txt");
    if (weights.empty()) {
        std::cerr << "Error al cargar pesos" << std::endl;
        return 1;
    }

    // 2) Cargo muestra de prueba
    std::vector<std::vector<double>> X_test, Y_test_oh;
    Reader::load_csv("database/mnist_test_flat.csv", X_test, Y_test_oh);
    if (X_test.empty()) {
        std::cerr << "Error al cargar datos de prueba" << std::endl;
        return 1;
    }
    auto test_sample = X_test.front();

    // 3) Calculo BMU
    pred_idx = find_bmu(test_sample);
    std::cout << "BMU del primer sample: " << pred_idx << std::endl;

    // 4) Creo ViewNeuron y resalto la BMU
    float mesh_r = 182.0f/255.0f;
    float mesh_g = 174.0f/255.0f;
    float mesh_b = 235.0f/255.0f;
    for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
        neurons.emplace_back(0.85f, weights[i], 1.0f, 1.0f, 1.0f, mesh_r, mesh_g, mesh_b);
        if (i == pred_idx) neurons.back().set_highlight(true);
    }

    // 5) Iniciar OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 1000);
    glutCreateWindow("SOM Predictor Highlight");
    initGL();
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMainLoop();
    return 0;
}