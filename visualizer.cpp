// SOM Viewer con Entrada Manual de Índice y Comparación de Etiquetas
#include <GL/glut.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include "Loader.hpp"
#include "Reader.hpp"

const int GRID_SIZE = 10;
const int IMAGE_SIZE = 28;
const float SPACING = 2.0f;

static float angle = 0.0f;
static float cam_radius = 45.0f;
static float cam_yaw = 45.0f, cam_pitch = 30.0f;
static int last_x = 0, last_y = 0;
static bool left_down = false;

std::vector<std::vector<double>> weights;
std::vector<std::vector<double>> X_test, Y_test_oh;
int pred_idx = -1, pred_digit = -1, true_digit = -1;

std::string input_text = "";
bool over_button = false;

// BMU
int find_bmu(const std::vector<double> &sample) {
    double best = std::numeric_limits<double>::max();
    int bi = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        double d = 0;
        for (size_t j = 0; j < sample.size(); ++j) {
            double diff = sample[j] - weights[i][j];
            d += diff * diff;
        }
        if (d < best) {
            best = d;
            bi = static_cast<int>(i);
        }
    }
    return bi;
}

// Clase ViewNeuron
class ViewNeuron {
    const std::vector<double> &image;
    float radius;
    float mesh_r, mesh_g, mesh_b;
    bool highlight;
public:
    ViewNeuron(float r_in, const std::vector<double> &img,
               float br = 1.0f, float bg = 1.0f, float bb = 1.0f,
               float mr = 0.0f, float mg = 1.0f, float mb = 0.0f)
        : image(img), radius(r_in), mesh_r(mr), mesh_g(mg), mesh_b(mb), highlight(false) {}

    void set_highlight(bool h) { highlight = h; }

    void draw(float x, float y, float z) const {
        if (image.empty()) return;
        auto [min_it, max_it] = std::minmax_element(image.begin(), image.end());
        double mn = *min_it, mx = *max_it, range = mx - mn;

        glPushMatrix();
        glTranslatef(x, y, z);
        glRotatef(angle, 0.0f, 1.0f, 0.0f);

        glBegin(GL_QUADS);
        float scale = radius * 1.5f / IMAGE_SIZE;
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                int idx = i * IMAGE_SIZE + j;
                float gray = (range > 1e-5) ? (image[idx] - mn) / range : 0.5f;
                glColor3f(gray, gray, gray);
                float x0 = (j - IMAGE_SIZE / 2.0f) * scale;
                float y0 = (i - IMAGE_SIZE / 2.0f) * scale;
                float x1 = x0 + scale;
                float y1 = y0 + scale;
                glVertex3f(x0, y0, 0.0f); glVertex3f(x1, y0, 0.0f);
                glVertex3f(x1, y1, 0.0f); glVertex3f(x0, y1, 0.0f);
            }
        }
        glEnd();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(highlight ? 1.0f : mesh_r, highlight ? 0.0f : mesh_g, highlight ? 0.0f : mesh_b, highlight ? 0.5f : 0.25f);
        glutSolidSphere(radius, 20, 20);
        glDisable(GL_BLEND);
        glPopMatrix();
    }
};

std::vector<ViewNeuron> neurons;

void drawText(float x, float y, std::string text, void *font = GLUT_BITMAP_HELVETICA_18) {
    glRasterPos2f(x, y);
    for (char c : text) glutBitmapCharacter(font, c);
}

void drawUI() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 1000, 0, 1000);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Etiqueta
    glColor3f(1, 1, 1);
    drawText(20, 960, "Indice de muestra [0-" + std::to_string(X_test.size() - 1) + "]:");

    // Cuadro de texto (input)
    glColor3f(1, 1, 1);
    glBegin(GL_LINE_LOOP);  // Borde blanco
    glVertex2f(335, 950);
    glVertex2f(485, 950);
    glVertex2f(485, 975);
    glVertex2f(335, 975);
    glEnd();
    drawText(340, 955, input_text);  // Texto dentro del input

    // Botón gris
    glColor3f(over_button ? 0.7f : 0.5f, 0.5f, 0.5f);  // Gris dinámico
    glBegin(GL_QUADS);
    glVertex2f(800, 940);
    glVertex2f(950, 940);
    glVertex2f(950, 980);
    glVertex2f(800, 980);
    glEnd();

    // Texto del botón centrado
    std::string label = "Predecir";
    int text_width = glutBitmapLength(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)label.c_str());
    int button_center_x = 800 + (150 - text_width) / 2;
    int button_center_y = 955;  // Ajustado para mejor visibilidad

    glColor3f(1, 1, 1);  // Texto blanco
    drawText(button_center_x, button_center_y, label);

    // Mostrar resultados si existen
    if (true_digit != -1 && pred_digit != -1) {
        bool correcto = (true_digit == pred_digit);
        if (correcto)
            glColor3f(37.0f / 255.0f, 232.0f / 255.0f, 92.0f / 255.0f);  // verde
        else
            glColor3f(242.0f / 255.0f, 65.0f / 255.0f, 48.0f / 255.0f);  // rojo

        drawText(20, 920, "Esperado: " + std::to_string(true_digit));
        drawText(20, 880, "Prediccion: " + std::to_string(pred_digit));
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}



void predict_and_highlight(int idx) {
    if (idx < 0 || idx >= (int)X_test.size()) return;
    pred_idx = find_bmu(X_test[idx]);

    // Calcular etiqueta verdadera y predicha
    auto &y_true = Y_test_oh[idx];
    true_digit = std::distance(y_true.begin(), std::max_element(y_true.begin(), y_true.end()));
    pred_digit = pred_idx % 10; // heurística (ajústala si tienes etiquetas de BMU reales)

    for (int i = 0; i < (int)neurons.size(); ++i)
        neurons[i].set_highlight(i == pred_idx);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float cx = cam_radius * cosf(cam_pitch * M_PI / 180.0f) * sinf(cam_yaw * M_PI / 180.0f);
    float cy = cam_radius * sinf(cam_pitch * M_PI / 180.0f);
    float cz = cam_radius * cosf(cam_pitch * M_PI / 180.0f) * cosf(cam_yaw * M_PI / 180.0f);
    gluLookAt(cx, cy, cz, 0, 0, 0, 0, 1, 0);

    int index = 0;
    for (int x = 0; x < GRID_SIZE; ++x)
        for (int y = 0; y < GRID_SIZE; ++y)
            for (int z = 0; z < GRID_SIZE; ++z)
                neurons[index++].draw((x - GRID_SIZE / 2) * SPACING,
                                      (y - GRID_SIZE / 2) * SPACING,
                                      (z - GRID_SIZE / 2) * SPACING);

    drawUI();
    glutSwapBuffers();
}

void idle() {
    angle += 0.2f;
    if (angle > 360.0f) angle -= 360.0f;
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        if (x >= 800 && x <= 950 && (1000 - y) >= 940 && (1000 - y) <= 980) {
            if (!input_text.empty()) predict_and_highlight(std::stoi(input_text));
        }
        left_down = true;
        last_x = x;
        last_y = y;
    } else {
        left_down = false;
    }
    if (button == 3) cam_radius = std::max(5.0f, cam_radius - 1.0f);
    if (button == 4) cam_radius = std::min(150.0f, cam_radius + 1.0f);
}

void motion(int x, int y) {
    if (!left_down) return;
    cam_yaw += (x - last_x) * 0.5f;
    cam_pitch -= (y - last_y) * 0.5f;
    cam_pitch = std::clamp(cam_pitch, -89.0f, 89.0f);
    last_x = x;
    last_y = y;
}

void passive_motion(int x, int y) {
    over_button = (x >= 800 && x <= 950 && (1000 - y) >= 940 && (1000 - y) <= 980);
}

void keyboard(unsigned char key, int, int) {
    if (key >= '0' && key <= '9') input_text += key;
    else if (key == '\b' || key == 127) { if (!input_text.empty()) input_text.pop_back(); }
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

int main(int argc, char **argv) {
    weights = load_som_weights_txt("output/mnist_gaussian_radius/best_model.dat");
    if (weights.empty()) { std::cerr << "Error al cargar pesos\n"; return 1; }

    Reader::load_csv("database/mnist_test_flat.csv", X_test, Y_test_oh, 10);
    if (X_test.empty()) { std::cerr << "Error al cargar test\n"; return 1; }

    float mesh_r = 182.0f / 255.0f, mesh_g = 174.0f / 255.0f, mesh_b = 235.0f / 255.0f;
    for (size_t i = 0; i < weights.size(); ++i)
        neurons.emplace_back(0.85f, weights[i], 1.0f, 1.0f, 1.0f, mesh_r, mesh_g, mesh_b);

    pred_idx = find_bmu(X_test[0]);
    neurons[pred_idx].set_highlight(true);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1000, 1000);
    glutCreateWindow("SOM Predictor Input");
    initGL();
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(passive_motion);
    glutKeyboardFunc(keyboard);
    glutMainLoop();
    return 0;
}
