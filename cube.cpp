#include <GL/glut.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>

const int rows = 5, cols = 5, layers = 5;
const float spacing = 1.2f;

float rotX = 0.0f, rotY = 0.0f;
float zoomZ = 20.0f;
bool isDragging = false;
int lastMouseX = 0, lastMouseY = 0;

void drawText(float x, float y, float z, const std::string& text) {
    glRasterPos3f(x, y, z);
    for (char c : text)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
}

class SphereGrid {
public:
    std::vector<std::vector<std::vector<int>>> grid;
    int selectedID = -1;

    SphereGrid(int r, int c, int l)
        : grid(r, std::vector<std::vector<int>>(c, std::vector<int>(l, 1))) {}

    int coordToID(int i, int j, int k) {
        return i * cols * layers + j * layers + k;
    }

    void encodeIDColor(int id) {
        glColor3ub((id & 0xFF), (id >> 8) & 0xFF, (id >> 16) & 0xFF);
    }

    int decodeIDColor(unsigned char r, unsigned char g, unsigned char b) {
        return r + (g << 8) + (b << 16);
    }

    void draw(bool forPicking = false) {
        glPushMatrix();
        glRotatef(rotX, 1.0, 0.0, 0.0);
        glRotatef(rotY, 0.0, 1.0, 0.0);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < layers; ++k) {
                    if (grid[i][j][k] == 0) continue;

                    glPushMatrix();
                    float fx = (i - rows / 2.0f) * spacing;
                    float fy = (j - cols / 2.0f) * spacing;
                    float fz = (k - layers / 2.0f) * spacing;
                    int id = coordToID(i, j, k);

                    if (id == selectedID) {
                        glTranslatef(6.0f, 0.0f, 0.0f);
                        glTranslatef(fx, fy, fz);
                        glScalef(2.0f, 2.0f, 2.0f);
                    } else {
                        glTranslatef(fx, fy, fz);
                    }

                    if (forPicking) {
                        encodeIDColor(id);
                    } else {
                        GLfloat color[4];
                        if (id == selectedID) {
                            color[0] = 0.0f; color[1] = 1.0f; color[2] = 0.0f; color[3] = 1.0f;
                        } else {
                            color[0] = float(i) / (rows - 1);
                            color[1] = float(j) / (cols - 1);
                            color[2] = float(k) / (layers - 1);
                            color[3] = 1.0f;
                        }
                        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
                    }

                    glutSolidSphere(0.3, 16, 16);

                    if (!forPicking) {
                        std::ostringstream oss;
                        oss << "(" << i << "," << j << "," << k << ")";
                        drawText(0.0f, 0.0f, 0.0f, oss.str());
                    }

                    glPopMatrix();
                }
            }
        }

        glPopMatrix();
    }

    void pick(int x, int y) {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_LIGHTING);
        draw(true);
        glFlush();

        unsigned char pixel[3];
        glReadPixels(x, viewport[3] - y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, &pixel);
        int id = decodeIDColor(pixel[0], pixel[1], pixel[2]);

        if (id == selectedID)
            selectedID = -1;
        else
            selectedID = id;

        glEnable(GL_LIGHTING);
        glutPostRedisplay();
    }

    void drawBoundingCube() {
        glPushMatrix();
        glRotatef(rotX, 1.0, 0.0, 0.0);
        glRotatef(rotY, 0.0, 1.0, 0.0);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        GLfloat cubeColor[] = {0.8f, 0.8f, 1.0f, 0.15f};
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, cubeColor);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        float sizeX = (rows - 1) * spacing;
        float sizeY = (cols - 1) * spacing;
        float sizeZ = (layers - 1) * spacing;
        glTranslatef(0, 0, 0);
        glScalef(sizeX + 2.0f, sizeY + 2.0f, sizeZ + 2.0f);
        glutSolidCube(1.0f);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable(GL_BLEND);

        glPopMatrix();
    }
};

SphereGrid grid(rows, cols, layers);

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(0, 0, zoomZ, 0, 0, 0, 0, 1, 0);

    grid.drawBoundingCube();
    grid.draw(false);

    glutSwapBuffers();
}

void reshape(int w, int h) {
    if (h == 0) h = 1;
    float ratio = float(w) / float(h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, ratio, 1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        isDragging = true;
        lastMouseX = x;
        lastMouseY = y;
    } else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        isDragging = false;
    }

    if (button == 3) {
        zoomZ = std::max(3.0f, zoomZ - 1.0f);
        glutPostRedisplay();
    } else if (button == 4) {
        zoomZ = std::min(100.0f, zoomZ + 1.0f);
        glutPostRedisplay();
    }
}

void motion(int x, int y) {
    if (isDragging) {
        rotX += (y - lastMouseY) * 0.5f;
        rotY += (x - lastMouseX) * 0.5f;
        lastMouseX = x;
        lastMouseY = y;
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y) {
    if (key == 27) exit(0); // ESC
    if (key == 'q' || key == 'Q') {
        grid.pick(x, y);
    }
}

void init() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    GLfloat light_pos[] = {5.0f, 5.0f, 10.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glShadeModel(GL_SMOOTH);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Grid de Esferas con Cubo Transparente");

    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    return 0;
}
