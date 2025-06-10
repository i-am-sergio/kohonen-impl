# Versión mínima de CMake requerida.
cmake_minimum_required(VERSION 3.10)

# Definición del proyecto.
project(ProyectoRedKohonen LANGUAGES CXX)

# Estándares de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# -----------------------------------------------------------------------------
# BUSCAR DEPENDENCIAS
# -----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

# -----------------------------------------------------------------------------
# EJECUTABLE DE ENTRENAMIENTO (KohonenTrainer)
# -----------------------------------------------------------------------------
add_executable(KohonenTrainer src/main.cpp)

target_link_libraries(KohonenTrainer PRIVATE 
    OpenMP::OpenMP_CXX
)

# -----------------------------------------------------------------------------
# EJECUTABLE DEL VISUALIZADOR (KohonenVisualizer)
# -----------------------------------------------------------------------------
add_executable(KohonenVisualizer src/mainVisualizer.cpp)

target_link_libraries(KohonenVisualizer PRIVATE 
    GLUT::GLUT
    OpenGL::GLU
    OpenGL::GL
)

# -----------------------------------------------------------------------------
# MENSAJES DE SALIDA
# -----------------------------------------------------------------------------
message(STATUS "Configuración de CMake completada.")
message(STATUS "Para compilar, crea un directorio 'build', entra en él y ejecuta: 'cmake ..' y luego 'make'")
message(STATUS "Ejecutables generados: ./KohonenTrainer y ./KohonenVisualizer")
