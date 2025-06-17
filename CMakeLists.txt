cmake_minimum_required(VERSION 3.10)
project(ProyectoRedKohonen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Incluir headers desde 'include/'
include_directories(include)

# Dependencias para visualización
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

# Fuente principal
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Ejecutable para entrenamiento
add_executable(KohonenTrainer main.cpp ${SRC_FILES})
target_link_libraries(KohonenTrainer PRIVATE OpenMP::OpenMP_CXX)

# Ejecutable para visualización
add_executable(KohonenVisualizer visualizer.cpp ${SRC_FILES})
target_link_libraries(KohonenVisualizer PRIVATE 
    OpenMP::OpenMP_CXX
    GLUT::GLUT 
    OpenGL::GL 
    OpenGL::GLU
)
