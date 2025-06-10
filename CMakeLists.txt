cmake_minimum_required(VERSION 3.10)
project(kohonen)

set(CMAKE_CXX_STANDARD 17)

# Permite que el ejecutable escriba archivos (importante para algunos sistemas)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

# Incluir directorios de encabezados
include_directories(include)
include_directories(.)  # para -I.

# Crear el ejecutable
add_executable(kohonen
    main.cpp
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kohonen PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enlazar con librerías específicas manualmente
target_link_libraries(kohonen
    PRIVATE
    glut
    GL
    GLU
    m
)
