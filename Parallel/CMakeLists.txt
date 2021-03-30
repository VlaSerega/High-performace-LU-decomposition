cmake_minimum_required(VERSION 3.16)
project(Parallel C)

set(CMAKE_C_STANDARD 11)
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
endif()

add_executable(Parallel main.c dgemm.c dgetrf.c)