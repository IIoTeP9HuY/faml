cmake_minimum_required(VERSION 2.8)
project(faml-examples)

add_definitions(-O2 -std=c++0x -Wall)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-fopenmp)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_definitions(-openmp)
endif()

include_directories(..)
include_directories(../../foreign)

add_subdirectory(digits_recognition)
add_subdirectory(50k)
add_subdirectory(linear_models)
