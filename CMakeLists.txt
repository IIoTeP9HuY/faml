cmake_minimum_required(VERSION 2.8)
project(faml CXX)

set(FAML_VERSION_MAJOR 0)
set(FAML_VERSION_MINOR 1)
set(FAML_VERSION_PATCH 0)
set(FAML_VERSION "${FAML_VERSION_MAJOR}.${FAML_VERSION_MINOR}.${FAML_VERSION_PATCH}")

option(ENABLE_EXAMPLES "Enable examples" ON)

add_subdirectory(src)

set(CMAKE_MACOSX_RPATH 1)
