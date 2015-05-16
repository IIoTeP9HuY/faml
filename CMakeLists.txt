cmake_minimum_required(VERSION 2.8)
project(faml CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(FAML_VERSION_MAJOR 0)
set(FAML_VERSION_MINOR 1)
set(FAML_VERSION_PATCH 0)
set(FAML_VERSION "${FAML_VERSION_MAJOR}.${FAML_VERSION_MINOR}.${FAML_VERSION_PATCH}")

option(ENABLE_EXAMPLES "Enable examples" ON)
option(ENABLE_TESTING "Enable testing" OFF)

add_subdirectory(src)

set(CMAKE_MACOSX_RPATH 1)
