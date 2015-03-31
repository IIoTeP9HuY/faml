cmake_minimum_required(VERSION 2.8)

project(faml CXX)

set(CMAKE_MACOSX_RPATH 1)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/foreign)

add_definitions(-O2 -std=c++0x -Wall)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-fopenmp)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_definitions(-openmp)
endif()

set(BUILD_EXAMPLES TRUE)

file(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/include/faml/*.hpp)
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

ADD_LIBRARY(faml SHARED
	${HEADERS}
	${SOURCES}
	)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(faml gomp)
endif()
