cmake_minimum_required(VERSION 2.8)

project(faml CXX)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/foreign)
add_definitions(-O2 --std=c++0x)
add_definitions(-fopenmp)

set(BUILD_EXAMPLES TRUE)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)

file(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/include/faml/*.hpp)
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

ADD_LIBRARY(faml SHARED
	${HEADERS}
	${SOURCES}
	)

target_link_libraries(faml gomp)
