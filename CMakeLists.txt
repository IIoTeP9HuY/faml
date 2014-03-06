cmake_minimum_required(VERSION 2.8)

project(faml CXX)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(-O2 --std=c++0x)

set(BUILD_EXAMPLES TRUE)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)

file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/faml/*.hpp)
file(GLOB HEADERS ${HEADERS};${PROJECT_SOURCE_DIR}/include/faml/models/*.hpp)

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

ADD_LIBRARY(faml SHARED
	${HEADERS}
	src/distances
	)
