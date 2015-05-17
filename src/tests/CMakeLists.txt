cmake_minimum_required(VERSION 2.8)
set(PROJECT faml-tests)
project(${PROJECT})

# Prepare testing infrastructure.
include(DownloadGoogleTesting)

DOWNLOAD_GOOGLE_TESTING()

INCLUDE_DIRECTORIES(
    SYSTEM ${GTEST_INCLUDE_DIR}
    SYSTEM ${GMOCK_INCLUDE_DIR}
)

LINK_DIRECTORIES(${GTEST_BINARY_DIR} ${GMOCK_BINARY_DIR})

set(CMAKE_BUILD_TYPE "Debug")
add_definitions(-std=c++0x -Wall -pedantic -Wextra)

set(SOURCE_MAIN main.cpp)

set(SOURCES_TEST
	   test_tables.cpp
)

set(SOURCES ${SOURCE_MAIN} ${SOURCES_TEST})

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(${PROJECT} ${SOURCES})

add_dependencies(${PROJECT} googlemock)

# Pthreads linking behaves differently on clang and gcc, so a few hacks are
# required to make it working.
find_package(Threads REQUIRED)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(PUBLIC ${PROJECT} "-pthread")
endif()

target_link_libraries(${PROJECT}
    ${CMAKE_THREAD_LIBS_INIT}
    gmock
    gtest
    pthread
)
