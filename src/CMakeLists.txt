cmake_minimum_required(VERSION 2.8)
project(src)

if(ENABLE_EXAMPLES)
	add_subdirectory(examples)
endif()

if(ENABLE_TESTING)
	add_subdirectory(tests)
endif()
