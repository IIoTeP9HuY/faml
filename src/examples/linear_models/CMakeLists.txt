add_executable(linear_regression linear_regression.cpp)
add_executable(logistic_regression logistic_regression.cpp)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(linear_regression gomp)
    target_link_libraries(logistic_regression gomp)
endif()
