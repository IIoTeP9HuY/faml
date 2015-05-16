add_executable(riad riad.cpp)
add_executable(riad_result riad_result.cpp)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(riad gomp)
    target_link_libraries(riad_result gomp)
endif()
