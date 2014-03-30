add_executable(riad riad.cpp)
target_link_libraries(riad faml)

add_executable(riad_result riad_result.cpp)
target_link_libraries(riad_result faml)
