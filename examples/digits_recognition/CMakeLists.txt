add_executable(predict predict.cpp)
target_link_libraries(predict faml)

add_executable(prepare prepare.cpp)
target_link_libraries(prepare faml)

add_executable(riad riad.cpp)
target_link_libraries(riad faml)
