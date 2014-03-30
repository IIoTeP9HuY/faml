set(AUTHOR "riad")
set(NAME_SUFFIX ${EXAMPLE_NAME}_${AUTHOR})

add_executable(predict_${NAME_SUFFIX} predict.cpp)
target_link_libraries(predict_${NAME_SUFFIX} faml)

add_executable(test_${NAME_SUFFIX} test.cpp)
target_link_libraries(test_${NAME_SUFFIX} faml)
