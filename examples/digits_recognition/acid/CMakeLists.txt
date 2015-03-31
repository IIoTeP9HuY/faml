set(AUTHOR "acid")
set(NAME_SUFFIX ${EXAMPLE_NAME}_${AUTHOR})

add_executable(predict_${NAME_SUFFIX} predict.cpp)
target_link_libraries(predict_${NAME_SUFFIX} faml)
