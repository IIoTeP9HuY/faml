set(AUTHOR "acid")
set(NAME_SUFFIX ${EXAMPLE_NAME}_${AUTHOR})

add_executable(predict_${NAME_SUFFIX} predict.cpp)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(predict_${NAME_SUFFIX} gomp)
endif()
