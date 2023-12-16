cmake_minimum_required(VERSION 3.12)
project(example)

include_directories(../module)
link_directories(${PROJECT_SOURCE_DIR}/libs)

add_executable(example example.cc)

target_link_libraries(example examples)

install(
    FILES ${SOLUTION_DIR}/build/test/cpp/ops/example/example
    DESTINATION ${BINARY_DIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
