cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(A)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_definitions(-DTIXML_USE_STL)

add_compile_options(-g)
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("./include")
include_directories("./tinyxml2")

file(GLOB srcs A.cpp cpp/*/*.cpp tinyxml2/*.cpp)
add_executable(A ${srcs})
target_link_libraries(A "${TORCH_LIBRARIES}" ${PYTHON_LIBRARIES})
set_property(TARGET A PROPERTY CXX_STANDARD 14)

