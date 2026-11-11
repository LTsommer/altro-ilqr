cmake_minimum_required(VERSION 3.24)
project(alilqr)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O2)

find_package(Eigen3 REQUIRED)
message("Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

#file(GLOB_RECURSE BC basic_constraint *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE CD constrain_definition *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE CF cost_function *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE FC frenet_coordinate *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE M model *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE S solver *.h *.cpp *.hpp *.cc)
# 收集源文件
file(GLOB_RECURSE BC "basic_constraint/*.h" "basic_constraint/*.cpp" "basic_constraint/*.hpp" "basic_constraint/*.cc")
file(GLOB_RECURSE CD "constrain_definition/*.h" "constrain_definition/*.cpp" "constrain_definition/*.hpp" "constrain_definition/*.cc")
file(GLOB_RECURSE CF "cost_function/*.h" "cost_function/*.cpp" "cost_function/*.hpp" "cost_function/*.cc")
file(GLOB_RECURSE FC "frenet_coordinate/*.h" "frenet_coordinate/*.cpp" "frenet_coordinate/*.hpp" "frenet_coordinate/*.cc")
file(GLOB_RECURSE M "model/*.h" "model/*.cpp" "model/*.hpp" "model/*.cc")
file(GLOB_RECURSE S "augmented_ilqr/*.h" "augmented_ilqr/*.cpp" "augmented_ilqr/*.hpp" "augmented_ilqr/*.cc")
file(GLOB_RECURSE AL "augmented_ilqr/polished_altro/*.h" "augmented_ilqr/polished_altro/*.cpp"
        "augmented_ilqr/polished_altro/*.hpp" "augmented_ilqr/polished_altro/*.cc")

# 手动排除不必要的文件
list(FILTER BC EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER CD EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER CF EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER FC EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER M EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER S EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER AL EXCLUDE REGEX ".*CMakeFiles/.*")


include_directories(constrain_definition)
include_directories(cost_function)
include_directories(basic_constraint)
include_directories(model)
include_directories(augmented_ilqr)
include_directories(augmented_ilqr/polished_altro)
include_directories(frenet_coordinate)

set(Header
        calculus.h
        ilqr_system_declaration.hpp
        matplotlibcpp.h
        clock.hpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

if (${PYTHON_NUMPY_FOUND})
    target_link_directories(${PROJECT_NAME} INTERFACE ${PYTHON_NUMPY_INCLUDE_DIR})
else()
    message(WARNING "Python3 NumPy not found, proceeding with -DWITHOUT_NUMPY."
            " Some functions might not work.")
    add_definitions(-DWITHOUT_NUMPY)
endif()

if ( ${Python3_FOUND} )
    include_directories(${Python3_INCLUDE_DIRS})
else()
    message ( FATAL_ERROR "Python3 not found, please install it." )
endif()

add_executable(${PROJECT_NAME} main.cpp ${Header}
        ${BC}
        ${CD}
        ${CF}
        ${FC}
        ${M}
        ${S}
        ${AL})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen ${Python3_LIBRARIES})

