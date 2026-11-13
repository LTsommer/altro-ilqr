cmake_minimum_required(VERSION 3.24)
project(alilqr)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O2)

# Eigen 使用随仓库提交的 third_party/eigen（3.4.0，纯头文件），不需要系统安装。
# 想换成别的版本时传 -DEIGEN3_INCLUDE_DIR=/path/to/eigen（该目录下要有 Eigen/Dense）。
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/eigen" CACHE PATH
        "Directory that contains Eigen/Dense")
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR "Eigen/Dense not found under ${EIGEN3_INCLUDE_DIR}")
endif()
message(STATUS "Using Eigen at ${EIGEN3_INCLUDE_DIR}")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
# SYSTEM：不对 Eigen 头文件报编译警告
target_include_directories(Eigen3::Eigen SYSTEM INTERFACE "${EIGEN3_INCLUDE_DIR}")

#file(GLOB_RECURSE BC basic_constraint *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE CD constraint_definition *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE CF cost_function *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE FC frenet_coordinate *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE M model *.h *.cpp *.hpp *.cc)
#file(GLOB_RECURSE S solver *.h *.cpp *.hpp *.cc)
# 收集源文件
file(GLOB_RECURSE BC "basic_constraint/*.h" "basic_constraint/*.cpp" "basic_constraint/*.hpp" "basic_constraint/*.cc")
file(GLOB_RECURSE CD "constraint_definition/*.h" "constraint_definition/*.cpp" "constraint_definition/*.hpp" "constraint_definition/*.cc")
file(GLOB_RECURSE CF "cost_function/*.h" "cost_function/*.cpp" "cost_function/*.hpp" "cost_function/*.cc")
file(GLOB_RECURSE FC "frenet_coordinate/*.h" "frenet_coordinate/*.cpp" "frenet_coordinate/*.hpp" "frenet_coordinate/*.cc")
file(GLOB_RECURSE M "model/*.h" "model/*.cpp" "model/*.hpp" "model/*.cc")
file(GLOB_RECURSE S "augmented_ilqr/*.h" "augmented_ilqr/*.cpp" "augmented_ilqr/*.hpp" "augmented_ilqr/*.cc")
file(GLOB_RECURSE AL "augmented_ilqr/polished_altro/*.h" "augmented_ilqr/polished_altro/*.cpp"
        "augmented_ilqr/polished_altro/*.hpp" "augmented_ilqr/polished_altro/*.cc")
file(GLOB_RECURSE MP "motion_planning/*.h" "motion_planning/*.cpp" "motion_planning/*.hpp" "motion_planning/*.cc"
        "motion_planning/constraints/*.h" "motion_planning/constraints/*.hpp" "motion_planning/constraints/*.cpp"
        "motion_planning/cost_functions/*.h" "motion_planning/cost_functions/*.hpp" "motion_planning/cost_functions/*.cpp")

# 手动排除不必要的文件
list(FILTER BC EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER CD EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER CF EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER FC EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER M EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER S EXCLUDE REGEX ".*CMakeFiles/.*")
list(FILTER AL EXCLUDE REGEX ".*CMakeFiles/.*")

set(PROJECTION_SOLVER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/augmented_ilqr/polished_altro/projection_solver.cc")
set(SOLVER_CONFIG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/augmented_ilqr/solver_config.cpp")
list(REMOVE_ITEM S "${PROJECTION_SOLVER_SOURCE}" "${SOLVER_CONFIG_SOURCE}")
list(REMOVE_ITEM AL "${PROJECTION_SOLVER_SOURCE}")
add_subdirectory(../third_party/qdldl "${CMAKE_CURRENT_BINARY_DIR}/qdldl" EXCLUDE_FROM_ALL)
add_library(projection_solver STATIC "${PROJECTION_SOLVER_SOURCE}")
target_link_libraries(projection_solver PUBLIC Eigen3::Eigen PRIVATE qdldlstatic)
add_library(altro_solver_assembly STATIC "${SOLVER_CONFIG_SOURCE}")
target_link_libraries(altro_solver_assembly PUBLIC projection_solver)

include_directories(constraint_definition)
include_directories(cost_function)
include_directories(basic_constraint)
include_directories(model)
include_directories(augmented_ilqr)
include_directories(augmented_ilqr/polished_altro)
include_directories(frenet_coordinate)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set(Header
        common/calculus.h
        common/trig_table.h
        common/math_utils.h
        ilqr_system_declaration.hpp
        csv_writer.hpp
        clock.hpp)

add_executable(${PROJECT_NAME} main.cpp ${Header}
        ${BC}
        ${CD}
        ${CF}
        ${FC}
        ${M}
        ${S}
        ${AL})
target_link_libraries(${PROJECT_NAME} PRIVATE altro_solver_assembly)

set(mp MOTION_PLANNING)

add_executable(mp motion_planning.cpp
        ${Header}
        ${BC}
        ${CD}
        ${CF}
        ${FC}
        ${M}
        ${S}
        ${AL}
        ${MP})
target_include_directories(mp PUBLIC motion_planning)
target_link_libraries(mp PRIVATE altro_solver_assembly)

foreach(solver_app IN ITEMS alilqr mp)
    set(default_config "${CMAKE_CURRENT_BINARY_DIR}/${solver_app}.conf")
    configure_file("config/${solver_app}.conf" "${default_config}" COPYONLY)
    target_sources(${solver_app} PRIVATE common/solver_cli.cpp)
    target_compile_definitions(${solver_app} PRIVATE
            ALTRO_DEFAULT_CONFIG="${default_config}")
endforeach()

add_executable(test test_constraint.cpp
        ${Header}
        ${BC}
        ${CD}
        ${CF}
        ${FC}
        ${M}
        ${S}
        ${AL})
target_link_libraries(test PUBLIC Eigen3::Eigen)

add_executable(test_projection test_projection.cpp)
target_link_libraries(test_projection PRIVATE projection_solver)
