cmake_minimum_required(VERSION 3.24)
project(alilqr)

set(CMAKE_CXX_STANDARD 17)

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

add_subdirectory(../third_party/qdldl "${CMAKE_CURRENT_BINARY_DIR}/qdldl" EXCLUDE_FROM_ALL)

add_library(projection_solver STATIC augmented_ilqr/polished_altro/projection_solver.cc)
target_include_directories(projection_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(projection_solver PUBLIC Eigen3::Eigen PRIVATE qdldlstatic)

add_library(altro_solver_assembly STATIC augmented_ilqr/solver_config.cpp)
target_link_libraries(altro_solver_assembly PUBLIC projection_solver)

add_library(altro_solver STATIC
        augmented_ilqr/alilqr_solver.cpp
        augmented_ilqr/cost_calc.cpp
        augmented_ilqr/ilqr_process.cpp
        augmented_ilqr/polished_altro/altro_solver.cpp
        basic_constraint/basic_constraint.cpp
        constraint_definition/constraint.cpp
        constraint_definition/constraint_formulation.cpp
        constraint_definition/constraint_values.cpp
        cost_function/cost_terms.cpp
        frenet_coordinate/cartesian_coordinate_system.cc
        frenet_coordinate/frenet_coordinate_system.cc
        frenet_coordinate/spline.cc
        frenet_coordinate/vec2d.cc
        model/smooth_kappa_model.cc
        motion_planning/constraints/constraints.cpp
        motion_planning/cost_functions/cost_terms.cpp
        motion_planning/mix_model.cc)
target_link_libraries(altro_solver PUBLIC altro_solver_assembly)

add_executable(alilqr main.cpp)
add_executable(mp motion_planning.cpp)

foreach(solver_app IN ITEMS alilqr mp)
    set(default_config "${CMAKE_CURRENT_BINARY_DIR}/${solver_app}.conf")
    configure_file("config/${solver_app}.conf" "${default_config}" COPYONLY)
    target_sources(${solver_app} PRIVATE common/solver_cli.cpp)
    target_compile_definitions(${solver_app} PRIVATE
            ALTRO_DEFAULT_CONFIG="${default_config}")
    target_link_libraries(${solver_app} PRIVATE altro_solver)
endforeach()

add_executable(test_constraint test_constraint.cpp)
# Keep the existing executable name. CTest reserves the build target 'test'.
set_target_properties(test_constraint PROPERTIES OUTPUT_NAME test)
target_link_libraries(test_constraint PRIVATE altro_solver)

add_executable(test_projection test_projection.cpp)
target_link_libraries(test_projection PRIVATE projection_solver)

foreach(target IN ITEMS projection_solver altro_solver_assembly altro_solver alilqr mp test_constraint test_projection)
    target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
            $<$<CXX_COMPILER_ID:MSVC>:/W3>)
endforeach()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME constraints COMMAND test_constraint)
    add_test(NAME projection COMMAND test_projection)
    if(UNIX)
        foreach(case IN ITEMS missing-terminal-boundary missing-terminal-boundary-update
                missing-boundary-endpoints missing-cost-union invalid-step-size
                invalid-initial-heading missing-comfort-acceleration nonfinite-initial-rollout)
            add_test(NAME ${case} COMMAND "${CMAKE_COMMAND}"
                    -DTEST_PROGRAM=$<TARGET_FILE:test_constraint> -DTEST_CASE=${case}
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_expect_abort.cmake")
        endforeach()
    endif()
endif()
