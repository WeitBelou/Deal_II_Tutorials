cmake_minimum_required(VERSION 3.5)
project(Deal_II_Tutorials)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#Find deal.II
find_package(deal.II REQUIRED
        HINTS $ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()

file(GLOB_RECURSE STEPS_SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/" *.cpp)

foreach(SOURCE ${STEPS_SOURCES})
    get_filename_component(filename ${SOURCE} NAME)
    add_executable(${filename} ${SOURCE})
    deal_ii_setup_target(${filename})
endforeach()
