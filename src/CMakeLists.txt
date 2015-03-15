cmake_minimum_required( VERSION 2.4 ) 
PROJECT ( particle_filter_proj )

set( NAME_SRC  particle_filter.cpp )

find_package( OpenCV REQUIRED core imgproc highgui )

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories( ${CMAKE_BINARY_DIR}/bin )

add_executable( particle_filter ${NAME_SRC} )
target_link_libraries( particle_filter ${OpenCV_LIBS})