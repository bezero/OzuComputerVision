cmake_minimum_required( VERSION 2.4 ) 
PROJECT ( OzuComputerVision )

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    SET(CRT_SECURE_NO_WARNINGS ON)
    SET(LAMBDAEXPRESSIONS ON)
    SET(UNICODE ON)
    SET(QT_LARGEFILE_SUPPORT ON)

    ADD_DEFINITIONS(-DNOMINMAX)

    IF(HAVE_OPENMP)
        SET(FLAGS_OPENMP "/openmp")
    ENDIF()
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MP /Ob1 /Oy- /GF ${FLAGS_OPENMP}")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MP /Ox /Ob2 /Oi /Ot /Oy /GL /GF /Gm- /GS- /Gy /fp:fast ${FLAGS_OPENMP}")
    IF(MSVC32)
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:SSE2")
    ENDIF()

elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(FLAGS_VISIBILITY "-fvisibility=hidden -fvisibility-inlines-hidden")
    IF(WITH_AVX)
        SET(AVXFLAGS "-march=native -Wa,-q -fabi-version=0")
    ENDIF()
    IF(WITH_INSPECTION)
        SET(FLAGS_INSPECTION "-fsanitize=address -fno-omit-frame-pointer")
    ENDIF()

    IF(HAVE_OPENMP)
        SET(FLAGS_OPENMP "-fopenmp")
    ENDIF()

    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function -Werror=return-type -march=native -Wa,-q ${FLAGS_OPENMP}")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Werror=return-type -std=c++11 -g -m64 ${FLAGS_OPENMP} ${AVXFLAGS} ${FLAGS_VISIBILITY} ${FLAGS_INSPECTION}")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Werror=return-type -std=c++11 -O2 -g -finline-functions -m64 -mtune=native ${FLAGS_OPENMP} ${AVXFLAGS} ${FLAGS_VISIBILITY}")

elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    SET(FLAGS_VISIBILITY "-fvisibility=hidden -fvisibility-inlines-hidden")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -g -m64 -fvisibility=hidden")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O2 -g -finline-functions -m64 -mtune=native ${FLAGS_VISIBILITY}")
endif()

set( NAME_SRC particle_filter.cpp particle_filter.h)

find_package(OpenCV REQUIRED core imgproc highgui imgcodecs)

include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories( ${CMAKE_BINARY_DIR}/bin )

add_executable( OzuComputerVision ${NAME_SRC} )
target_link_libraries( OzuComputerVision ${OpenCV_LIBS})
