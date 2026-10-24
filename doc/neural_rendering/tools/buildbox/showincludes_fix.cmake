# Local fix for a non-English MSVC with sccache: sccache drops the non-ASCII byte of cl's /showIncludes prefix
# ("Nota: inclusión del archivo: " arrives as "Nota: inclusin del archivo: "), so Ninja finds no header
# dependencies. Make the generator expect the text that actually arrives. Used via -DCMAKE_PROJECT_INCLUDE=.
set(_nr_prefix "Nota: inclusin del archivo: ")
set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "${_nr_prefix}")
set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "${_nr_prefix}")
set(CMAKE_CL_SHOWINCLUDES_PREFIX "${_nr_prefix}")
