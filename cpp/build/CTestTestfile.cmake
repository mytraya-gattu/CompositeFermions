# CMake generated Testfile for 
# Source directory: /Users/aragorn/Documents/CompositeFermions/cpp
# Build directory: /Users/aragorn/Documents/CompositeFermions/cpp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_cfsonsphere "/Users/aragorn/Documents/CompositeFermions/cpp/build/test_cfsonsphere")
set_tests_properties(test_cfsonsphere PROPERTIES  _BACKTRACE_TRIPLES "/Users/aragorn/Documents/CompositeFermions/cpp/CMakeLists.txt;22;add_test;/Users/aragorn/Documents/CompositeFermions/cpp/CMakeLists.txt;0;")
add_test(test_reference "/Users/aragorn/Documents/CompositeFermions/cpp/build/test_reference" "/Users/aragorn/Documents/CompositeFermions/cpp/reference")
set_tests_properties(test_reference PROPERTIES  _BACKTRACE_TRIPLES "/Users/aragorn/Documents/CompositeFermions/cpp/CMakeLists.txt;27;add_test;/Users/aragorn/Documents/CompositeFermions/cpp/CMakeLists.txt;0;")
add_test(test_logpsi "/Users/aragorn/Documents/CompositeFermions/cpp/build/test_logpsi" "/Users/aragorn/Documents/CompositeFermions/cpp/reference")
set_tests_properties(test_logpsi PROPERTIES  _BACKTRACE_TRIPLES "/Users/aragorn/Documents/CompositeFermions/cpp/CMakeLists.txt;32;add_test;/Users/aragorn/Documents/CompositeFermions/cpp/CMakeLists.txt;0;")
