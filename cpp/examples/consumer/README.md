# Consumer example

A minimal, self-contained project showing how a downstream codebase links the header-only
`cfsonsphere` library. It is **not** part of the main `cpp/` build — copy this directory (or
just its `CMakeLists.txt`) into your own project.

```sh
cd cpp/examples/consumer
cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew   # so find_package(Eigen3) succeeds
cmake --build build
./build/consumer
```

`CMakeLists.txt` shows two ways to obtain the library:

- **Option A (default):** `add_subdirectory(.../cpp ...)` against a vendored copy of the tree.
  The library's own tests and samplers auto-disable when it is added as a subproject
  (`CFSONSPHERE_BUILD_TESTS`/`CFSONSPHERE_BUILD_EXAMPLES` default OFF off-top-level), so only
  the `cfsonsphere::cfsonsphere` INTERFACE target is configured.
- **Option B:** `FetchContent` with `SOURCE_SUBDIR cpp` to pull it straight from git.

Either way you just link `cfsonsphere::cfsonsphere`; it carries the include path, the C++17
requirement, and the Eigen dependency transitively.
