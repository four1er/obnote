---
{"dg-publish":true,"permalink":"/杂项/llvm build/","title":"llvm build"}
---


Cmake Build三部曲

现代 CMake 提供了更方便的 -B 和 --build 指令，不同平台，统一命令！

```
cmake -B build # 在源码目录用 -B 直接创建 build 目录并生成 build/Makefile
cmake --build build -j4 # 自动调用本地的构建系统在 build 里构建，即：make -C build -j4
sudo cmake --build build --target install # 调用本地的构建系统执行 install 这个目标，即：安装
```

cmake -B build 免去了先创建 build 目录再切换进去再指定源码目录的麻烦。
cmake --build build统一了不同平台（Linux 上会调用 make，Windows 上调用 devenv.exe）
结论：从现在开始，如果在命令行操作 cmake，请使用更方便的 -B 和 --build 命令。

LLVM build

```shell
git clone https://github.com/llvm/llvm-project.git
cd llvm-project  
mkdir build  
cd build  
cmake -DLLVM_ENABLE_PROJECTS=clang -G “Unix Makefiles” …/llvm  
#或者  
#cmake -DLLVM_ENABLE_PROJECTS=clang -G “Unix Makefiles” -G Xcode …/llvm  
make -j10
```

MLIR build
```shell
cd llvm-project
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -G "Unix Makefiles" ../llvm -DCMAKE_BUILD_TYPE=Debug

cmake --build . -- ${MAKEFLAGS} # 等待编译完成
cmake --build . --target check-mlir
```

Build Serene
```shell
cmake -G "Unix Makefiles"  ../llvm \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_ENABLE_PROJECTS='clang;lldb;lld;mlir;clang-tools-extra;compiler-rt' \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLDB_INCLUDE_TESTS=OFF

------------------------------------------------------

cmake --build .

cmake --build . --target check-mlir

cmake -DCMAKE_INSTALL_PREFIX=/your/target/location -P cmake_install.cmake

cmake -DCMAKE_INSTALL_PREFIX=/usr/local/bin -P cmake_install.cmake
```