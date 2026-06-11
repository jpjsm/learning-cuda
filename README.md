# Learning CUDA

This is a place to start learning CUDA, both from C++ and python at the same time.

## Motivation for this workspace (repo)

In my own words: "I want to setup a workspace in my laptop to learn CUDA with Microsoft C++ and Python, using VSCode as my IDE, where I can run the examples in 'Programming in Parallel with CUDA, A Practical Guide' by Richard Ansorge.
Ansorge's book comes with its own set of include files (that can be downloaded to any location in the workspace you are designing); adding those include files should not disrupt the include files provided by Microsoft C++.
I'll be porting the CUDA examples, in the book, to *.cu files; and I would NVCC to be part of the workspace."

Also, I want to merge my two other repos about learning CUDA into one place with everything:

- [Understanding CUDA](<https://github.com/jpjsm/understanding_cuda>)
- [Understanding Python CUDA](<https://github.com/jpjsm/understanding-python-cuda>)

As of today 2026-06-04, for most ML and AI workflows, in Python, it's better to install CUDA 12.4 toolkit.

## Setup

### Prerequisites to Install

#### CUDA Toolkit

- Download CUDA 12.4.1 Toolkit from [cuda-toolkit-archive](<https://developer.nvidia.com/cuda-toolkit-archive>).
- This installs NVCC and the CUDA runtime.
  - During install, make sure "CUDA -> Development -> Compiler" is checked.
- After install, verify: `nvcc --version`

#### Visual Studio Build Tools (MSVC)

Install [Visual Studio Build Tools](<https://visualstudio.microsoft.com/visual-cpp-build-tools/>) — you do not need the full VS IDE.
Select the "Desktop development with C++" workload.

NVCC on Windows requires MSVC as its host compiler.

#### VSCode Extensions

- C/C++ (Microsoft) — IntelliSense, debugging
- CUDA C++ (NVIDIA or community) — .cu syntax highlighting
- CMake Tools (Microsoft) — optional but recommended for multi-file projects

### Workspace Folder Structure

```txt
cuda-workspace/
├── .vscode/
│   ├── c_cpp_properties.json   ← IntelliSense config
│   ├── tasks.json              ← Build tasks (nvcc)
│   └── launch.json             ← Debug config
├── ansorge-includes/           ← Ansorge's book headers go here
│   └── *.h / *.cuh
├── examples/
│   ├── ch01/
│   │   └── hello.cu
│   ├── ch02/
│   └── ...
└── CMakeLists.txt              ← Optional, for CMake builds
```

Ansorge's headers live in their own subfolder, completely isolated from MSVC's system headers. You pass their path explicitly to NVCC — they never collide.

#### `.vscode/c_cpp_properties.json`

```json
{
  "configurations": [
    {
      "name": "CUDA Windows",
      "includePath": [
        "${workspaceFolder}/**",
        "${workspaceFolder}/ansorge-includes",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include",
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/include",
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt"
      ],
      "defines": ["_WIN64", "__CUDACC__"],
      "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe",
      "cStandard": "c17",
      "cppStandard": "c++17",
      "intelliSenseMode": "windows-msvc-x64"
    }
  ],
  "version": 4
}
```

> **Warning**: Replace v12.x and 14.xx.xxxxx with your actual installed versions. Run dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" and dir "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC" to find them.

> **Warning**: If you have the full VS 2022 Community (not Build Tools), the path shifts from BuildTools to Community: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\`
>
> Two differences to note:
>
> - It installs to Program Files (64-bit), not Program Files (x86)
> - The folder name is Community instead of BuildTools

#### `.vscode/tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "NVCC: Build active .cu file",
      "type": "shell",
      "command": "nvcc",
      "args": [
        "-o", "${fileDirname}/${fileBasenameNoExtension}.exe",
        "${file}",
        "-I", "${workspaceFolder}/ansorge-includes",
        "-arch=sm_86",          // ← change to your GPU's compute capability; it's OK for RTX 30xx
        "-std=c++17",
        "-Xcompiler", "/EHsc",  // MSVC exception handling
        "-g", "-G"              // debug symbols; remove for release builds
      ],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "problemMatcher": ["$nvcc"],
      "detail": "Compile active .cu file with NVCC + MSVC host compiler"
    }
  ]
}
```
