Building and running OpenVINO Model Server (OVMS) natively on Windows

This document explains a minimal native Windows build and run workflow for OVMS.

Prerequisites
- Visual Studio 2019/2022 with "Desktop development with C++" workload.
- CMake (>=3.20)
- Git
- Python 3.8+
- OpenVINO runtime for Windows (installed and path noted)

High-level steps
1. Clone the model_server repository.
2. Configure the build with CMake from a Visual Studio Developer PowerShell (x64).
3. Build with CMake/MSBuild.
4. Run ovms.exe pointing to your exported models config.json.

Example commands (Developer PowerShell for VS, x64)
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
mkdir build
cd build
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release -DOPENVINO_DIR="C:/Program Files (x86)/Intel/openvino_2025" -G "Visual Studio 17 2022"
cmake --build . --config Release -- /m

Run the server (adjust path to your build and config):
"C:\path\to\build\bin\Release\ovms.exe" --rest_port 8000 --config_path C:\path\to\models\config.json

Health check (PowerShell):
curl http://localhost:8000/v3/health

Quick completion test (PowerShell):
$body = '{"model":"my-model","prompt":"Hello world","max_tokens":2}'
curl -X POST http://localhost:8000/v3/completions -H "Content-Type: application/json" -d $body

Notes and troubleshooting
- Ensure OpenVINO runtime libraries are available (add OpenVINO bin to PATH or set OPENVINO_DIR during configure).
- If ports are blocked, allow 8000 in Windows Firewall.
- If CMake cannot find OpenVINO, set the OPENVINO_DIR to your installed path.

References
- https://github.com/openvinotoolkit/model_server
- Intel OpenVINO Windows documentation
