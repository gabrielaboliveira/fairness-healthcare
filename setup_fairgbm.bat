@echo off
setlocal enabledelayedexpansion

:: === CONFIGURAÇÕES INICIAIS ===
set REPO_DIR=%cd%\fairgbm
set BUILD_DIR=%REPO_DIR%\build
set PYTHON_PACKAGE=%REPO_DIR%\python-package
set DLL_TARGET=%PYTHON_PACKAGE%\fairgbm\lib_lightgbm.dll

:: Verifica se foi passado "overwrite" como argumento
set OVERWRITE=false
if /I "%1"=="overwrite" set OVERWRITE=true

echo.
echo === CLONANDO O REPOSITORIO FAIRGBM ===
if exist "%REPO_DIR%" (
    if "%OVERWRITE%"=="true" (
        echo Diretório fairgbm já existe. Apagando...
        rmdir /s /q "%REPO_DIR%"
        git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
    ) else (
        echo ⚠️  Diretório fairgbm já existe. Pulando o clone.
    )
) else (
    git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
)
if errorlevel 1 (
    echo ❌ ERRO ao clonar o repositório. Abortando.
    exit /b 1
)

echo.
echo === AJUSTANDO TOKENS LÓGICOS PARA COMPATIBILIDADE COM MSVC ===
python substituir_tokens_cpp.py

echo.
echo === CRIANDO PASTA BUILD ===
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
) else (
    echo ⚠️  Build já existe em %BUILD_DIR%.
)
cd "%BUILD_DIR%"

echo.
echo === CONFIGURANDO CMAKE ===
cmake .. -DUSE_CUDA=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
if errorlevel 1 (
    echo ❌ ERRO ao rodar CMake. Verifique a instalação do Visual Studio e do CMake.
    exit /b 1
)

echo.
echo === COMPILANDO _lightgbm ===
cmake --build . --target _lightgbm --config Release
if errorlevel 1 (
    echo ❌ ERRO ao compilar o projeto.
    exit /b 1
)

echo.
echo === INSTALANDO PACOTE COM PIP ===
cd "%PYTHON_PACKAGE%"
pip uninstall -y fairgbm lightgbm
pip install -e .
if errorlevel 1 (
    echo ❌ ERRO ao instalar com pip.
    exit /b 1
)

echo.
echo === COPIANDO DLL PARA O LOCAL ESPERADO PELO PYTHON ===
copy "%REPO_DIR%\Release\lib_lightgbm.dll" "%DLL_TARGET%"
if errorlevel 1 (
    echo ❌ ERRO ao copiar lib_lightgbm.dll
    exit /b 1
) else (
    echo ✅ lib_lightgbm.dll copiado com sucesso.
)

echo.
echo === TUDO CONCLUÍDO COM SUCESSO ===
endlocal
pause
