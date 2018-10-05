REM Install packages using pip
set PATH=%PYTHON%;%PYTHON%\Scripts;%PATH%
python -m pip install -U pip

IF Defined NUMPY (
    IF Defined SCIPY (
        pip install numpy==%NUMPY% scipy==%SCIPY% cython pandas nose patsy
    ) else (
        pip install numpy==%NUMPY% scipy cython pandas nose patsy
    )
) else (
    IF Defined SCIPY (
        pip install numpy scipy==%SCIPY% cython pandas nose patsy
    ) else (
        pip install numpy scipy cython pandas nose patsy
    )
)
