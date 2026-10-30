# Use the bundled runtime and the project's existing YAML package.
$pythonRuntime = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $pythonRuntime)) {
    $pythonRuntime = (Get-Command python -ErrorAction Stop).Source
}
$savedPythonPath = $env:PYTHONPATH
$localDependencies = Join-Path $PSScriptRoot '.venv\Lib\site-packages'
try {
    if (Test-Path -LiteralPath (Join-Path $localDependencies 'yaml\__init__.py')) {
        $env:PYTHONPATH = $localDependencies
        if ($savedPythonPath) {
            $env:PYTHONPATH += [IO.Path]::PathSeparator + $savedPythonPath
        }
    }
    & $pythonRuntime (Join-Path $PSScriptRoot 'run_experiment_2.py') @args
    $runnerExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $savedPythonPath
}
exit $runnerExitCode
