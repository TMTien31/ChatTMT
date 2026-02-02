# Run all tests EXCEPT e2e (fast, no API cost for most tests)
# Usage: .\run_tests_fast.ps1

Write-Host "`n=== Running Fast Tests (Excluding E2E) ===" -ForegroundColor Cyan
Write-Host "This will run all unit tests, most with mocked API calls`n" -ForegroundColor Yellow

pytest tests/ -v --ignore=tests/test_e2e.py

$exitCode = $LASTEXITCODE

Write-Host "`n=== Test Summary ===" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "[PASS] All fast tests passed" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Some tests failed (exit code: $exitCode)" -ForegroundColor Red
}
Write-Host "[SKIP] E2E tests skipped (run separately with .\run_e2e_tests.ps1)" -ForegroundColor Yellow

exit $exitCode
