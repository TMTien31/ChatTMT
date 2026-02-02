# Run ONLY e2e conversation tests (slow, costs ~$1-2)
# Usage: .\run_e2e_tests.ps1

Write-Host "`n=== Running E2E Conversation Tests ===" -ForegroundColor Cyan
Write-Host "WARNING: This will cost ~`$1-2 in OpenAI API calls" -ForegroundColor Yellow
Write-Host "Estimated time: 5-10 minutes`n" -ForegroundColor Yellow

$confirmation = Read-Host "Continue? (y/n)"
if ($confirmation -ne 'y') {
    Write-Host "`nAborted." -ForegroundColor Red
    exit 1
}

Write-Host "`nRunning 3 conversation tests (25 turns each)...`n" -ForegroundColor Green

pytest tests/test_e2e.py::TestConversationLogs -v -s

$exitCode = $LASTEXITCODE

Write-Host "`n=== E2E Test Summary ===" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "[PASS] Conversation logs generated in data/sessions/" -ForegroundColor Green
    Write-Host "Check the 3 newest .json files for test data" -ForegroundColor Yellow
} else {
    Write-Host "[FAIL] E2E tests failed (exit code: $exitCode)" -ForegroundColor Red
}

exit $exitCode
