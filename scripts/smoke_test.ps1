# Calls one read-only endpoint per microservice through the gateway.
# Usage:  .\scripts\smoke_test.ps1                      (local docker compose)
#         .\scripts\smoke_test.ps1 http://<ec2-ip>:8080  (deployed server)
# Uses a fake user and email, so it never changes real data.
param([string]$BaseUrl = "http://localhost:8080")

$tests = @(
    @{ Service = "gateway"; Path = "/gateway/health"; Body = $null },
    @{ Service = "auth";    Path = "/registrationRoute/check_email_exists";       Body = @{ email = "smoke-test-nobody@example.com" } },
    @{ Service = "profile"; Path = "/netwirkProfileRoute/get_country_list";        Body = @{ country = "" } },
    @{ Service = "geo";     Path = "/netwirkProfileRoute/get_industry_user_count"; Body = @{ placeId = "smoke-test"; userId = "0"; eventId = "0" } },
    @{ Service = "social";  Path = "/netwirkProfileRoute/get_total_likes";         Body = @{ userId = 0; placeId = "smoke-test" } },
    @{ Service = "recsys";  Path = "/netwirkProfileRoute/get_later_remaining";     Body = @{ userId = 0 } },
    @{ Service = "legacy";  Path = "/health";                                      Body = $null }
)

$failed = 0
foreach ($test in $tests) {
    $url = "$BaseUrl$($test.Path)"
    try {
        if ($null -eq $test.Body) {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 30
        } else {
            $response = Invoke-WebRequest -Uri $url -Method Post -ContentType "application/json" `
                -Body ($test.Body | ConvertTo-Json -Compress) -UseBasicParsing -TimeoutSec 30
        }
        $ok = $response.StatusCode -eq 200 -and $response.Content.Trim() -ne '"Error"' -and $response.Content.Trim() -ne 'Error'
        $detail = "$($response.StatusCode)"
    } catch {
        $ok = $false
        $detail = $_.Exception.Message
    }
    if (-not $ok) { $failed++ }
    "{0,-5} {1,-8} {2}  ({3})" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $test.Service, $test.Path, $detail
}

if ($failed) {
    "`n$failed check(s) failed. See: docker compose logs <service>"
    exit 1
}
"`nAll services OK."
