$ErrorActionPreference = "Stop"

$appPath = Join-Path (Get-Location) "src\App.tsx"
if (-not (Test-Path $appPath)) {
  throw "Could not find src\App.tsx. Run this script from the repository root."
}

$content = Get-Content -Raw -Encoding UTF8 $appPath

$content = $content -replace '<h2>NPC World.*?RL</h2>', '<h2>NPC World - Genetic Memory + RL</h2>'
$content = [regex]::Replace(
  $content,
  'Start a new .*?world, or load a previously .*?saved world\s+\(JSON file\)\.',
  'Start a new world, or load a previously saved world from a JSON file.',
  [System.Text.RegularExpressions.RegexOptions]::Singleline
)
$content = $content -replace '<strong>Last Rule .*?</strong>\{" "\}', '<strong>Last Rule:</strong>{" "}'
$content = $content -replace 'Click an agente in the grid to inspect its .*?\.', 'Click an agent in the grid to inspect its genetic memory.'
$content = $content -replace '<h3>Statistics .*?</h3>', '<h3>Statistics</h3>'
$content = $content -replace '<h3>Agent Inspector .*?</h3>', '<h3>Agent Inspector</h3>'
$content = $content -replace '<h3>Population Over Time .*?</h3>', '<h3>Population Over Time</h3>'
$content = $content -replace 'Trait Distribution.*', 'Trait Distribution'
$content = $content -replace '<h3>Action Log .*?</h3>', '<h3>Action Log</h3>'
$content = $content -replace 'and ate food \(\+5 energia\)', 'and ate food (+5 energy)'
$content = $content -replace 'energia now', 'energy now'
$content = $content -replace 'ran out of energia', 'ran out of energy'
$content = $content -replace 'energia \$\{childEnergy\}', 'energy ${childEnergy}'
$content = $content -replace 'agente may split energia', 'agent may split energy'

Set-Content -Encoding UTF8 $appPath $content
Write-Host "Updated src\App.tsx to use English-only dashboard labels."
