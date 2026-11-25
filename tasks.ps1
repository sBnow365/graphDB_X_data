# <#
# Windows task runner (PowerShell). Usage:  .\tasks.ps1 <task>
# If scripts are blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# Tasks: setup, synthetic, preprocess, classify, offline-demo, neo4j-up, neo4j-down, init,
#        ingest, gds, communities, temporal, ask, check-llm, demo, test, test-integration,
#        eval, results, clean-synthetic
# #>
# param([Parameter(Position=0)][string]$Task = "help", [Parameter(Position=1)][string]$Arg = "")
# $ErrorActionPreference = "Stop"
# $py = ".\.venv\Scripts\python.exe"
# function Run($cmd) { Write-Host ">> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "Failed: $cmd" } }

# switch ($Task) {
#   "setup" {
#     if (-not (Test-Path .venv)) { Run "py -3.11 -m venv .venv" }
#     Run "$py -m pip install --upgrade pip"
#     Run "$py -m pip install -r requirements.txt"
#     Run "$py -m pip install -e ."
#     if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - edit NEO4J_PASSWORD now." -ForegroundColor Yellow }
#   }
#   "synthetic"      { Run "$py scripts/generate_synthetic_data.py" }
#   "preprocess"     { Run "$py scripts/preprocess_dataset.py --dataset synthetic" }
#   "classify"       { Run "$py scripts/classify_posts.py --dataset synthetic" }
#   "offline-demo"   { Run "$py scripts/run_demo.py --offline" }
#   "neo4j-up"       { Run "docker compose --env-file .env up -d"; Write-Host "Browser: http://localhost:7474 (wait ~30-60 s on first start)" }
#   "neo4j-down"     { Run "docker compose down" }
#   "init"           { Run "$py scripts/initialize_neo4j.py" }
#   "ingest"         { Run "$py scripts/ingest_graph.py --dataset synthetic" }
#   "gds"            { Run "$py scripts/run_graph_algorithms.py" }
#   "communities"    { Run "$py scripts/run_community_analysis.py" }
#   "temporal"       { Run "$py scripts/run_temporal_analysis.py --source csv"; Run "$py scripts/run_temporal_analysis.py --source neo4j" }
#   "check-llm"      { Run "$py scripts/check_llm.py" }
#   "ask"            { Run "$py scripts/ask.py `"$Arg`"" }
#   "demo"           { Run "$py scripts/run_demo.py" }
#   "test"           { Run "$py -m pytest -q" }
#   "test-integration" { $env:RUN_INTEGRATION = "1"; Run "$py -m pytest -q -m integration"; Remove-Item Env:RUN_INTEGRATION }
#   "eval" {
#     Run "$py scripts/evaluate_text_to_cypher.py --safety-only"
#     Run "$py scripts/evaluate_graph_rag.py --routing-only"
#   }
#   "results"        { Run "$py scripts/run_baselines.py" }
#   "clean-synthetic" { Remove-Item data\synthetic\*.csv, data\processed\synthetic -Recurse -ErrorAction SilentlyContinue }
#   default { Get-Help $MyInvocation.MyCommand.Path; Write-Host "See the comment block at the top of tasks.ps1 for task names." }
# }


<#
Windows task runner (PowerShell). Usage:  .\tasks.ps1 <task>
If scripts are blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
Tasks: setup, dashboard, synthetic, preprocess, classify, offline-demo, neo4j-up, neo4j-down, init,
       ingest, gds, communities, temporal, ask, check-llm, demo, test, test-integration,
       eval, results, clean-synthetic
#>
param([Parameter(Position=0)][string]$Task = "help", [Parameter(Position=1)][string]$Arg = "")
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
function Run($cmd) { Write-Host ">> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "Failed: $cmd" } }

switch ($Task) {
  "setup" {
    if (-not (Test-Path .venv)) { Run "py -3.11 -m venv .venv" }
    Run "$py -m pip install --upgrade pip"
    Run "$py -m pip install -r requirements.txt"
    Run "$py -m pip install -e ."
    if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - edit NEO4J_PASSWORD now." -ForegroundColor Yellow }
  }
  "dashboard"      { Run "$py -m streamlit run app/streamlit_app.py" }
  "synthetic"      { Run "$py scripts/generate_synthetic_data.py" }
  "preprocess"     { Run "$py scripts/preprocess_dataset.py --dataset synthetic" }
  "classify"       { Run "$py scripts/classify_posts.py --dataset synthetic" }
  "offline-demo"   { Run "$py scripts/run_demo.py --offline" }
  "neo4j-up"       { Run "docker compose --env-file .env up -d"; Write-Host "Browser: http://localhost:7474 (wait ~30-60 s on first start)" }
  "neo4j-down"     { Run "docker compose down" }
  "init"           { Run "$py scripts/initialize_neo4j.py" }
  "ingest"         { Run "$py scripts/ingest_graph.py --dataset synthetic" }
  "gds"            { Run "$py scripts/run_graph_algorithms.py" }
  "communities"    { Run "$py scripts/run_community_analysis.py" }
  "temporal"       { Run "$py scripts/run_temporal_analysis.py --source csv"; Run "$py scripts/run_temporal_analysis.py --source neo4j" }
  "check-llm"      { Run "$py scripts/check_llm.py" }
  "ask"            { Run "$py scripts/ask.py `"$Arg`"" }
  "demo"           { Run "$py scripts/run_demo.py" }
  "test"           { Run "$py -m pytest -q" }
  "test-integration" { $env:RUN_INTEGRATION = "1"; Run "$py -m pytest -q -m integration"; Remove-Item Env:RUN_INTEGRATION }
  "eval" {
    Run "$py scripts/evaluate_text_to_cypher.py --safety-only"
    Run "$py scripts/evaluate_graph_rag.py --routing-only"
  }
  "results"        { Run "$py scripts/run_baselines.py" }
  "clean-synthetic" { Remove-Item data\synthetic\*.csv, data\processed\synthetic -Recurse -ErrorAction SilentlyContinue }
  default { Get-Help $MyInvocation.MyCommand.Path; Write-Host "See the comment block at the top of tasks.ps1 for task names." }
}
