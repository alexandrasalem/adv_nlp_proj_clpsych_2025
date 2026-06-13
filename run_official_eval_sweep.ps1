$ErrorActionPreference = "Stop"
$Root    = "C:\Users\owner\Documents\PhD\adv_nlp_proj_clpsych_2025"
$EvalDir = Join-Path $Root "clpsych2025\evaluation"
$SubDir  = Join-Path $EvalDir "submissions_dev"
$ResDir  = Join-Path $EvalDir "results"

New-Item -ItemType Directory -Force -Path $SubDir, $ResDir | Out-Null

# ── Convert all 15 one-shot sweep files ──
Set-Location $Root
$pattern = '^task_b_results_(predicted|gold|random)_(.+)_\d{8}-\d{6}\.json$'
Get-ChildItem (Join-Path $Root "outputs\task_b_results_*.json") | ForEach-Object {
    if ($_.Name -match $pattern) {
        $label = $Matches[1]
        $tag   = $Matches[2]
        $out   = Join-Path $SubDir "${label}_${tag}.json"
        python .\edit_output_for_official_eval.py --task B --model_name one_shot_summary `
          --input_json_filepath $_.FullName --output_json_filepath $out
        Write-Host "one-shot  -> $(Split-Path $out -Leaf)"
    }
}

# ── Convert zero-shot: one representative file per label source ──
# (zero-shot ignores retrieval, so it's identical across the 5 configs of a label)
foreach ($label in @("predicted","gold","random")) {
    $rep = Get-ChildItem (Join-Path $Root "outputs\task_b_results_${label}_hybrid-a0.25_*.json") |
           Select-Object -First 1
    if ($rep) {
        $out = Join-Path $SubDir "${label}_zeroshot.json"
        python .\edit_output_for_official_eval.py --task B --model_name zero_shot_summary `
          --input_json_filepath $rep.FullName --output_json_filepath $out
        Write-Host "zero-shot -> $(Split-Path $out -Leaf)"
    }
}

# ── Score everything in submissions_dev in one pass ──
Set-Location $EvalDir
python run.py --tasks B
Set-Location $Root
