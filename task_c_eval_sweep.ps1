$Root   = "C:\Users\owner\Documents\PhD\adv_nlp_proj_clpsych_2025"
$SubDir = Join-Path $Root "clpsych2025\evaluation\submissions_dev"
Set-Location $Root

Get-ChildItem (Join-Path $Root "outputs\task_c_*.json") | ForEach-Object {
    if ($_.Name -match '^task_c_(.+)\.json$') {
        $out = Join-Path $SubDir "c_$($Matches[1]).json"
        python .\edit_output_for_official_eval.py --task C `
          --input_json_filepath $_.FullName --output_json_filepath $out
        Write-Host "converted -> $(Split-Path $out -Leaf)"
    }
}

Set-Location (Join-Path $Root "clpsych2025\evaluation")
python run.py --tasks C --team c
Set-Location $Root
