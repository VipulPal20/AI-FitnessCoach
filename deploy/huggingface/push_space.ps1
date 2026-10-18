# Publish FitnessCoach to a Hugging Face Docker Space.
#
# Usage (from the project root, in PowerShell):
#   .\deploy\huggingface\push_space.ps1 -Space YOUR_HF_USERNAME/fitnesscoach
#
# Pushes a single-commit snapshot of the committed files (HEAD), not the
# repository history: the history contains large deleted datasets that
# Hugging Face would reject. Git asks for your Hugging Face username and an
# access token (with write permission) as the password.

param(
    [Parameter(Mandatory = $true)][string]$Space
)

$ErrorActionPreference = 'Stop'
$root = (git rev-parse --show-toplevel).Trim()
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("fitnesscoach-space-" + [guid]::NewGuid().ToString('N'))

Write-Host "Exporting HEAD of $root ..."
New-Item -ItemType Directory -Path $staging | Out-Null
$tarFile = Join-Path ([System.IO.Path]::GetTempPath()) ("fitnesscoach-" + [guid]::NewGuid().ToString('N') + ".tar")
git -C $root archive --format=tar --output=$tarFile HEAD
tar -x -C $staging -f $tarFile
Remove-Item -Force $tarFile

# The Space reads its settings from the YAML header at the top of README.md.
$header = Get-Content (Join-Path $root 'deploy\huggingface\SPACE_HEADER.md') -Raw
$readme = Join-Path $staging 'README.md'
Set-Content -Path $readme -Value ($header + (Get-Content $readme -Raw)) -Encoding utf8 -NoNewline

Push-Location $staging
try {
    git init --quiet --initial-branch=main
    # Hugging Face refuses binary files in plain git; store them with Git LFS.
    git lfs install --local | Out-Null
    $binary = @('*.png', '*.jpg', '*.jpeg', '*.webp', '*.gif', '*.svg', '*.webm', '*.mp4',
                '*.task', '*.joblib', '*.pkl', '*.ico', '*.woff', '*.woff2', '*.ttf')
    foreach ($pattern in $binary) { git lfs track $pattern | Out-Null }
    git add .gitattributes
    git add -A
    git -c user.name="FitnessCoach deploy" -c user.email="deploy@localhost" commit --quiet -m "Deploy FitnessCoach"
    Write-Host "Pushing to https://huggingface.co/spaces/$Space ..."
    git push --force "https://huggingface.co/spaces/$Space" main
    if ($LASTEXITCODE -ne 0) { throw "Push to Hugging Face failed (see the messages above)." }
    Write-Host "Done. Build logs and the live app: https://huggingface.co/spaces/$Space"
}
finally {
    Pop-Location
    Remove-Item -Recurse -Force $staging
}
