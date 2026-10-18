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
git -C $root archive --format=tar HEAD | tar -x -C $staging

# The Space reads its settings from the YAML header at the top of README.md.
$header = Get-Content (Join-Path $root 'deploy\huggingface\SPACE_HEADER.md') -Raw
$readme = Join-Path $staging 'README.md'
Set-Content -Path $readme -Value ($header + (Get-Content $readme -Raw)) -Encoding utf8 -NoNewline

Push-Location $staging
try {
    git init --quiet --initial-branch=main
    git add -A
    git -c user.name="FitnessCoach deploy" -c user.email="deploy@localhost" commit --quiet -m "Deploy FitnessCoach"
    Write-Host "Pushing to https://huggingface.co/spaces/$Space ..."
    git push --force "https://huggingface.co/spaces/$Space" main
    Write-Host "Done. Build logs and the live app: https://huggingface.co/spaces/$Space"
}
finally {
    Pop-Location
    Remove-Item -Recurse -Force $staging
}
