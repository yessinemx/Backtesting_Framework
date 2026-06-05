# Compile the Beamer slides to PDF.
# Usage:  .\compile.ps1
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    # Two passes so that the table-of-contents and section links resolve.
    pdflatex -interaction=nonstopmode -halt-on-error presentation.tex | Out-Null
    pdflatex -interaction=nonstopmode -halt-on-error presentation.tex | Out-Null
    Write-Host ("`nPDF: " + (Resolve-Path 'presentation.pdf'))
}
finally {
    Pop-Location
}
