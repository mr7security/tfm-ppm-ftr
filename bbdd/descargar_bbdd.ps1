# Descarga (pg_dump) de la base de datos definida en credencialesVALENFFTR.json
# Uso: abre PowerShell en C:\Users\MR7\Downloads\BBDD y ejecuta:
#   powershell -ExecutionPolicy Bypass -File .\descargar_bbdd.ps1

$ErrorActionPreference = 'Stop'
$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$cred = Get-Content "$dir\credencialesVALENFFTR.json" -Raw | ConvertFrom-Json

$port = if ($cred.port) { $cred.port } else { 5432 }
$fecha = Get-Date -Format 'yyyyMMdd_HHmm'
$salida = "$dir\$($cred.dbname)_$fecha.dump"

# Localizar pg_dump (PATH o instalación típica de PostgreSQL)
$pgdump = Get-Command pg_dump -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $pgdump) {
    $pgdump = Get-ChildItem 'C:\Program Files\PostgreSQL\*\bin\pg_dump.exe' -ErrorAction SilentlyContinue |
              Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $pgdump) {
    Write-Host "No se encuentra pg_dump. Instala el cliente con:  winget install PostgreSQL.PostgreSQL" -ForegroundColor Red
    exit 1
}

Write-Host "Servidor : $($cred.host):$port"
Write-Host "BBDD     : $($cred.dbname)"
Write-Host "Usuario  : $($cred.user)"
Write-Host "Salida   : $salida"
Write-Host ""

# Comprobar conectividad al puerto
if (-not (Test-NetConnection $cred.host -Port $port -InformationLevel Quiet -WarningAction SilentlyContinue)) {
    Write-Host "No hay conexión con $($cred.host):$port. Si estás fuera de la UJI, conéctate a la VPN y vuelve a ejecutar." -ForegroundColor Yellow
    exit 1
}

$env:PGPASSWORD = $cred.password
& $pgdump -h $cred.host -p $port -U $cred.user -d $cred.dbname -Fc -v -f $salida
$rc = $LASTEXITCODE
Remove-Item Env:\PGPASSWORD

if ($rc -eq 0) {
    $mb = [math]::Round((Get-Item $salida).Length / 1MB, 2)
    Write-Host "`nListo: $salida ($mb MB)" -ForegroundColor Green
    Write-Host "Restaurar en local:  createdb $($cred.dbname) ; pg_restore -d $($cred.dbname) `"$salida`""
} else {
    Write-Host "`npg_dump terminó con error (código $rc)." -ForegroundColor Red
}
