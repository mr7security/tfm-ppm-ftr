# Restaura ftr_cige_v01.dump en el PostgreSQL local (si aún no existe la BBDD)
# y exporta todas las tablas a CSV en la carpeta .\csv
# Uso: powershell -ExecutionPolicy Bypass -File .\exportar_csv.ps1

$ErrorActionPreference = 'Stop'
$dir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$pg    = (Get-ChildItem 'C:\Program Files\PostgreSQL\*\bin' | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$db    = 'ftr_cige_v01'
$dump  = Get-ChildItem "$dir\ftr_cige_v01*.dump" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$out   = "$dir\csv"

if (-not $env:PGPASSWORD) {
    $sec = Read-Host "Contraseña del usuario local 'postgres'" -AsSecureString
    $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}
$env:PGCLIENTENCODING = 'UTF8'

# 1) Restaurar si la BBDD no existe
$existe = & "$pg\psql.exe" -U postgres -d postgres -tAc "select 1 from pg_database where datname='$db'"
if ($existe -ne '1') {
    Write-Host "Creando y restaurando $db desde $($dump.Name)..."
    & "$pg\createdb.exe" -U postgres $db
    & "$pg\pg_restore.exe" -U postgres -d $db --no-owner --no-privileges $dump.FullName
} else {
    Write-Host "La BBDD $db ya existe, se exporta directamente."
}

# 2) Listar tablas (todos los esquemas de usuario)
$tablas = & "$pg\psql.exe" -U postgres -d $db -tAc "select table_schema||'.'||table_name from information_schema.tables where table_type='BASE TABLE' and table_schema not in ('pg_catalog','information_schema') order by 1"
New-Item -ItemType Directory -Force $out | Out-Null

# 3) Exportar cada tabla a CSV (UTF-8, con cabecera, separador ;)
foreach ($t in $tablas) {
    $fichero = "$out\$($t -replace '^public\.','' -replace '\.','__').csv"
    & "$pg\psql.exe" -U postgres -d $db -c "\copy $t to '$fichero' with (format csv, header, delimiter ';', encoding 'UTF8')"
    $filas = (Get-Content $fichero | Measure-Object -Line).Lines - 1
    Write-Host ("{0,-40} {1,8} filas" -f $t, $filas)
}

Remove-Item Env:\PGPASSWORD
Write-Host "`nCSV generados en $out" -ForegroundColor Green
