$DownloadUrl = "https://www.kaggle.com/api/v1/datasets/download/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
$OutputFile = "archive.zip"
$ExtractPath = (Get-Location).Path + "\Data"
Write-Host "START DOWNLOADING..."
Invoke-WebRequest -Uri $DownloadUrl -OutFile $OutputFile
Write-Host "Done 👍"

# Unzip the downloaded file
Write-Host "Unzipping the archive..."
Expand-Archive -Path $OutputFile -DestinationPath $ExtractPath -Force
Write-Host "Unzip completed! Files extracted to $ExtractPath."

Remove-Item $OutputFile