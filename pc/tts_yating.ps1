# 用 Windows 內建的中文語音（Microsoft Yating，台灣口音女聲）把一段文字轉成 wav。離線產生，不會上傳。
#   powershell -ExecutionPolicy Bypass -File pc\tts_yating.ps1 -TextFile in.txt -Out out.wav [-Rate 1.0]
param([string]$TextFile, [string]$Out, [double]$Rate = 1.0, [string]$Voice = "Yating")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq "IAsyncOperation``1" } | Select-Object -First 1
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result }

$s = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$v = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -like "*$Voice*" } | Select-Object -First 1
if ($null -eq $v) { throw "找不到語音 $Voice" }
$s.Voice = $v
$s.Options.SpeakingRate = $Rate
$text = [System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8)
$stream = Await ($s.SynthesizeTextToStreamAsync($text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
$n = [uint32]$stream.Size
$null = Await ($reader.LoadAsync($n)) ([uint32])
$bytes = New-Object byte[] $n
$reader.ReadBytes($bytes)
[System.IO.File]::WriteAllBytes($Out, $bytes)
