param(
  [Parameter(Mandatory = $true)]
  [string]$ConfigPath,

  [Parameter(Mandatory = $true)]
  [string]$BaseUrl,

  [Parameter(Mandatory = $true)]
  [string]$RefAudioPath,

  [Parameter(Mandatory = $true)]
  [string]$PromptTextBase64
)

$promptText = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($PromptTextBase64))
$refAudioToml = $RefAudioPath.Replace('\', '\\')

$lines = @(
  '[tts]'
  'enabled = true'
  'provider = "gpt_sovits"'
  'voice = "shigeju"'
  'rate = "+0%"'
  'volume = "+0%"'
  'proxy = ""'
  "base_url = `"$BaseUrl`""
  "ref_audio_path = `"$refAudioToml`""
  "prompt_text = `"$promptText`""
  'prompt_lang = "zh"'
  'text_lang = "zh"'
  'text_split_method = "cut5"'
  'media_type = "wav"'
  'timeout_seconds = 120.0'
  'max_chars = 300'
  ''
)

$block = ($lines -join "`r`n") + "`r`n"
$content = Get-Content -Raw -Encoding UTF8 $ConfigPath
$sectionPattern = '(?ms)^\[tts\].*?(?=^\[|\z)'

if ($content -match $sectionPattern) {
  $content = [regex]::Replace($content, $sectionPattern, $block, 1)
} else {
  $content = $content.TrimEnd() + "`r`n`r`n" + $block
}

Set-Content -Encoding UTF8 -NoNewline -Path $ConfigPath -Value $content
