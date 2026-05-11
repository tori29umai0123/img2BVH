; Inno Setup script for image2BVH.
;
; Build with:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer\image2bvh.iss
; or via packaging\build.ps1 (preferred — it also runs PyInstaller first and
; assembles the concatenated license bundle).
;
; Output: packaging\installer\Output\image2bvh-setup-<version>.exe

#define MyAppName "image2BVH"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "image2BVH project"
#define MyAppExeName "image2bvh.exe"
; Stable AppId — change ONLY when introducing an intentionally separate install
; (e.g. a side-by-side major version). Reusing it across builds is what allows
; an upgrade installer to detect & replace the previous install.
#define MyAppId "{{F8E83E11-7B4D-4C5A-9E2B-IMAGE2BVH001}}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppSupportURL=https://github.com/anthropics/claude-code/issues
VersionInfoVersion={#MyAppVersion}

; Install per-user under %LOCALAPPDATA%\Programs so we DON'T need admin
; privileges. This also gives image2bvh.exe a writable install dir for
; tmp/, config.ini, triton-cache/, hf-cache/, runtime/mhr_rest.json.
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; LICENSE_BUNDLE.txt is assembled by build.ps1 from SAM_LICENSE.txt +
; DINOv3_LICENSE.md + the top-level LICENSE. Required by SAM License §1.b.i
; / DINOv3 License §1.b.i (licensee must provide a copy of the license with
; any redistribution).
LicenseFile=..\..\dist\LICENSE_BUNDLE.txt

; Wizard
WizardStyle=modern
ShowLanguageDialog=yes
SetupLogging=yes

; Compression — installer is large (~6-9 GB), so squeeze hard once.
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

; Disk spanning is REQUIRED because the compressed installer payload is
; larger than ~4.2 GB (Windows imposes that as a practical maximum for a
; single Setup.exe). With disk spanning enabled, ISCC emits:
;   image2bvh-setup-<ver>.exe   (small launcher, ~5-10 MB)
;   image2bvh-setup-<ver>-1.bin (data slice 1)
;   image2bvh-setup-<ver>-2.bin (data slice 2)
;   ...
; The end-user must keep all files in the same directory; Setup.exe finds
; the .bin slices by name. Slice size below: 2.0 GB each → ~3 slices.
DiskSpanning=yes
DiskSliceSize=2100000000
SlicesPerDisk=1

; Disk space — torch nightly + triton + bundled SAM3/SAM3DBody runs ~9 GB
; on disk; pad a bit so the installer warns before extraction fails.
ExtraDiskSpaceRequired=10737418240

; 64-bit only — torch cu130 wheels are x64 Windows.
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; Where the .exe lands relative to this .iss file.
OutputDir=Output
OutputBaseFilename=image2bvh-setup-{#MyAppVersion}

UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Pull the entire PyInstaller --onedir tree.
Source: "..\..\dist\image2bvh\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Also keep a copy of each underlying license file at the install root for
; easy discovery by end users.
Source: "..\..\LICENSES\SAM_LICENSE.txt"; DestDir: "{app}\LICENSES"; Flags: ignoreversion
Source: "..\..\LICENSES\DINOv3_LICENSE.md"; DestDir: "{app}\LICENSES"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Offer to launch immediately after install completes. Gradio opens a browser
; tab on http://127.0.0.1:7860 once it's ready.
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Wipe runtime state that isn't tracked by [Files]:
;   tmp/             — per-run BVH scratch
;   triton-cache/    — Triton JIT cache populated by runtime hook
;   hf-cache/        — HF re-downloads if user deleted a bundled weight
;   runtime/mhr_rest.json — baked rest skeleton (regenerated on first run)
Type: filesandordirs; Name: "{app}\tmp"
Type: filesandordirs; Name: "{app}\triton-cache"
Type: filesandordirs; Name: "{app}\hf-cache"
Type: files; Name: "{app}\runtime\mhr_rest.json"
Type: files; Name: "{app}\config.ini"
