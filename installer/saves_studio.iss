; Installer of SAVES Studio for Windows (Inno Setup 6).
; Built by .github/workflows/release-windows.yml after PyInstaller:
;   iscc /DAppVersion=1.0.0 installer/saves_studio.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{2F1A9C4D-7B3E-4C8A-A1D2-9E6F0B5C3A27}
AppName=SAVES Studio
AppVersion={#AppVersion}
AppPublisher=PROFITH, Universidad de Granada
DefaultDirName={autopf}\SAVES Studio
DefaultGroupName=SAVES Studio
UninstallDisplayIcon={app}\SAVES_Studio.exe
OutputDir=..\build\installer
OutputBaseFilename=SAVES_Studio_setup_{#AppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\SAVES_Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SAVES Studio"; Filename: "{app}\SAVES_Studio.exe"
Name: "{autodesktop}\SAVES Studio"; Filename: "{app}\SAVES_Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SAVES_Studio.exe"; Description: "{cm:LaunchProgram,SAVES Studio}"; Flags: nowait postinstall skipifsilent
