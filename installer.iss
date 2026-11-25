[Setup]
AppName=Blueprint RAG
AppVersion=1.0
DefaultDirName={autopf}\BlueprintRAG
DefaultGroupName=Blueprint RAG
PrivilegesRequired=lowest
OutputBaseFilename=BlueprintRAG-Setup
SetupIconFile=icon.ico
Compression=lzma2
SolidCompression=yes

[Tasks]
Name: desktopicon; Description: "Create a desktop icon"

[Files]
Source: "dist\BlueprintRAG\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Blueprint RAG"; Filename: "{app}\BlueprintRAG.exe"
Name: "{autodesktop}\Blueprint RAG"; Filename: "{app}\BlueprintRAG.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\BlueprintRAG.exe"; Description: "Launch Blueprint RAG"; Flags: postinstall nowait skipifsilent
