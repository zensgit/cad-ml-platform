# DWG to DXF Converters

This guide documents supported converter strategies for DWG -> DXF.

## Selection

Use `DWG_CONVERTER` to choose a strategy:

```
DWG_CONVERTER=auto  # auto|oda|cmd|autocad|bricscad|draftsight
```

- `auto`: prefer ODA if available, otherwise fall back to command templates.
- `oda`: require ODA (fails if missing).
- `cmd`: use `DWG_TO_DXF_CMD`.
- `autocad|bricscad|draftsight`: use the tool-specific command template.

## ODA (recommended)

```
ODA_FILE_CONVERTER_EXE=/path/to/ODAFileConverter
ODA_OUTPUT_VERSION=ACAD2018
```

## Command templates

All command templates accept `{input}` and `{output}` placeholders.

### AutoCAD (Windows)

```
DWG_AUTOCAD_CMD="C:\\Program Files\\Autodesk\\AutoCAD 2024\\accoreconsole.exe /i {input} /o {output} /s dwg2dxf.scr"
```

### BricsCAD (Windows/Linux)

```
DWG_BRICSCAD_CMD="/opt/bricscad/bricscad -b -s dwg2dxf.scr -o {output} {input}"
```

### DraftSight (Windows/Linux)

```
DWG_DRAFTSIGHT_CMD="/opt/draftsight/DSBatch -in {input} -out {output} -format dxf"
```

### Generic command

```
DWG_TO_DXF_CMD="<your-command> {input} {output}"
```

## Notes

- Command syntax varies by installation; test locally before production use.
- Use `DWG_CONVERTER=auto` to prefer ODA but still allow command fallback.
