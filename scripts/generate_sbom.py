#!/usr/bin/env python3
"""
SBOM Generator (Software Bill of Materials).

Generates a CycloneDX-compatible JSON SBOM for the Python environment.
Tries to use 'syft' if available, otherwise falls back to Python metadata inspection.
"""

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any

try:
    from importlib.metadata import distributions
except ImportError:
    # Fallback for older Python
    import pkg_resources
    def distributions():
        return pkg_resources.working_set

def generate_cyclonedx_python() -> Dict[str, Any]:
    """Generate CycloneDX JSON using Python metadata."""
    print("Generating SBOM using Python metadata...")
    
    components = []
    
    for dist in distributions():
        try:
            # Handle different distribution objects (importlib vs pkg_resources)
            if hasattr(dist, 'metadata'):
                name = dist.metadata['Name']
                version = dist.version
                license_name = dist.metadata.get('License', 'UNKNOWN')
            else:
                name = dist.project_name
                version = dist.version
                try:
                    lines = dist.get_metadata_lines('METADATA')
                    license_name = 'UNKNOWN'
                    for line in lines:
                        if line.startswith('License:'):
                            license_name = line.split(':', 1)[1].strip()
                            break
                except:
                    license_name = 'UNKNOWN'

            purl = f"pkg:pypi/{name}@{version}"
            
            component = {
                "type": "library",
                "name": name,
                "version": version,
                "licenses": [
                    {
                        "license": {
                            "name": license_name
                        }
                    }
                ],
                "purl": purl,
                "bom-ref": purl
            }
            components.append(component)
        except Exception as e:
            print(f"Warning: Could not process package: {e}")

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": [
                {
                    "vendor": "CAD-ML-Platform",
                    "name": "Internal SBOM Generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "name": "cad-ml-platform",
                "version": "1.6.0-dev"
            }
        },
        "components": components
    }
    
    return sbom

def generate_sbom_syft(output_file: str) -> bool:
    """Try to generate SBOM using syft."""
    syft_path = shutil.which("syft")
    if not syft_path:
        return False
        
    print(f"Found syft at {syft_path}, running scan...")
    try:
        subprocess.run(
            [syft_path, ".", "-o", f"cyclonedx-json={output_file}"],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Syft failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate SBOM for the project")
    parser.add_argument("--output", default="sbom.json", help="Output file path")
    parser.add_argument("--force-python", action="store_true", help="Force Python-only generation (skip syft)")
    args = parser.parse_args()

    success = False
    
    # Try syft first unless forced not to
    if not args.force_python:
        success = generate_sbom_syft(args.output)
    
    # Fallback to Python generation
    if not success:
        if not args.force_python:
            print("Syft not found or failed. Falling back to Python metadata inspection.")
        
        sbom_data = generate_cyclonedx_python()
        
        with open(args.output, "w") as f:
            json.dump(sbom_data, f, indent=2)
            
    print(f"SBOM generated at: {args.output}")

if __name__ == "__main__":
    main()
