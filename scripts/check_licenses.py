#!/usr/bin/env python3
"""
License Compliance Checker.

Scans Python dependencies and verifies they match the approved license list.
"""

import argparse
import sys
import pkg_resources
import logging
from typing import List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("license_check")

# Approved Licenses (Permissive)
APPROVED_LICENSES = {
    "MIT",
    "Apache Software License",
    "Apache 2.0",
    "BSD",
    "BSD License",
    "New BSD License",
    "ISC",
    "ISC License (ISCL)",
    "Mozilla Public License 2.0 (MPL 2.0)",
    "Python Software Foundation License",
    "PSF",
    "Unlicense",
    "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"
}

# Copyleft Licenses (Warning/Forbidden depending on policy)
RESTRICTED_LICENSES = {
    "GPL",
    "GPLv2",
    "GPLv3",
    "LGPL",
    "LGPLv2",
    "LGPLv3",
    "AGPL"
}

def get_pkg_license(pkg):
    """Retrieve license for a package."""
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        try:
            lines = pkg.get_metadata_lines('PKG-INFO')
        except:
            return "UNKNOWN"

    for line in lines:
        if line.startswith("License:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("Classifier: License :: OSI Approved ::"):
            return line.split("::")[-1].strip()
            
    return "UNKNOWN"

def check_licenses(fail_on_restricted: bool = False) -> bool:
    """Check installed packages against approved licenses."""
    logger.info("Scanning installed packages...")
    
    issues = []
    unknowns = []
    
    for pkg in pkg_resources.working_set:
        license_name = get_pkg_license(pkg)
        pkg_name = pkg.project_name
        version = pkg.version
        
        # Normalize for checking
        is_approved = False
        for approved in APPROVED_LICENSES:
            if approved.lower() in license_name.lower():
                is_approved = True
                break
                
        if is_approved:
            continue
            
        # Check restricted
        is_restricted = False
        for restricted in RESTRICTED_LICENSES:
            if restricted.lower() in license_name.lower():
                is_restricted = True
                break
                
        if is_restricted:
            issues.append(f"{pkg_name} ({version}): {license_name} (RESTRICTED)")
        else:
            unknowns.append(f"{pkg_name} ({version}): {license_name}")

    # Report
    if issues:
        logger.warning("Found restricted licenses:")
        for issue in issues:
            print(f"  ❌ {issue}")
    
    if unknowns:
        logger.info("Found unknown/unclassified licenses (manual review needed):")
        for unknown in unknowns:
            print(f"  ⚠️  {unknown}")
            
    if not issues and not unknowns:
        logger.info("All packages have approved licenses!")
        return True
        
    if fail_on_restricted and issues:
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Check Python dependency licenses")
    parser.add_argument("--fail-on-restricted", action="store_true", help="Fail if restricted licenses are found")
    args = parser.parse_args()
    
    success = check_licenses(args.fail_on_restricted)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
