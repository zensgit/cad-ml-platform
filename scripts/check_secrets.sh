#!/bin/bash
# Wrapper script for Gitleaks to check for secrets in the repository

# Check if gitleaks is installed
if ! command -v gitleaks &> /dev/null; then
    echo "⚠️  gitleaks is not installed."
    echo "Please install it to scan for secrets:"
    echo "  brew install gitleaks  # macOS"
    echo "  or visit https://github.com/gitleaks/gitleaks"
    
    # In CI, we might want to fail or download it. 
    # For now, we exit 0 with a warning to avoid breaking local dev if not installed.
    if [ "$CI" = "true" ]; then
        echo "❌ CI environment detected. Gitleaks is required."
        exit 1
    fi
    exit 0
fi

echo "🔒 Running Gitleaks..."
gitleaks detect --source . -v

if [ $? -eq 0 ]; then
    echo "✅ No secrets detected."
else
    echo "❌ Secrets detected! Please remove them before committing."
    exit 1
fi
