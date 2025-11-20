#!/bin/bash
#
# Pre-commit check script for CAD ML Platform
# Run this before pushing to ensure code quality
#
# Usage:
#   ./scripts/pre_commit_check.sh
#
# Or install as Git hook:
#   ln -s ../../scripts/pre_commit_check.sh .git/hooks/pre-commit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Running pre-commit checks for CAD ML Platform...${NC}"
echo "================================================"

# Function to check if a command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 failed${NC}"
        return 1
    fi
}

# Track overall status
FAILED=0

# 1. Check for unstaged changes
echo -e "\n${YELLOW}Step 1/5: Checking for unstaged changes...${NC}"
if ! git diff --quiet; then
    echo -e "${YELLOW}⚠ Warning: You have unstaged changes. These won't be committed.${NC}"
    echo "  Consider staging them with: git add -A"
fi

# 2. Run soft validation
echo -e "\n${YELLOW}Step 2/5: Running evaluation validation (non-blocking)...${NC}"
if command -v make &> /dev/null; then
    make eval-validate-soft 2>/dev/null || true
else
    echo -e "${YELLOW}⚠ Make not available, skipping validation${NC}"
fi

# 3. Check Python syntax
echo -e "\n${YELLOW}Step 3/5: Checking Python syntax...${NC}"
if command -v python3 &> /dev/null; then
    python3 -m py_compile scripts/*.py 2>/dev/null
    check_status "Python syntax check" || FAILED=1
else
    echo -e "${YELLOW}⚠ Python3 not available, skipping syntax check${NC}"
fi

# 4. Check JSON files
echo -e "\n${YELLOW}Step 4/5: Validating JSON files...${NC}"
for json_file in config/*.json docs/*.schema.json; do
    if [ -f "$json_file" ]; then
        python3 -m json.tool "$json_file" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓ $json_file is valid${NC}"
        else
            echo -e "  ${RED}✗ $json_file has invalid JSON${NC}"
            FAILED=1
        fi
    fi
done

# 5. Check for large files
echo -e "\n${YELLOW}Step 5/5: Checking for large files...${NC}"
LARGE_FILES=$(find . -type f -size +1M -not -path "./.git/*" -not -path "./reports/*" 2>/dev/null)
if [ -n "$LARGE_FILES" ]; then
    echo -e "${YELLOW}⚠ Large files detected (>1MB):${NC}"
    echo "$LARGE_FILES" | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "  - $file ($size)"
    done
    echo -e "${YELLOW}  Consider using Git LFS for large files${NC}"
fi

# Summary
echo ""
echo "================================================"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All pre-commit checks passed!${NC}"
    echo -e "${GREEN}Ready to commit and push.${NC}"
else
    echo -e "${RED}❌ Some checks failed!${NC}"
    echo -e "${YELLOW}Fix the issues above before pushing.${NC}"
    echo -e "${YELLOW}You can still commit locally, but CI may fail.${NC}"
fi
echo "================================================"

# Exit with appropriate code
# For pre-commit hook, we use exit 0 to allow commit but warn
# For manual run, we show the actual status
if [ -n "$GIT_DIR" ]; then
    # Running as git hook - allow commit but warn
    exit 0
else
    # Running manually - show actual status
    exit $FAILED
fi