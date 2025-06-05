#!/bin/bash

# Check if version argument is provided
if [ -z "$1" ]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <version>"
    echo "Example: $0 1.2.4"
    exit 1
fi

export VERSION=$1

# Validate version format (basic semver check)
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: Version must be in semver format (e.g., 1.2.3)"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Get list of changed files for commit message
CHANGED_FILES=$(git diff --name-only HEAD 2>/dev/null || echo "")
STAGED_FILES=$(git diff --cached --name-only 2>/dev/null || echo "")

# Combine changed and staged files
ALL_CHANGED_FILES="$CHANGED_FILES $STAGED_FILES"
ALL_CHANGED_FILES=$(echo "$ALL_CHANGED_FILES" | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/[[:space:]]*$//')

# Update package.json version
echo "Updating package.json version to $VERSION..."
npm version "$VERSION" --no-git-tag-version || {
    echo "Error: Failed to update package.json version"
    exit 1
}

# Add all changes
echo "Adding changes to git..."
git add . || {
    echo "Error: Failed to add changes to git"
    exit 1
}

# Create detailed commit message
if [ -n "$ALL_CHANGED_FILES" ]; then
    COMMIT_MSG="v$VERSION

Changed files:
$(echo "$ALL_CHANGED_FILES" | tr ' ' '\n' | sed 's/^/- /' | grep -v '^- $')"
else
    COMMIT_MSG="v$VERSION"
fi

# Commit with version and file list
echo "Committing version $VERSION..."
git commit -m "$COMMIT_MSG" || {
    echo "Error: Failed to commit changes"
    exit 1
}

# Create git tag
echo "Creating git tag $VERSION..."
git tag "$VERSION" || {
    echo "Error: Failed to create git tag"
    exit 1
}

# Push commits
echo "Pushing commits..."
git push || {
    echo "Error: Failed to push commits"
    exit 1
}

# Push tags
echo "Pushing tags..."
git push origin "$VERSION" || {
    echo "Error: Failed to push tags"
    exit 1
}

echo "Publishing to NPM..."

npm publish

echo "Successfully published version $VERSION!"