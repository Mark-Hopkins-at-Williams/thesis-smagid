#!/bin/bash

echo "Running npm build..."

npm run build

# Check if build was successful
if [ $? -eq 0 ]; then
  echo "Build completed successfully!"

  # Remove leading '/' from paths in dist/index.html
  echo "Removing leading '/' from paths in dist/index.html..."
  sed -i.bak 's|/assets|assets|g; s|/vite.svg|vite.svg|g' dist/index.html
  rm dist/index.html.bak
  echo "Paths updated successfully!"

  # Remove 'src/' from specific paths in JS files in dist/assets
  echo "Updating JavaScript files in dist/assets..."
  for js_file in dist/assets/*.js; do
    sed -i.bak 's|src:"src/assets/shuffle.svg"|src:"assets/shuffle.svg"|g; s|src:"src/assets/back.svg"|src:"assets/back.svg"|g' "$js_file"
    rm "$js_file.bak"  # Remove backup file created by sed
  done
  echo "JavaScript files updated successfully!"

  # copy all asset files
  echo "Copying assets from src/assets to dist/assets..."
  cp -R src/assets/* dist/assets/
  echo "Assets copied successfully!"

  # Move old public_html to backup_html on remote server
  echo "Backing up old deployment on remote server..."
  ssh 25sm39@sysnet.cs.williams.edu 'rm -rf ~/backup_html && mv ~/public_html ~/backup_html'
  if [ $? -eq 0 ]; then
    echo "Backup completed successfully!"
  else
    echo "Failed to create backup on remote server. Deployment aborted."
    exit 1
  fi

  # SCP the dist folder to the remote server
  echo "Copying the dist folder to the remote server..."
  scp -r dist/ 25sm39@sysnet.cs.williams.edu:~/public_html

  if [ $? -eq 0 ]; then
    echo "Files copied to the remote server successfully!"
  else
    echo "Failed to copy files to the remote server."
    exit 1
  fi
else
  echo "Build failed. Check the errors above."
  exit 1
fi

