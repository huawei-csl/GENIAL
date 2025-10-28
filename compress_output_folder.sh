#!/bin/bash

# Function to display usage information
usage() {
  echo "Usage: $0 [--uncompress] <source_directory|compressed_file> <target_file|target_directory>"
  exit 1
}

# Check if pigz is installed
if ! command -v pigz &> /dev/null; then
  echo "Error: pigz is not installed. Install it with 'sudo apt-get install pigz' or 'sudo yum install pigz'."
  exit 1
fi

# Check if pv is installed
if ! command -v pv &> /dev/null; then
  echo "Error: pv is not installed. Install it with 'sudo apt-get install pv' or 'sudo yum install pv'."
  exit 1
fi

# Parse the --uncompress option
UNCOMPRESS=false
if [ "$1" == "--uncompress" ]; then
  UNCOMPRESS=true
  shift
fi

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  usage
fi

# Get the source and target files from arguments
SOURCE=$1
TARGET=$2

# Uncompress if the --uncompress option is set
if [ "$UNCOMPRESS" = true ]; then
  # Check if the source file exists
  if [ ! -f "$SOURCE" ]; then
    echo "Error: Source compressed file '$SOURCE' does not exist."
    exit 1
  fi

  # Get the uncompressed file size for progress calculation
  FILE_SIZE=$(gzip -l "$SOURCE" | awk 'NR==2 {print $2}')
  DOUBLED_SIZE=$((FILE_SIZE * 2)) # Because for some reason, uncompressed folder has a doubled size than what is initially expected

  # Uncompress the source file into the target directory
  echo "Uncompressing '$SOURCE' into '$TARGET' using pigz..."
  mkdir -p "$TARGET" # Create the target directory if it doesn't exist
  pv -s "$DOUBLED_SIZE" "$SOURCE" | pigz -d | tar -xvf - -C "$TARGET" > /dev/null

  # Check if the uncompression was successful
  if [ $? -eq 0 ]; then
    echo "Uncompression successful. Files extracted to: $TARGET"
  else
    echo "Error: Uncompression failed."
    exit 1
  fi

else
  # Check if source directory exists
  if [ ! -d "$SOURCE" ]; then
    echo "Error: Source directory '$SOURCE' does not exist."
    exit 1
  fi

  # Estimate the total size of the source directory using a dry run with tar
  echo "Calculating the size of files to be compressed..."
  TOTAL_SIZE=$(tar -cf - "$SOURCE" | wc -c)

  # Compress the source directory into the target file using pigz
  echo "Compressing '$SOURCE' into '$TARGET' using pigz..."
  tar -cf - "$SOURCE" | pv -s "$TOTAL_SIZE" | pigz -p 64 > "$TARGET"

  # Check if the compression was successful
  if [ $? -eq 0 ]; then
    echo "Compression successful. Output file: $TARGET"
  else
    echo "Error: Compression failed."
    exit 1
  fi
fi
