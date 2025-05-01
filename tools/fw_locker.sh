#!/usr/bin/env bash

function show_help {
    echo "Usage: $0 <camera_serial_number> <firmware_file_1> <firmware_file_2>"
    echo "Updates the firmware of a RealSense camera with the specified serial number."
    echo ""
    echo "Positional arguments:"
    echo "  camera_serial_number  The serial number of the camera to update."
    echo "  firmware_file_1      The path to the first firmware file to use."
    echo "  firmware_file_2      The path to the second firmware file to use."
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help           Show this help message and exit."
}

if [[ "$#" -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

if [ "$#" -ne 3 ]; then
    echo "Error: Invalid number of arguments."
    echo ""
    show_help
    exit 1
fi

echo If at any point this program hangs for more than two minutes, just Ctrl + Z.
echo Update 1
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 2
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 3
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 4
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 5
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 6
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 7
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 8
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 9
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 10
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 11
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 12
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 13
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 14
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 15
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 16
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 17
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 18
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo
echo ========================
echo 
echo Update 19
time timeout 45s rs-fw-update -f "$2" -s "$1"
echo
echo ========================
echo 
echo Update 20
time timeout 45s rs-fw-update -f "$3" -s "$1"
echo 
echo ========================
echo 
echo Camera should be locked by now.
echo Following command should show a D4XX device in recovery mode.
echo If it just shows two normal cameras, simply run this script again and Ctrl + Z after the first update.
rs-enumerate-devices -s
echo Time taken to lock camera:
